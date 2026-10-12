"""Results page + registry API for EmbeddingsBench. There is no authentication: bind it to a private network only.

Static files come from the site directory (pre-gzipped *.gz served with Content-Encoding). The /api routes
manage splits, model versions and the job queue; one worker thread runs queued jobs one at a time as
`python jobs.py run <id>` subprocesses, so GPU work never overlaps inside the registry.

usage: python app.py [<port> <site-dir> <code-dir>]     defaults from EB_PORT, EB_BIND, EB_SITE_DIR, EB_CODE_DIR
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
if len(sys.argv) == 4:
    PORT, SITE, CODE = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
else:
    PORT = int(os.getenv("EB_PORT", "8095"))
    SITE = Path(os.getenv("EB_SITE_DIR") or HERE.parent / "site")
    CODE = Path(os.getenv("EB_CODE_DIR") or HERE.parent / "bench")
BIND = os.getenv("EB_BIND", "127.0.0.1")
sys.path.insert(0, str(CODE))
import registry  # noqa: E402

RUNNING = registry.RUNNING  # job id -> Popen (shared with the registry worker)
UPLOAD_LIMIT = 50_000_000  # bytes per dataset upload (JSON body with the file content)
_set_job, _mark_cancelled, _tail = registry.set_job, registry.mark_cancelled, registry.tail_log


def state():
    registry.load_custom_specs()  # pick up datasets uploaded since start-up
    datasets = []
    for name, spec in registry.SPECS.items():
        custom = registry.DATASETS / name / "dataset.json"
        datasets.append({"name": name, "title": spec["title"], "official_val": registry.official_val_split(name),
                         "splits": registry.list_splits(name),
                         **({"custom": json.loads(custom.read_text())} if spec.get("custom") and custom.exists() else {})})
    models = registry.list_models()
    for m in models:  # keep the list light; details come with /api/models/<name>/<v>
        if "test" in m:
            m["test"] = {k: m["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "mcc") if k in m["test"]}
        m.pop("selection", None); m.pop("history", None)
    return {"datasets": datasets, "models": models, "jobs": registry.list_jobs(100),
            "methods": list(registry.METHODS), "param_types": {k: {p: t.__name__ for p, t in v.items()} for k, v in registry.PARAM_TYPES.items()},
            "stages": list(registry.STAGES), "method_info": registry.METHOD_INFO,
            "dataset_formats": list(registry.DATASET_FORMATS)}


class Handler(SimpleHTTPRequestHandler):
    # ---------------------------------------------------------- static
    def send_head(self):
        path = self.translate_path(self.path)
        gz = path + ".gz"
        if "gzip" in self.headers.get("Accept-Encoding", "") and os.path.isfile(gz):
            f = open(gz, "rb")
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(os.fstat(f.fileno()).st_size))
            self.end_headers()
            return f
        return super().send_head()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args):
        pass

    # ---------------------------------------------------------- api
    def _json(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self, limit=100_000):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError(f"request too large (limit {limit // 1_000_000 or 0.1} MB)")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if not self.path.startswith("/api/"):
            return super().do_GET()
        try:
            if self.path == "/api/state":
                return self._json(200, state())
            m = re.fullmatch(r"/api/models/([a-z0-9_-]+)/(\d+)", self.path)
            if m:
                return self._json(200, json.loads((registry.model_dir(m[1], int(m[2])) / "model.json").read_text()))
            m = re.fullmatch(r"/api/models/([a-z0-9_-]+)/(\d+)/predictions", self.path)
            if m:
                p = registry.model_dir(m[1], int(m[2])) / "predictions.json.gz"
                raw = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                return self.wfile.write(raw)
            m = re.fullmatch(r"/api/jobs/([0-9a-f-]+)/log", self.path)
            if m:
                return self._json(200, {"log": _tail(m[1])})
            return self._json(404, {"error": "not found"})
        except FileNotFoundError:
            return self._json(404, {"error": "not found"})
        except Exception as e:
            return self._json(400, {"error": str(e)})

    def do_POST(self):
        try:
            b = self._body(UPLOAD_LIMIT if self.path == "/api/datasets" else 100_000)
            if self.path == "/api/datasets":
                fmt = b.get("format")
                if fmt not in registry.DATASET_FORMATS:
                    raise ValueError(f"format must be one of {registry.DATASET_FORMATS}")
                uploads = registry.DATASETS / "_uploads"
                uploads.mkdir(parents=True, exist_ok=True)
                src = uploads / f"{re.sub(r'[^a-z0-9_-]', '_', str(b.get('name', 'upload')))[:40]}.{fmt}"
                src.write_text(b.get("content", ""))
                meta, job = registry.reserve_dataset(b.get("name", ""), src, fmt, b.get("title", ""), b.get("lang", "en"),
                                                     b.get("domain", ""), b.get("description", ""), bool(b.get("benchmark")))
                return self._json(200, {"dataset": meta, "job": job})
            if self.path == "/api/splits":
                v = b.get("val") or {}
                split_id, job = registry.reserve_split(b.get("dataset"), b.get("title"), b.get("description"),
                                                       b.get("train_per_class"), b.get("test_per_class"),
                                                       (v.get("mode"), v.get("value")), int(b.get("seed", 13)))
                return self._json(200, {"split": split_id, "job": job})
            if self.path == "/api/models":
                m, job = registry.reserve_model(b.get("name", ""), b.get("dataset"), b.get("split"), b.get("method"),
                                                b.get("params") or {}, b.get("description", ""), b.get("tags") or [])
                return self._json(200, {"model": m, "job": job})
            m = re.fullmatch(r"/api/models/([a-z0-9_-]+)/(\d+)/retrain", self.path)
            if m:
                prev = json.loads((registry.model_dir(m[1], int(m[2])) / "model.json").read_text())
                new, job = registry.reserve_model(m[1], prev["dataset"], b.get("split") or prev["split"], prev["method"],
                                                  b["params"] if "params" in b else prev.get("params"),
                                                  b.get("description", prev.get("description", "")), prev.get("tags", []),
                                                  parent=f"v{m[2]}")
                return self._json(200, {"model": new, "job": job})
            m = re.fullmatch(r"/api/models/([a-z0-9_-]+)/(\d+)/describe", self.path)
            if m:
                return self._json(200, registry.describe_model(m[1], int(m[2]), b.get("description"), b.get("tags"), b.get("stage")))
            m = re.fullmatch(r"/api/jobs/([0-9a-f-]+)/cancel", self.path)
            if m:
                job = json.loads((registry.JOBS / f"{m[1]}.json").read_text())
                if job["status"] == "queued":
                    _set_job(m[1], status="cancelled", finished=registry.now())
                    _mark_cancelled(job)
                elif job["status"] == "running" and m[1] in RUNNING:
                    _set_job(m[1], status="cancelled", finished=registry.now())
                    os.killpg(RUNNING[m[1]].pid, signal.SIGTERM)
                else:
                    raise ValueError(f"job is {job['status']}")
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "not found"})
        except FileNotFoundError:
            return self._json(404, {"error": "not found"})
        except Exception as e:
            return self._json(400, {"error": str(e)})


if __name__ == "__main__":
    threading.Thread(target=registry.worker, daemon=True).start()
    ThreadingHTTPServer((BIND, PORT), partial(Handler, directory=str(SITE))).serve_forever()
