"""Inference service: predictions from registry models, plus an embedding endpoint for remote clients.

  GET  /health                 device and model information
  GET  /v1/models              ready registry models (name:version, method, dataset, input type, test accuracy)
  POST /v1/predict             {"model": "name[:version]", "embeddings": [[1024 floats], …] | "texts": ["…"],
                                "top_k": 3, "distribution": false}
                               → per item: label, label_id, index, confidence, top_k [, distribution]
  GET  /openapi.yaml           OpenAPI 3 description (clients/openapi.yaml), e.g. for NSwag / Kiota client generation
  POST /v1/embeddings          OpenAI-compatible: {"input": ["…"], "max_tokens": 2048} → {"data": [{"index", "embedding"}]}
                               (the jina v5 classification encoder with the "Document: " prompt, L2-normalised)

Embeddings must come from jinaai/jina-embeddings-v5-text-small-classification with the "Document: " prompt (as
returned by /v1/embeddings). Set EB_API_KEY to require "Authorization: Bearer <key>". Bind with EB_INFER_BIND
(default 127.0.0.1) and EB_INFER_PORT (default 8096). With EB_EMBED_URL the service itself embeds remotely.

usage: python web/inference.py
"""
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv("EB_CODE_DIR") or HERE.parent / "bench")))
import gpu  # noqa: E402
import inference  # noqa: E402

API_KEY = os.getenv("EB_API_KEY", "")
MAX_BODY = 20_000_000


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorised(self):
        if API_KEY and self.headers.get("Authorization", "") != f"Bearer {API_KEY}":
            self._json(401, {"error": "missing or wrong bearer token"})
            return False
        return True

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        if not self._authorised():
            return
        if self.path == "/health":
            return self._json(200, {"status": "ok", "device": gpu.describe(), "embed_url": os.getenv("EB_EMBED_URL") or None})
        if self.path == "/v1/models":
            return self._json(200, {"models": inference.models()})
        if self.path == "/openapi.yaml":
            raw = (HERE.parent / "clients" / "openapi.yaml").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/yaml")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            return self.wfile.write(raw)
        self._json(404, {"error": "not found"})

    def _read_body(self):
        """Request body with Content-Length or chunked transfer encoding (as sent by e.g. .NET JsonContent)."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            chunks, total = [], 0
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return b"".join(chunks)
                total += size
                if total > MAX_BODY:
                    return None
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return None if n > MAX_BODY else self.rfile.read(n)

    def do_POST(self):
        if not self._authorised():
            return
        try:
            raw = self._read_body()
            if raw is None:
                return self._json(413, {"error": f"request too large (limit {MAX_BODY // 1_000_000} MB)"})
            body = json.loads(raw or b"{}")
            if self.path == "/v1/predict":
                return self._json(200, inference.predict(body["model"], body.get("embeddings"), body.get("texts"),
                                                         int(body.get("top_k", 3)), bool(body.get("distribution", False))))
            if self.path == "/v1/embeddings":
                texts = body.get("input")
                texts = [texts] if isinstance(texts, str) else texts
                if not texts or not all(isinstance(t, str) for t in texts):
                    raise ValueError("input must be a string or a list of strings")
                from embed import MODEL_ID
                with inference._LOCK:
                    vecs = _embed(texts, int(body.get("max_tokens", 2048)))
                return self._json(200, {"object": "list", "model": MODEL_ID,
                                        "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)]})
            self._json(404, {"error": "not found"})
        except inference.ModelNotFound as e:
            self._json(404, {"error": str(e)})
        except KeyError as e:
            self._json(400, {"error": f"missing field {e}"})
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def _embed(texts, max_tokens):
    from embed import encode, load_encoder, token_lengths
    if inference._ENCODER is None:
        inference._ENCODER = load_encoder()
    enc = inference._ENCODER
    return encode(enc, texts, token_lengths(enc, texts), max_tokens)


if __name__ == "__main__":
    bind, port = os.getenv("EB_INFER_BIND", "127.0.0.1"), int(os.getenv("EB_INFER_PORT", "8096"))
    print(f"inference service on http://{bind}:{port} ({gpu.describe()})", flush=True)
    ThreadingHTTPServer((bind, port), Handler).serve_forever()
