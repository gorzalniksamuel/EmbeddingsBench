"""Manual control of the model/data registry (the web app's Models and Data tabs use the same functions).

  python jobs.py list                                  splits, models and jobs
  python jobs.py split <dataset> [--title T] [--train-per-class N] [--test-per-class N]
                       [--val official | holdout:200 | holdout:0.15] [--seed S] [--description D]
  python jobs.py train <name> --dataset D --split S --method M [--params JSON] [--description D] [--tags a,b]
  python jobs.py retrain <name> <version> [--split S] [--params JSON] [--description D]
  python jobs.py describe <name> <version> [--description D] [--tags a,b] [--stage candidate|production|archived|none]
  python jobs.py add-dataset <name> <file.csv|file.jsonl> [--title T] [--lang de] [--domain D] [--description D] [--benchmark]
                                                       upload format: columns text, label (optional split = train/validation/test)
  python jobs.py register-benchmark                    add saved benchmark fine-tunes (models/<dataset>…) to the registry
  python jobs.py run <job-id>                          execute a queued job (normally done by the web app worker)
  python jobs.py worker                                run queued jobs without the web app (headless GPU box);
                                                       with EB_REMOTE_HOST set, jobs run on that GPU server (remote.py)

New model types: add a module to bench/classifiers/ (see classifiers/__init__.py); it appears as --method.

Jobs are queued in registry/jobs and executed one at a time by the web service.
Without --params, every setting is chosen on the validation split.
"""
import argparse
import json
import sys

import registry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("split")
    s.add_argument("dataset"); s.add_argument("--title", default="")
    s.add_argument("--train-per-class", type=int); s.add_argument("--test-per-class", type=int)
    s.add_argument("--val", default="holdout:0.15"); s.add_argument("--seed", type=int, default=13)
    s.add_argument("--description", default="")
    t = sub.add_parser("train")
    t.add_argument("name"); t.add_argument("--dataset", required=True); t.add_argument("--split", default="benchmark")
    t.add_argument("--method", required=True, choices=registry.METHODS); t.add_argument("--params", default="{}")
    t.add_argument("--description", default=""); t.add_argument("--tags", default="")
    rt = sub.add_parser("retrain")
    rt.add_argument("name"); rt.add_argument("version", type=int); rt.add_argument("--split")
    rt.add_argument("--params"); rt.add_argument("--description")
    d = sub.add_parser("describe")
    d.add_argument("name"); d.add_argument("version", type=int); d.add_argument("--description")
    d.add_argument("--tags"); d.add_argument("--stage")
    ad = sub.add_parser("add-dataset")
    ad.add_argument("name"); ad.add_argument("file"); ad.add_argument("--title", default=""); ad.add_argument("--lang", default="en")
    ad.add_argument("--domain", default=""); ad.add_argument("--description", default="")
    ad.add_argument("--benchmark", action="store_true", help="also run the full benchmark and add it to the overview")
    sub.add_parser("register-benchmark")
    r = sub.add_parser("run"); r.add_argument("job_id")
    sub.add_parser("worker")
    a = ap.parse_args()

    if a.cmd == "list":
        for ds in registry.SPECS:
            for sp in registry.list_splits(ds):
                st = sp.get("stats", {})
                print(f"split  {ds}/{sp['id']:10s} {sp['status']:8s} " + " ".join(f"{k}={v['n']}" for k, v in st.items()) + f"  {sp.get('title', '')}")
        for m in registry.list_models():
            acc = m.get("test", {}).get("accuracy")
            print(f"model  {m['name']}:v{m['version']} {m['dataset']}/{m['split']} {m['method']:9s} {m['status']:7s} "
                  f"stage={m['stage']} test={'%.4f' % acc if acc is not None else '-'}  {m.get('description', '')[:60]}")
        for j in registry.list_jobs(10):
            print(f"job    {j['id']} {j['type']:11s} {j['status']:9s} {json.dumps(j['args'])}")
    elif a.cmd == "split":
        if a.val == "official":
            val = ("official", None)
        else:
            mode, _, v = a.val.partition(":")
            val = ("holdout_fraction", float(v)) if "." in v else ("holdout_per_class", int(v))
        split_id, job = registry.reserve_split(a.dataset, a.title, a.description, a.train_per_class, a.test_per_class, val, a.seed)
        print(f"queued split {a.dataset}/{split_id} (job {job['id']})")
    elif a.cmd == "train":
        m, job = registry.reserve_model(a.name, a.dataset, a.split, a.method, json.loads(a.params), a.description,
                                        [x for x in a.tags.split(",") if x])
        print(f"queued model {m['name']}:v{m['version']} (job {job['id']})")
    elif a.cmd == "retrain":
        prev = json.loads((registry.model_dir(a.name, a.version) / "model.json").read_text())
        m, job = registry.reserve_model(a.name, prev["dataset"], a.split or prev["split"], prev["method"],
                                        json.loads(a.params) if a.params else prev.get("params"),
                                        a.description if a.description is not None else prev.get("description", ""),
                                        prev.get("tags", []), parent=f"v{a.version}")
        print(f"queued model {m['name']}:v{m['version']} from v{a.version} (job {job['id']})")
    elif a.cmd == "describe":
        m = registry.describe_model(a.name, a.version, a.description,
                                    [x for x in a.tags.split(",")] if a.tags is not None else None, a.stage)
        print(json.dumps({k: m[k] for k in ("name", "version", "description", "tags", "stage")}, indent=1))
    elif a.cmd == "add-dataset":
        fmt = "jsonl" if a.file.endswith((".jsonl", ".json")) else "csv"
        meta, job = registry.reserve_dataset(a.name, a.file, fmt, a.title, a.lang, a.domain, a.description, a.benchmark)
        print(f"queued dataset {a.name}: {meta['rows']} rows, {meta['classes']} classes (job {job['id']})")
    elif a.cmd == "register-benchmark":
        from register_benchmark import register_all
        for m in register_all():
            print(f"registered {m['name']}:v{m['version']} ({m['method']}, test {m['test']['accuracy']:.4f})")
    elif a.cmd == "worker":
        print("job worker started (Ctrl+C to stop)", flush=True)
        registry.worker()
    elif a.cmd == "run":
        registry.run_job(a.job_id, log=lambda msg: print(msg, flush=True))


if __name__ == "__main__":
    sys.exit(main())
