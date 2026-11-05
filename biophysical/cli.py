"""Command-line interface.

    python -m biophysical.cli datasets list
    python -m biophysical.cli datasets inspect <id>
    python -m biophysical.cli datasets download <id> [--allow-large]
    python -m biophysical.cli datasets validate <id>
    python -m biophysical.cli datasets set-local <id> <path>
    python -m biophysical.cli data preprocess <id> [--config configs/benchmark_norman.yaml]
    python -m biophysical.cli benchmark run [--config configs/benchmark_norman.yaml]
    python -m biophysical.cli benchmark representations [--gene2vec path]
    python -m biophysical.cli predict <perturbation> [--top 20]
    python -m biophysical.cli rank --candidates A,B,C --budget 5 [--strategy hybrid]
    python -m biophysical.cli geo summary GSE133344
    python -m biophysical.cli string fetch --genes A,B,C
    python -m biophysical.cli annotate targets A,B

Heavy modules are imported lazily so that dataset commands work with minimal dependencies.
"""
from __future__ import annotations

import argparse
import json
import sys

from biophysical.logging_utils import get_logger

log = get_logger("biophysical.cli")
DEFAULT_BENCH_CFG = "configs/benchmark_norman.yaml"


def _registry():
    from biophysical.data.registry import DatasetRegistry

    return DatasetRegistry()


def cmd_datasets_list(args) -> int:
    reg = _registry()
    print(f"{'dataset_id':30s} {'size_MB':>8s}  {'status':24s} {'role':34s} name")
    for e in reg.list():
        print(f"{e.dataset_id:30s} {e.size_mb or 0:8.1f}  {e.current_status(reg.paths):24s} {e.role or '':34s} {e.name}")
    return 0


def cmd_datasets_inspect(args) -> int:
    from biophysical.data.inspect import inspect_dataset

    reg = _registry()
    entry = reg.get(args.dataset_id)
    report = inspect_dataset(entry, paths=reg.paths, write=True)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_datasets_download(args) -> int:
    from biophysical.data.download import download_dataset

    reg = _registry()
    entry = reg.get(args.dataset_id)
    path = download_dataset(entry, allow_large=args.allow_large, max_default_mb=reg.defaults.get("max_default_download_mb", 800), paths=reg.paths)
    print(path)
    return 0


def cmd_datasets_validate(args) -> int:
    from biophysical.data.validate import validate_dataset

    reg = _registry()
    rec = validate_dataset(reg.get(args.dataset_id), paths=reg.paths)
    print(json.dumps(rec, indent=2, default=str))
    return 0 if rec["status"] == "valid" else 1


def cmd_datasets_set_local(args) -> int:
    from pathlib import Path

    from biophysical.config import dump_yaml, get_paths, load_yaml

    paths = get_paths()
    reg = _registry()
    reg.get(args.dataset_id)  # must be a registered id
    p = Path(args.path).expanduser().resolve()
    if not p.exists():
        print(f"error: {p} does not exist", file=sys.stderr)
        return 2
    cfg = load_yaml(paths.local_datasets) if paths.local_datasets.exists() else {}
    cfg[args.dataset_id] = str(p)
    dump_yaml(cfg, paths.local_datasets)
    print(f"{args.dataset_id} -> {p} (recorded in {paths.local_datasets})")
    return 0


def cmd_data_preprocess(args) -> int:
    from biophysical.preprocessing.pipeline import preprocess_dataset

    out = preprocess_dataset(args.dataset_id, config_path=args.config, force=args.force)
    print(out)
    return 0


def cmd_benchmark_run(args) -> int:
    from biophysical.evaluation.benchmark import run_benchmark

    run_dir = run_benchmark(args.config, quick=args.quick)
    print(run_dir)
    return 0


def cmd_benchmark_representations(args) -> int:
    from pathlib import Path

    from biophysical.config import get_paths
    from biophysical.evaluation.representations import run_representation_benchmark

    out = args.out or Path(get_paths().results) / "representations"
    print(run_representation_benchmark(args.config, out, use_string=not args.no_string, gene2vec_path=args.gene2vec))
    return 0


def cmd_predict(args) -> int:
    from biophysical.api.service import PredictionService

    svc = PredictionService.from_latest(args.run)
    res = svc.predict(args.perturbation, top_k=args.top, allow_unvalidated=args.allow_unvalidated)
    res.pop("predicted_response", None)  # the full gene table; top_genes is printed instead
    print(json.dumps(res, indent=2, default=str))
    if args.out and res["status"] != "abstain":
        svc.predict(args.perturbation, allow_unvalidated=args.allow_unvalidated)["predicted_response"].to_csv(args.out, index=False)
    return 0


def cmd_rank(args) -> int:
    from biophysical.api.service import PredictionService

    svc = PredictionService.from_latest(args.run)
    cands = [c.strip() for c in args.candidates.split(",") if c.strip()] if args.candidates else None
    table = svc.rank(cands, budget=args.budget, strategy=args.strategy, record=True)
    print(table.to_string(index=False))
    return 0


def cmd_geo_summary(args) -> int:
    from biophysical.data.geo import GEOClient

    print(json.dumps(GEOClient().summary(args.accession), indent=2))
    return 0


def cmd_string_fetch(args) -> int:
    from biophysical.networks.string_db import StringClient

    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    client = StringClient()
    df = client.network(genes, required_score=args.min_score)
    print(df.head(50).to_string(index=False))
    print(f"\n{len(df)} edges; STRING version: {client.version()}")
    return 0


def cmd_annotate_targets(args) -> int:
    from biophysical.annotations.opentargets import OpenTargetsClient

    client = OpenTargetsClient()
    for sym in args.symbols.split(","):
        print(json.dumps(client.target_summary(sym.strip()), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biophysical", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="group", required=True)

    ds = sub.add_parser("datasets").add_subparsers(dest="cmd", required=True)
    ds.add_parser("list").set_defaults(func=cmd_datasets_list)
    for name, fn in [("inspect", cmd_datasets_inspect), ("validate", cmd_datasets_validate)]:
        sp = ds.add_parser(name)
        sp.add_argument("dataset_id")
        sp.set_defaults(func=fn)
    sp = ds.add_parser("download")
    sp.add_argument("dataset_id")
    sp.add_argument("--allow-large", action="store_true")
    sp.set_defaults(func=cmd_datasets_download)
    sp = ds.add_parser("set-local", help="register a locally downloaded file for a dataset id")
    sp.add_argument("dataset_id")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_datasets_set_local)

    dp = sub.add_parser("data").add_subparsers(dest="cmd", required=True)
    sp = dp.add_parser("preprocess")
    sp.add_argument("dataset_id")
    sp.add_argument("--config", default=DEFAULT_BENCH_CFG)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_data_preprocess)

    bp = sub.add_parser("benchmark").add_subparsers(dest="cmd", required=True)
    sp = bp.add_parser("run")
    sp.add_argument("--config", default=DEFAULT_BENCH_CFG)
    sp.add_argument("--quick", action="store_true", help="fewer folds/epochs; smoke test only")
    sp.set_defaults(func=cmd_benchmark_run)

    sp = bp.add_parser("representations", help="compare STRING and pretrained representations against co-expression")
    sp.add_argument("--config", default=DEFAULT_BENCH_CFG)
    sp.add_argument("--out", default=None)
    sp.add_argument("--no-string", action="store_true")
    sp.add_argument("--gene2vec", default=None, help="path to a text embedding file")
    sp.set_defaults(func=cmd_benchmark_representations)

    sp = sub.add_parser("predict")
    sp.add_argument("perturbation", help="gene symbol, or two symbols joined by '_' for a combination")
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--out", default=None, help="write the full predicted gene table to this CSV")
    sp.add_argument("--allow-unvalidated", action="store_true", help="return a prediction even where the benchmark shows it is worse than baseline")
    sp.add_argument("--run", default=None, help="benchmark run directory (default: latest)")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("rank")
    sp.add_argument("--candidates", default=None, help="comma-separated perturbations (default: all unmeasured candidates)")
    sp.add_argument("--budget", type=int, default=10)
    sp.add_argument("--strategy", default="hybrid", choices=["uncertainty", "diversity", "hybrid", "magnitude", "expected_improvement", "random"])
    sp.add_argument("--run", default=None)
    sp.set_defaults(func=cmd_rank)

    gp = sub.add_parser("geo").add_subparsers(dest="cmd", required=True)
    sp = gp.add_parser("summary")
    sp.add_argument("accession")
    sp.set_defaults(func=cmd_geo_summary)

    stp = sub.add_parser("string").add_subparsers(dest="cmd", required=True)
    sp = stp.add_parser("fetch")
    sp.add_argument("--genes", required=True)
    sp.add_argument("--min-score", type=int, default=400)
    sp.set_defaults(func=cmd_string_fetch)

    ap = sub.add_parser("annotate").add_subparsers(dest="cmd", required=True)
    sp = ap.add_parser("targets")
    sp.add_argument("symbols")
    sp.set_defaults(func=cmd_annotate_targets)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
