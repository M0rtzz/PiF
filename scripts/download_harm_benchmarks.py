from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_datasets import (  # noqa: E402
    HARMFUL_BENCH_SPECS,
    PAPER_HARMFUL_BENCHMARKS,
    expand_harmful_benchmark_names,
    load_named_harmful_benchmark,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and materialize COMBAT paper-suite harmful benchmarks for PiF. "
            "If network download fails, existing local materialized/cache files are reused."
        ),
    )
    parser.add_argument(
        "--harmful-benchmarks",
        nargs="+",
        default=None,
        help="Harmful benchmarks to download. Defaults to COMBAT paper harmful suite.",
    )
    parser.add_argument(
        "--all-builtins",
        action="store_true",
        help="Download every built-in harmful benchmark.",
    )
    parser.add_argument(
        "--benchmark-cache-dir",
        type=Path,
        default=Path("outputs/benchmark_cache"),
        help="Cache directory for raw URL files.",
    )
    parser.add_argument(
        "--write-jsonl-dir",
        type=Path,
        default=Path("outputs/downloaded_benchmarks"),
        help="Directory where normalized benchmark JSONL files are written.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-hub-download-timeout", type=int, default=300)
    parser.add_argument("--hf-hub-etag-timeout", type=int, default=60)
    parser.add_argument("--disable-hf-xet", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any requested benchmark cannot be downloaded or loaded from cache.",
    )
    return parser


def _configure_hf_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(int(args.hf_hub_download_timeout)))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(int(args.hf_hub_etag_timeout)))
    if args.hf_cache_dir is not None:
        hf_cache_dir = Path(args.hf_cache_dir)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache_dir))
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_dir / "datasets"))
    if args.disable_hf_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


def _selected_benchmarks(args: argparse.Namespace) -> list[str]:
    if args.all_builtins:
        return list(HARMFUL_BENCH_SPECS)
    names = args.harmful_benchmarks
    if names is None:
        names = list(PAPER_HARMFUL_BENCHMARKS)
    return expand_harmful_benchmark_names(list(names))


def main() -> int:
    args = build_parser().parse_args()
    _configure_hf_env(args)

    failures: dict[str, list[str]] = {}
    for name in _selected_benchmarks(args):
        records, errors = load_named_harmful_benchmark(
            name,
            cache_dir=args.benchmark_cache_dir,
            timeout_seconds=int(args.timeout_seconds),
            require_external=False,
            materialized_dir=args.write_jsonl_dir,
            local_files_only=False,
        )
        if records:
            out_path = Path(args.write_jsonl_dir) / "harmful" / f"{str(name).lower()}.jsonl"
            write_jsonl(out_path, records)
            print(f"[ok] {name}: wrote {len(records)} records to {out_path}")
        else:
            failures[str(name)] = errors
            print(f"[fail] {name}: no records loaded; errors={errors}")

    if failures:
        print("[summary] failed benchmarks:")
        for name, errors in failures.items():
            print(f"  - {name}: {errors}")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
