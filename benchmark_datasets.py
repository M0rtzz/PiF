from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from pathlib import Path
from typing import Any


PAPER_HARMFUL_BENCHMARKS = [
    "harmbench",
    "jailbreakbench",
    "strongreject",
    "advbench",
    "malicious_instruct",
    "do_not_answer",
    "xstest_unsafe",
    "sorrybench",
    "wildjailbreak",
]


HARMFUL_BENCH_SPECS: dict[str, dict[str, Any]] = {
    "harmbench": {
        "urls": [
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        ],
        "hf": [
            "hf:swiss-ai/harmbench:DirectRequest:test",
            "hf:swiss-ai/harmbench:HumanJailbreaks:test",
            "hf:mariagrandury/harmbench:DirectRequest:test",
            "hf:cais/HarmBench::test",
            "hf:centerforaisafety/HarmBench::test",
        ],
    },
    "jailbreakbench": {
        "urls": [
            "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv",
            "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/data/behaviors.csv",
            "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/data/behaviors.jsonl",
        ],
        "hf": [
            "hf:JailbreakBench/JBB-Behaviors:behaviors:train",
            "hf:JailbreakBench/JailbreakBench::test",
            "hf:walledai/JailbreakBench::test",
        ],
    },
    "strongreject": {
        "urls": [
            "https://raw.githubusercontent.com/alexandrasouly/strongreject/refs/heads/main/strongreject_dataset/strongreject_dataset.csv",
            "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
            "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/data/strongreject_dataset.csv",
        ],
        "hf": [
            "hf:walledai/StrongREJECT::train",
            "hf:Machlovi/strongreject-dataset::train",
            "hf:allenai/strongreject::test",
            "hf:StrongREJECT/StrongREJECT::test",
        ],
    },
    "advbench": {
        "urls": [
            "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
        ],
        "hf": [
            "hf:walledai/AdvBench::train",
            "hf:JailbreakBench/AdvBench::train",
        ],
    },
    "malicious_instruct": {
        "urls": [
            "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt",
        ],
        "hf": [
            "hf:walledai/MaliciousInstruct::train",
            "hf:TrustAIRLab/in-the-wild-jailbreak-prompts:MaliciousInstruct:train",
        ],
    },
    "do_not_answer": {
        "urls": [
            "https://huggingface.co/datasets/LibrAI/do-not-answer/resolve/main/data_en.csv?download=true",
        ],
        "hf": [
            "hf:LibrAI/do-not-answer::train",
            "hf:allenai/do-not-answer::train",
        ],
    },
    "xstest_unsafe": {
        "urls": [
            "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
        ],
        "hf": [
            "hf:Paul/XSTest::train",
            "hf:natolambert/xstest-v2-copy::train",
            "hf:walledai/XSTest::test",
            "hf:allenai/xstest-response-pairs:unsafe:test",
            "hf:XSTest/XSTest::test",
        ],
    },
    "sorrybench": {
        "hf": [
            "hf:Sorry-Bench/sorry-bench-202406::train",
        ],
    },
    "wildjailbreak": {
        "hf": [
            "hf:allenai/wildjailbreak:train:train",
            "hf:allenai/wildjailbreak::train",
            "hf:walledai/WildJailbreak::train",
        ],
    },
}


def clean_benchmark_names(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        item = str(name).strip()
        if not item or item.lower() in {"none", "baseline", "baseline_only"}:
            continue
        if item not in out:
            out.append(item)
    return out


def expand_harmful_benchmark_names(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in clean_benchmark_names(names):
        if name.lower() == "all":
            for expanded_name in PAPER_HARMFUL_BENCHMARKS:
                if expanded_name not in out:
                    out.append(expanded_name)
            continue
        if name not in out:
            out.append(name)
    return out


def records_to_instructions(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["instruction"]) for record in records]


def benchmark_records_meta(records_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        str(name): {
            "count": len(records),
            "ids": [str(record.get("id", "")) for record in records],
            "categories": sorted(
                {
                    str(record.get("category"))
                    for record in records
                    if record.get("category") is not None
                }
            ),
        }
        for name, records in records_by_name.items()
    }


def _stable_unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        instruction = str(record.get("instruction", "")).strip()
        if not instruction or instruction in seen:
            continue
        seen.add(instruction)
        out.append(dict(record))
    return out


def sample_records_deterministic(
    records: list[dict[str, Any]],
    *,
    n: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    records = _stable_unique_records(records)
    if n is None or int(n) <= 0 or int(n) >= len(records):
        return records
    rng = random.Random(int(seed))
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[index] for index in sorted(indices[: int(n)])]


def _record_from_prompt(
    prompt: Any,
    *,
    bench_name: str,
    index: int,
    category: Any = None,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    instruction = str(prompt or "").strip()
    if not instruction:
        return None
    record = {
        "id": f"{bench_name}-{index:04d}",
        "instruction": instruction,
        "category": None if category is None else str(category),
        "benchmark": str(bench_name),
        "source": str(source),
    }
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                record[str(key)] = value
            else:
                record[str(key)] = json.dumps(value, ensure_ascii=False)
    return record


def _record_from_materialized_row(
    row: dict[str, Any],
    *,
    bench_name: str,
    index: int,
    source: str,
) -> dict[str, Any] | None:
    if not str(row.get("instruction", "")).strip():
        return None
    preserved_markers = {
        "benchmark",
        "task_type",
        "reference_answers",
        "reference_answer",
        "best_answer",
        "correct_answers",
        "incorrect_answers",
        "mc1_targets",
        "mc2_targets",
        "mc_choices",
        "mc_choice_labels",
        "mc_label_index",
        "reference_output",
        "reference_summary",
    }
    if not any(key in row for key in preserved_markers):
        return None
    record = dict(row)
    record.setdefault("id", f"{bench_name}-{index:04d}")
    record.setdefault("benchmark", str(bench_name))
    record.setdefault("category", str(row.get("category", "")) or None)
    record.setdefault("source", str(source))
    return record


def _row_to_prompt(row: dict[str, Any]) -> tuple[str | None, Any]:
    turns = row.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if str(turn or "").strip():
                return str(turn).strip(), row.get("Category") or row.get("category")
    if row.get("Goal") is not None and str(row.get("Goal")).strip():
        return str(row["Goal"]).strip(), row.get("Category") or row.get("category")
    if row.get("goal") is not None and str(row.get("goal")).strip():
        return str(row["goal"]).strip(), row.get("Category") or row.get("category")
    prompt_keys = (
        "instruction",
        "prompt",
        "query",
        "request",
        "behavior",
        "Behavior",
        "goal",
        "Goal",
        "question",
        "Question",
        "forbidden_prompt",
        "jailbreak_prompt",
        "advbench_prompt",
        "text",
        "input",
    )
    category_keys = ("category", "Category", "semantic_category", "FunctionalCategory", "source")
    prompt: str | None = None
    for key in prompt_keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            prompt = str(value).strip()
            break
    category = None
    for key in category_keys:
        if row.get(key) is not None:
            category = row.get(key)
            break
    return prompt, category


def _is_xstest_unsafe_row(row: dict[str, Any]) -> bool:
    label = str(row.get("label") or row.get("safety_label") or row.get("classification") or "").lower()
    row_type = str(row.get("type") or row.get("category") or row.get("prompt_type") or "").lower()
    if label in {"unsafe", "harmful", "dangerous", "disallowed"}:
        return True
    if label in {"safe", "benign", "allowed"}:
        return False
    return row_type.startswith("contrast") or "unsafe" in row_type or "harmful" in row_type


def records_from_rows(
    rows: list[dict[str, Any]],
    *,
    bench_name: str,
    source: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if str(bench_name).lower() == "xstest_unsafe" and not _is_xstest_unsafe_row(row):
            continue
        materialized = _record_from_materialized_row(
            dict(row),
            bench_name=bench_name,
            index=index,
            source=source,
        )
        if materialized is not None:
            records.append(materialized)
            continue
        prompt, category = _row_to_prompt(dict(row))
        record = _record_from_prompt(
            prompt,
            bench_name=bench_name,
            index=index,
            category=category,
            source=source,
        )
        if record is not None:
            records.append(record)
    return _stable_unique_records(records)


def _read_tabular_records_from_text(text: str, *, suffix: str) -> list[dict[str, Any]]:
    suffix_l = suffix.lower()
    stripped = text.strip()
    if not stripped:
        return []
    if suffix_l.endswith(".jsonl"):
        rows = []
        for lineno, line in enumerate(stripped.split("\n"), 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise json.JSONDecodeError(
                        f"{exc.msg} at JSONL physical line {lineno}",
                        exc.doc,
                        exc.pos,
                    ) from exc
        return [dict(row) for row in rows]
    if suffix_l.endswith(".csv"):
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
    if suffix_l.endswith(".txt"):
        return [
            {"prompt": line.strip()}
            for line in stripped.splitlines()
            if line.strip()
        ]
    data = json.loads(stripped)
    if isinstance(data, list):
        return [
            dict(row) if isinstance(row, dict) else {"instruction": row}
            for row in data
        ]
    if isinstance(data, dict):
        for key in ("data", "examples", "behaviors", "prompts", "train", "validation", "test"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(row) if isinstance(row, dict) else {"instruction": row} for row in value]
    return []


def _safe_cache_name(url: str) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(url))[-180:]
    suffix = Path(str(url).split("?", 1)[0]).suffix
    if suffix and not safe_name.endswith(suffix):
        safe_name = f"{safe_name}{suffix}"
    return safe_name or "downloaded_dataset"


def download_file(
    url: str,
    *,
    cache_dir: Path,
    timeout_seconds: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _safe_cache_name(url)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    request = urllib.request.Request(url, headers={"User-Agent": "PiF-benchmark-loader"})
    with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _read_tabular_records_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets package is required for parquet benchmark loading") from exc
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        return [dict(row) for row in dataset]
    text = path.read_text(encoding="utf-8")
    return _read_tabular_records_from_text(text, suffix=suffix)


def _load_records_from_url(
    url: str,
    *,
    bench_name: str,
    cache_dir: Path,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    cache_path = download_file(
        url,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    rows = _read_tabular_records_from_file(cache_path)
    return records_from_rows(rows, bench_name=bench_name, source=url)


def _load_records_from_path(path: str | Path, *, bench_name: str) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows = _read_tabular_records_from_text(text, suffix=path.suffix)
    return records_from_rows(rows, bench_name=bench_name, source=str(path))


def _parse_hf_spec(spec: str, *, default_split: str = "test") -> tuple[str, str | None, str]:
    body = spec[3:] if spec.startswith("hf:") else spec
    parts = body.split(":")
    dataset_name = parts[0]
    config = parts[1] if len(parts) >= 2 and parts[1] else None
    split = parts[2] if len(parts) >= 3 and parts[2] else default_split
    return dataset_name, config, split


def _load_raw_rows_from_hf(
    spec: str,
    *,
    default_split: str = "test",
    local_files_only: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    dataset_name, config, split = _parse_hf_spec(spec, default_split=default_split)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required for hf: benchmark loading") from exc

    load_args = [dataset_name]
    if config is not None:
        load_args.append(config)
    load_kwargs = {"split": split}
    if local_files_only:
        load_kwargs["download_mode"] = "reuse_cache_if_exists"
    try:
        dataset = load_dataset(*load_args, **load_kwargs)
    except Exception:
        fallback_split = "validation" if split == "test" else "train"
        load_kwargs["split"] = fallback_split
        dataset = load_dataset(*load_args, **load_kwargs)
        split = fallback_split
    rows = [dict(row) for row in dataset]
    return rows, f"hf:{dataset_name}:{config or ''}:{split}"


def _load_records_from_hf(
    spec: str,
    *,
    bench_name: str,
    default_split: str = "test",
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    rows, source = _load_raw_rows_from_hf(
        spec,
        default_split=default_split,
        local_files_only=local_files_only,
    )
    return records_from_rows(rows, bench_name=bench_name, source=source)


def load_named_harmful_benchmark(
    name: str,
    *,
    cache_dir: Path,
    timeout_seconds: int,
    require_external: bool,
    materialized_dir: Path = Path("outputs/downloaded_benchmarks"),
    local_files_only: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    key = str(name).strip()
    lower = key.lower()
    errors: list[str] = []

    if lower.startswith("hf:"):
        try:
            dataset_name, _, _ = _parse_hf_spec(key)
            bench_name = Path(dataset_name).name or "hf_benchmark"
            return _load_records_from_hf(
                key,
                bench_name=bench_name,
                local_files_only=local_files_only,
            ), errors
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            if require_external:
                raise
            return [], errors
    if lower.startswith("url:"):
        url = key[4:]
        try:
            return _load_records_from_url(
                url,
                bench_name=Path(url).stem or "url_benchmark",
                cache_dir=cache_dir,
                timeout_seconds=timeout_seconds,
            ), errors
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            if require_external:
                raise
            return [], errors
    if Path(key).exists():
        return _load_records_from_path(key, bench_name=Path(key).stem), errors

    spec = HARMFUL_BENCH_SPECS.get(lower)
    if spec is None:
        errors.append(f"unknown harmful benchmark: {name}")
        if require_external:
            raise ValueError(errors[-1])
        return [], errors

    materialized_path = Path(materialized_dir) / "harmful" / f"{lower}.jsonl"
    if materialized_path.exists():
        try:
            rows = _load_records_from_path(materialized_path, bench_name=lower)
            if rows:
                return rows, errors
        except Exception as exc:
            errors.append(f"{materialized_path}: {exc}")

    def _try_hf(use_local_files_only: bool) -> list[dict[str, Any]]:
        for hf_spec in spec.get("hf", []):
            try:
                rows = _load_records_from_hf(
                    str(hf_spec),
                    bench_name=lower,
                    local_files_only=use_local_files_only,
                )
                if rows:
                    return rows
            except Exception as exc:
                errors.append(f"{hf_spec}: {exc}")
        return []

    def _try_urls() -> list[dict[str, Any]]:
        for url in spec.get("urls", []):
            try:
                rows = _load_records_from_url(
                    str(url),
                    bench_name=lower,
                    cache_dir=cache_dir,
                    timeout_seconds=timeout_seconds,
                )
                if rows:
                    return rows
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        return []

    rows = _try_hf(local_files_only)
    if rows:
        return rows, errors
    if not local_files_only:
        rows = _try_urls()
        if rows:
            return rows, errors
    rows = _try_hf(True)
    if rows:
        return rows, errors

    if require_external:
        raise RuntimeError(f"failed to load harmful benchmark {name}: {' | '.join(errors)}")
    return [], errors


def load_harmful_benchmark_records(
    names: list[str],
    *,
    cache_dir: Path,
    timeout_seconds: int,
    max_examples: int,
    seed: int,
    require_external: bool,
    materialized_dir: Path = Path("outputs/downloaded_benchmarks"),
    local_files_only: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    errors_by_name: dict[str, list[str]] = {}
    clean_names = expand_harmful_benchmark_names(clean_benchmark_names(names))
    for offset, name in enumerate(clean_names):
        records, errors = load_named_harmful_benchmark(
            name,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            require_external=require_external,
            materialized_dir=materialized_dir,
            local_files_only=local_files_only,
        )
        sampled = sample_records_deterministic(
            records,
            n=max_examples,
            seed=int(seed) + 1009 + offset,
        )
        loaded[str(name)] = sampled
        errors_by_name[str(name)] = errors
        if sampled:
            print(f"[bench] Loaded {len(sampled)} harmful examples from {name}")
        else:
            print(f"[bench] Skipped harmful benchmark {name}; errors={errors}")
    return loaded, errors_by_name


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)
