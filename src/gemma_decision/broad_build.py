"""`gd-broad-build` - broad text decision data from hotchpotch/bekko-system-one-dataset-v0, commercial-safe only.

The bekko dataset converts ~150 NLP / retrieval datasets into System One judgments (Choice / Noul / Score), the
same schema as the S1MB benchmark. Its README lists each subset's upstream license; only subsets whose license
is in SAFE_LICENSES are used (no "unknown", non-commercial, custom, copyleft-code, "rights vary" or unconfirmed
terms). From each subset's `train` split we sample up to --per-source cases (seeded), drop every case whose
input hash or group also appears in the S1MB English test set (data/s1mb/english-v1), and render it with
gemma_decision.s1mb_format, the same rendering scripts/s1mb_bench.py evaluates with. Each subset's `validation`
split (up to 100 cases, never trained on) becomes our test split for per-source reporting.

--exclude DIR (a previous build) skips the train cases DIR already holds, so a continued run sees only new cases,
and copies DIR's test files instead of resampling them, so per-source test results stay comparable.

Input: `snapshot_download` of the dataset into --src (data/broad/bekko-v0). Output: --out/{train,test}/bk_<subset>.parquet
in the gd-pilot-build schema, and --out/sources.json (subset, license, upstream, counts, exclusions).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .pilot_build import SCHEMA, make_row
from .s1mb_format import KEYS, case_from_row, render, split_case

REPO = "hotchpotch/bekko-system-one-dataset-v0"
REVISION = "9c673a09c31e145684ab9915871afa72ba40a383"
# License strings exactly as the dataset README's table states them (links stripped). Attribution and
# share-alike licenses are allowed, as in the pilot mix; mixed entries only where every option qualifies.
SAFE_LICENSES = {
    "CC0-1.0", "CC0 (numeric data)", "CC BY 2.0", "CC BY 3.0", "CC BY 4.0", "CC BY 4.0 (captions)",
    "CC BY-SA 3.0", "CC BY-SA 4.0", "CC BY-SA 4.0 / GFDL", "Apache-2.0", "Apache-2.0 / CC BY-SA 4.0",
    "Apache-2.0; Wikipedia text: CC BY-SA / GFDL", "MIT", "BSD", "BSD-2-Clause (COPA)", "AFL-3.0",
    "cdla-sharing-1.0", "CC BY 3.0 / CC BY-SA 3.0 / OANC / public domain",
    "SNLI: CC BY-SA 4.0; MultiNLI: CC BY 3.0 / CC BY-SA 3.0 / OANC / public domain",
}
SKIP = {"laya__typed_decisions"}  # = LocalLLaMA/typed-decisions train, which gd-train already uses directly
MAX_SCORE_LEVELS = 16  # the head's level embedding


def license_table(readme: Path) -> dict[str, dict]:
    """{subset: {upstream, train, validation, test, license}} from the README's source table."""
    out = {}
    for line in readme.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            continue
        strip = lambda s: re.sub(r"\]\([^)]*\)", "]", s).replace("[", "").replace("]", "").replace("<br>", " ; ")  # noqa: E731
        train, val, test = (int(x.replace(",", "")) for x in cells[2:5])
        out[cells[0].strip("`")] = {"upstream": strip(cells[1]), "train": train, "validation": val, "test": test,
                                    "license": strip(cells[5])}
    return out


def s1mb_test_keys(root: Path) -> tuple[set, set]:
    hashes, groups = set(), set()
    for path in root.glob("data/*/test/*.parquet"):
        t = pq.read_table(path, columns=["input_hash", "group_id"]).to_pydict()
        hashes.update(t["input_hash"])
        groups.update(t["group_id"])
    return hashes, groups


def sample_rows(files: list[Path], n: int, rng: random.Random) -> list[dict]:
    """n random rows across files, reading only the row groups that hold them."""
    sizes = []
    for f in files:
        meta = pq.ParquetFile(f).metadata
        sizes += [(f, g, meta.row_group(g).num_rows) for g in range(meta.num_row_groups)]
    total = sum(s for *_, s in sizes)
    picks = sorted(rng.sample(range(total), min(n, total)))
    rows, offset, k = [], 0, 0
    for f, g, size in sizes:
        want = []
        while k < len(picks) and picks[k] < offset + size:
            want.append(picks[k] - offset)
            k += 1
        if want:
            table = pq.ParquetFile(f).read_row_group(g)
            rows += table.take(want).to_pylist()
        offset += size
    return rows


def convert(row: dict, source: str, license_: str) -> list[dict]:
    case = case_from_row(row)
    if case is None:
        return []
    out = []
    for part in split_case(case):
        for q in part["questions"]:
            if q["task"] == "choice" and len(q["options"]) > len(KEYS):
                return []
            if q["task"] == "score" and len(q["options"]) > MAX_SCORE_LEVELS:
                return []
        questions, labels, _ = render(part)
        rid = f"{source}:{part['id']}"
        out.append(make_row(rid, source, questions, labels, {"text": part["state"]}, ["text"], license_=license_))
    return out


def seen_cases(build: Path, source: str) -> set[str]:
    """Case ids in a previous build's train file for `source` (row ids are `source:case_id[#n]`)."""
    f = build / "train" / f"{source}.parquet"
    if not f.exists():
        return set()
    ids = pq.read_table(f, columns=["id"]).column("id").to_pylist()
    return {re.sub(r"#\d+$", "", i.split(":", 1)[1]) for i in ids}  # split_case suffixes every part #n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/broad/bekko-v0"))
    ap.add_argument("--s1mb", type=Path, default=Path("data/s1mb/english-v1"))
    ap.add_argument("--out", type=Path, default=Path("data/broad"))
    ap.add_argument("--per-source", type=int, default=500, help="train cases sampled per subset")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--exclude", type=Path, help="previous build: skip its train cases, reuse its test files")
    args = ap.parse_args()
    if args.exclude and args.exclude.resolve() == args.out.resolve():
        raise SystemExit("--exclude must differ from --out")

    table = license_table(args.src / "README.md")
    hashes, groups = s1mb_test_keys(args.s1mb)
    print(f"S1MB test: {len(hashes)} input hashes, {len(groups)} groups", flush=True)
    for split in ("train", "test"):
        (args.out / split).mkdir(parents=True, exist_ok=True)
    manifest = {"repo": REPO, "revision": REVISION, "per_source": args.per_source, "seed": args.seed,
                "exclude": str(args.exclude) if args.exclude else None, "used": {}, "excluded": {}}
    for subset, info in sorted(table.items()):
        if info["train"] == 0:
            continue
        if subset in SKIP or info["license"] not in SAFE_LICENSES:
            manifest["excluded"][subset] = info["license"] if subset not in SKIP else "duplicate of typed-decisions"
            continue
        source = f"bk_{subset}"
        rng = random.Random(f"{args.seed}-{subset}")
        stats = Counter()
        seen = seen_cases(args.exclude, source) if args.exclude else set()
        for split, out_split, n in (("train", "train", args.per_source), ("validation", "test", 100)):
            if split == "validation" and args.exclude:
                prev = args.exclude / "test" / f"{source}.parquet"
                if prev.exists():
                    shutil.copyfile(prev, args.out / "test" / prev.name)
                    stats["test_rows"] = pq.read_metadata(prev).num_rows
                continue
            files = sorted((args.src / "data" / subset / split).glob("*.parquet"))
            if not files:
                continue
            want = (n + len(seen)) * 2 if split == "train" else n  # oversample: some rows get dropped
            raw = sample_rows(files, want, rng)
            rows, kept = [], 0
            for r in raw:
                if split == "train" and r["case_id"] in seen:
                    stats["train_seen_before"] += 1
                    continue
                if r["input_hash"] in hashes or r["group_id"] in groups:
                    stats[f"{split}_s1mb_overlap"] += 1
                    continue
                if kept >= n:
                    break
                converted = convert(r, source, info["license"])
                if not converted:
                    stats[f"{split}_unconvertible"] += 1
                    continue
                rows += converted
                kept += 1
            stats[f"{split}_cases"] = kept
            stats[f"{split}_rows"] = len(rows)
            if rows:
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), args.out / out_split / f"{source}.parquet")
        manifest["used"][source] = {"subset": subset, "license": info["license"], "upstream": info["upstream"],
                                    **dict(stats)}
        print(f"{source:60s} {info['license'][:28]:28s} {dict(stats)}", flush=True)
    (args.out / "sources.json").write_text(json.dumps(manifest, indent=1))
    used = manifest["used"].values()
    print(f"\n{len(manifest['used'])} subsets used, {len(manifest['excluded'])} excluded; "
          f"train cases {sum(u.get('train_cases', 0) for u in used)}, test cases {sum(u.get('validation_cases', 0) for u in used)}"
          + (f" (test files copied from {args.exclude}: {sum(u.get('test_rows', 0) for u in used)} rows)"
             if args.exclude else ""))


if __name__ == "__main__":
    main()
