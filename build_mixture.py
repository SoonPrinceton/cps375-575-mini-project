"""Build SFT mixtures from cleaned JSONL files with fixed validation exclusion."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from prepare_data import CATEGORIES, write_json, prompt_hash


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            output = {"messages": row["messages"]}
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"{path}: {len(rows):,} rows")


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_provenance(source_dir):
    path = source_dir / "manifest.json"
    return read_json(path) if path.exists() else {}


def allocate_counts(total, ratios):
    """Allocate integer sample counts using largest remainders."""
    k = len(CATEGORIES)

    if (
        total <= 0
        or len(ratios) != k
        or any(not math.isfinite(r) or r < 0 for r in ratios)
        or sum(ratios) <= 0
    ):
        raise ValueError(
            f"Require positive size and {k} finite "
            "nonnegative ratios with positive sum"
        )

    quotas = [
        total * r / sum(ratios)
        for r in ratios
    ]

    counts = [int(q) for q in quotas]

    order = sorted(
        range(k),
        key=lambda i: quotas[i] - counts[i],
        reverse=True,
    )

    for i in order[:total - sum(counts)]:
        counts[i] += 1

    return counts


# ---------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------

def sample_jsonl(
    path,
    n,
    rng,
    excluded_ids=None,
    excluded_prompts=None,
):
    """Uniformly sample n eligible rows without replacement."""

    if n == 0:
        return []

    if excluded_ids is None:
        excluded_ids = set()

    if excluded_prompts is None:
        excluded_prompts = set()

    reservoir = []
    eligible = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row["id"] in excluded_ids:
                continue

            if (
                excluded_prompts
                and prompt_hash(row["messages"]) in excluded_prompts
            ):
                continue

            eligible += 1

            if len(reservoir) < n:
                reservoir.append(row)
            else:
                j = rng.randrange(eligible)

                if j < n:
                    reservoir[j] = row

    if len(reservoir) != n:
        raise ValueError(
            f"{path}: requested {n:,} samples, "
            f"but only {eligible:,} eligible rows"
        )

    return reservoir


# ---------------------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------------------

def load_validation(validation_path, provenance=None):
    """
    Load an existing validation dataset.

    Never create or modify validation during training data construction.
    """

    provenance = provenance or {}

    meta_path = validation_path.with_suffix(".manifest.json")

    for path in (validation_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Validation file not found: {path}. "
                "Run build_mixture.py --create-validation first."
            )

    meta = read_json(meta_path)
    rows = read_jsonl(validation_path)

    if not rows:
        raise ValueError(
            f"Validation dataset is empty: {validation_path}"
        )

    # IDs live in the manifest; prompts come from the messages-only JSONL.
    sample_ids = meta.get("sample_ids")

    if not isinstance(sample_ids, list) or len(sample_ids) != len(rows):
        raise ValueError("Validation manifest must contain one sample ID per row")

    validation_ids = set(sample_ids)

    validation_prompts = {
        prompt_hash(row["messages"])
        for row in rows
    }

    if len(validation_ids) != len(rows):
        raise ValueError("Validation contains duplicate sample IDs")

    if len(validation_prompts) != len(rows):
        raise ValueError("Validation contains duplicate prompts")

    # Prevent mixing validation and source data from different revisions.
    for key in ("dataset", "revision"):
        old = meta.get(key)
        new = provenance.get(key)

        if old is not None and new is not None and old != new:
            raise ValueError(
                f"Validation {key} mismatch: {old} != {new}. "
                "Check your source data and validation dataset."
            )

    # Detect accidental modification of validation.
    expected_hash = meta.get("validation_sha256")

    if expected_hash is not None and expected_hash != file_sha256(validation_path):
        raise ValueError(
            f"Validation file does not match its manifest: {validation_path}. "
            "Recreate validation explicitly if you intend to change it."
        )

    print(
        f"Excluding fixed validation: "
        f"{validation_path} ({len(rows):,} rows)"
    )

    return rows, validation_ids


# ---------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------

def build_mixture(
    source_dir,
    output,
    size,
    ratios,
    validation_path,
    seed=42,
    exclude_manifests=None,
    create_validation=False,
    force=False,
):
    """
    Sample validation or training from cleaned category files.

    Training always excludes the existing validation dataset.
    """

    meta_path = output.with_suffix(".manifest.json")

    if (output.exists() or meta_path.exists()) and not force:
        raise FileExistsError(
            f"Output already exists: {output} or {meta_path}. "
            "Use --force to overwrite it."
        )

    exclude_manifests = exclude_manifests or []
    provenance = load_provenance(source_dir)
    counts = allocate_counts(size, ratios)
    mode = "validation" if create_validation else "train"
    label = "Validation" if create_validation else "Training"
    validation = []
    validation_ids = set()
    validation_hash = None

    if create_validation:
        print(
            "WARNING: Previously generated training datasets and trained "
            "checkpoints may contain samples from the new validation set. "
            "Regenerate training mixtures or verify that they are disjoint "
            "before using the new validation set."
        )
    else:
        if validation_path.is_file():
            validation_hash = file_sha256(validation_path)
        validation, validation_ids = load_validation(
            validation_path,
            provenance=provenance,
        )

    validation_prompts = {
        prompt_hash(row["messages"])
        for row in validation
    }

    # 2. Exclude validation IDs and optionally previous training IDs.
    excluded_ids = set(validation_ids)

    for path in exclude_manifests:
        meta = read_json(path)
        excluded_ids.update(meta["sample_ids"])

    print(f"Excluded sample IDs: {len(excluded_ids):,}")

    # 3. Sample each category independently.
    rows = []

    for cat, n in zip(CATEGORIES, counts):
        rng = random.Random(f"{seed}:{mode}:{cat}")

        selected = sample_jsonl(
            source_dir / f"{cat}.jsonl",
            n,
            rng,
            excluded_ids=excluded_ids,
            excluded_prompts=validation_prompts,
        )

        rows.extend(selected)

        print(f"{label} {cat}: {len(selected):,}")

    random.Random(seed).shuffle(rows)

    # 4. Verify training and validation are disjoint.
    train_ids = {
        row["id"]
        for row in rows
    }

    train_prompts = {
        prompt_hash(row["messages"])
        for row in rows
    }

    if train_ids & excluded_ids:
        raise RuntimeError(
            "Training contains excluded sample IDs"
        )

    if train_prompts & validation_prompts:
        raise RuntimeError(
            "Training contains validation prompts"
        )

    if len(train_ids) != len(rows):
        raise RuntimeError(
            f"Duplicate sample IDs in {label.lower()} dataset"
        )

    if len(train_prompts) != len(rows):
        raise RuntimeError(
            f"Duplicate prompts in {label.lower()} dataset"
        )

    if len(rows) != size:
        raise RuntimeError(
            f"Expected {size:,} rows, "
            f"but obtained {len(rows):,}"
        )

    # Detect validation changes during mixture construction.
    if not create_validation and file_sha256(validation_path) != validation_hash:
        raise RuntimeError(
            "Validation changed during mixture construction. "
            "Run the command again with a fixed validation dataset."
        )

    # 5. Save messages separately from sample metadata in both modes.
    write_jsonl(output, rows)

    metadata = {
        "dataset": provenance.get("dataset"),
        "revision": provenance.get("revision"),
        "source_dir": str(source_dir),

        "size": len(rows),
        "seed": seed,

        "counts": dict(zip(CATEGORIES, counts)),
        "ratios": dict(zip(CATEGORIES, ratios)),

        "sample_ids": [
            row["id"]
            for row in rows
        ],
        "samples": [
            {key: row[key] for key in ("id", "source", "category")}
            for row in rows
        ],
    }

    if create_validation:
        metadata["validation_sha256"] = file_sha256(output)
    else:
        metadata.update({
            "validation_path": str(validation_path),
            "validation_size": len(validation),
            "validation_sha256": validation_hash,
            "excluded_manifests": [str(path) for path in exclude_manifests],
        })
        print(
            f"\nConfirmed: {len(rows):,} training rows are disjoint "
            f"from {len(validation):,} validation rows."
        )

    write_json(meta_path, metadata)
    print(f"{label} dataset: {output}")
    print(f"Manifest: {meta_path}")

    return rows


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)

    # Operation mode
    p.add_argument(
        "--create-validation",
        action="store_true",
        help="Create validation instead of building a training mixture",
    )

    # Source data
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/full"),
        help="Directory containing cleaned category JSONL files",
    )

    # Validation arguments
    p.add_argument(
        "--validation",
        type=Path,
        default=Path("data/validation.jsonl"),
        help="Existing validation JSONL path (default: data/validation.jsonl)",
    )

    # Shared sampling arguments
    p.add_argument(
        "--size",
        type=int,
        default=None,
        help="Total number of samples (required in both modes)",
    )

    p.add_argument(
        "--ratios",
        type=float,
        nargs=len(CATEGORIES),
        default=[1, 1, 1],
        metavar=("IF", "MATH", "CODE"),
        help="Relative sampling weights for IF, Math, and Coding",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path; defaults to data/validation.jsonl in validation mode",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed",
    )

    p.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Exclude sample IDs from previous training runs; repeatable",
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing existing output files",
    )

    a = p.parse_args()

    validation_path = a.validation

    if validation_path.suffix != ".jsonl":
        p.error("--validation must end in .jsonl")

    if a.size is None or a.size <= 0:
        p.error("--size must be positive in both modes")

    if a.create_validation:
        if a.output is None:
            a.output = Path("data/validation.jsonl")
        if a.exclude_manifest:
            p.error("--exclude-manifest is only used when building training mixtures")
    elif a.output is None:
        p.error("--output is required when building a mixture")

    if a.output.suffix != ".jsonl":
        p.error("--output must end in .jsonl")

    # Verify the cleaned source files exist.
    for cat in CATEGORIES:
        path = a.source_dir / f"{cat}.jsonl"

        if not path.is_file():
            p.error(
                f"Source file not found: {path}. "
                "Run prepare_data.py first."
            )

    # Protect the original cleaned source data.
    source_paths = {
        (a.source_dir / f"{cat}.jsonl").resolve()
        for cat in (*CATEGORIES, "all")
    }

    source_paths.add(
        (a.source_dir / "manifest.json").resolve()
    )

    validation_paths = {
        validation_path.resolve(),
        validation_path.with_suffix(".manifest.json").resolve(),
    }

    meta_path = a.output.with_suffix(".manifest.json")
    protected_paths = source_paths

    if not a.create_validation:
        protected_paths = protected_paths | validation_paths

    if (
        a.output.resolve() in protected_paths
        or meta_path.resolve() in protected_paths
    ):
        p.error("Output must not overwrite source or validation files")

    if (a.output.exists() or meta_path.exists()) and not a.force:
        p.error(
            f"Output already exists: {a.output} or {meta_path}. "
            "Use --force to overwrite it."
        )

    # Protect manifests used for excluding previous training samples.
    excluded_paths = {
        path.resolve()
        for path in a.exclude_manifest
    }

    if meta_path.resolve() in excluded_paths:
        p.error(
            "Output manifest must not overwrite an excluded manifest"
        )

    build_mixture(
        source_dir=a.source_dir,
        output=a.output,
        size=a.size,
        ratios=a.ratios,
        validation_path=validation_path,
        seed=a.seed,
        exclude_manifests=a.exclude_manifest,
        create_validation=a.create_validation,
        force=a.force,
    )


if __name__ == "__main__":
    main()
