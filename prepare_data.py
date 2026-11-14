"""Download Tulu 3, filter task-relevant data, and save clean JSONL files."""

import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path


DATASET = "ali-elganzory/tulu-3-sft-mixture-decontaminated"

CATEGORIES = ("if", "math", "coding")

# Only retain sources relevant to the three evaluation tasks.
# No general, safety, multilingual, or GSM8K data.
SOURCE_MAPPING = {
    # Instruction following
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980": "if",

    # Math
    "ai2-adapt-dev/personahub_math_v5_regen_149960": "math",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated": "math",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k": "math",

    # Coding
    "ai2-adapt-dev/personahub_code_v2_34999": "coding",
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": "coding",
}


# ---------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------

def classify_source(source):
    return SOURCE_MAPPING.get(source)


# ---------------------------------------------------------------------
# Basic conversation validation
# ---------------------------------------------------------------------

def valid_conversation(messages):
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    if not all(
        isinstance(m, dict)
        and m.get("role") in {"system", "user", "assistant"}
        and isinstance(m.get("content"), str)
        and m["content"].strip()
        for m in messages
    ):
        return False

    roles = [m["role"] for m in messages]

    if roles[0] == "system":
        roles = roles[1:]

    return (
        len(roles) >= 2
        and len(roles) % 2 == 0
        and all(
            r == ("user" if i % 2 == 0 else "assistant")
            for i, r in enumerate(roles)
        )
    )


def single_turn(messages):
    """Allow an optional system message followed by one user-assistant pair."""
    roles = [m["role"] for m in messages]

    if roles[0] == "system":
        roles = roles[1:]

    return roles == ["user", "assistant"]


def get_user_prompt(messages):
    return "\n".join(
        m["content"] for m in messages
        if m["role"] == "user"
    )


def get_assistant_response(messages):
    return "\n".join(
        m["content"] for m in messages
        if m["role"] == "assistant"
    )


def prompt_hash(messages):
    """Normalize whitespace but preserve case and internal indentation."""
    prompt = get_user_prompt(messages)
    prompt = prompt.replace("\r\n", "\n").strip()

    return hashlib.sha256(
        prompt.encode("utf-8")
    ).digest()


# ---------------------------------------------------------------------
# Coding filters
# ---------------------------------------------------------------------

PYTHON_BLOCK = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)

GENERIC_CODE_BLOCK = re.compile(
    r"```\s*\n(.*?)```",
    flags=re.DOTALL,
)

NON_PYTHON_LANGUAGE = re.compile(
    r"\b(?:javascript|typescript|java|c\+\+|c#|rust|"
    r"golang|ruby|php|swift|kotlin|matlab|sql|"
    r"bash|powershell|html|css)\b",
    flags=re.IGNORECASE,
)

FUNCTION_GENERATION = re.compile(
    r"\b(?:write|create|implement|define|develop|"
    r"design|construct|generate|complete)\b"
    r".{0,150}"
    r"\b(?:function|method|algorithm)\b",
    flags=re.IGNORECASE | re.DOTALL,
)

CODE_NON_GENERATION = re.compile(
    r"\b(?:debug|fix|explain|refactor|optimize|translate|"
    r"review|analyse|analyze)\b.{0,80}"
    r"\b(?:code|function|program|implementation)\b",
    flags=re.IGNORECASE | re.DOTALL,
)

EXTERNAL_DEPENDENCIES = re.compile(
    r"\b(?:django|flask|fastapi|tensorflow|pytorch|"
    r"selenium|beautifulsoup|streamlit|tkinter|"
    r"web scraping|rest api|database|web application)\b",
    flags=re.IGNORECASE,
)


def extract_python_code(response):
    """Extract candidate Python implementations from a response."""
    blocks = PYTHON_BLOCK.findall(response)

    if not blocks:
        blocks = GENERIC_CODE_BLOCK.findall(response)

    if blocks:
        return blocks

    # Some samples contain raw Python without Markdown fences.
    return [response.strip()]


def has_python_function(response):
    """Check for a syntactically valid top-level Python function."""
    for code in extract_python_code(response):
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            continue

        if any(
            isinstance(node, ast.FunctionDef)
            for node in tree.body
        ):
            return True

    return False


def filter_coding(source, prompt, response):
    """
    Retain Python function-generation tasks similar to MBPP.

    Return None when accepted, otherwise a rejection reason.
    """
    # Explicitly non-Python prompts are not useful.
    if NON_PYTHON_LANGUAGE.search(prompt) and not re.search(
        r"\bpython\b", prompt, flags=re.IGNORECASE
    ):
        return "non_python"

    # Exclude code explanation, debugging, translation, etc.
    if CODE_NON_GENERATION.search(prompt):
        return "not_code_generation"

    # Prefer standalone functions without application dependencies.
    if EXTERNAL_DEPENDENCIES.search(prompt):
        return "external_dependencies"

    # MBPP evaluates standalone Python functions.
    if not has_python_function(response):
        return "no_valid_python_function"

    # Evol CodeAlpaca contains many heterogeneous coding tasks,
    # so require stronger evidence of function generation.
    if "codealpaca" in source:
        if not FUNCTION_GENERATION.search(prompt):
            return "not_function_generation"

    return None


# ---------------------------------------------------------------------
# Math filters
# ---------------------------------------------------------------------

PROOF_TASK = re.compile(
    r"^\s*(?:prove|show|demonstrate|establish)\s+that\b",
    flags=re.IGNORECASE,
)

TOOL_EXECUTION = re.compile(
    r"<tool_call>|<tool_response>|<\|python\|>|"
    r"```(?:python|py|output|result)\b|"
    r"\b(?:python interpreter|code interpreter)\b",
    flags=re.IGNORECASE,
)


def filter_math(source, prompt, response):
    """
    Prefer single-answer mathematical problems over proof tasks
    and solutions that depend on executing external tools.
    """
    if PROOF_TASK.search(prompt):
        return "proof_task"

    # NuminaMath-TIR solutions may depend on Python execution.
    # Keep only responses without explicit tool/code markers.
    if "numinamath" in source and TOOL_EXECUTION.search(response):
        return "tool_dependent_math"

    return None


# ---------------------------------------------------------------------
# Sample-level filtering
# ---------------------------------------------------------------------

def filter_sample(row, category):
    """
    Return None if the sample is accepted.
    Otherwise return a string describing why it was rejected.
    """
    messages = row["messages"]

    if not valid_conversation(messages):
        return "invalid_conversation"

    # All three evaluation tasks are single-turn.
    if not single_turn(messages):
        return "multi_turn"

    prompt = get_user_prompt(messages)
    response = get_assistant_response(messages)
    source = row["source"]

    if category == "coding":
        return filter_coding(source, prompt, response)

    if category == "math":
        return filter_math(source, prompt, response)

    # Persona IF is already designed around verifiable constraints.
    # No extra heuristic filtering at this stage.
    return None


# ---------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------

def record(row, category):
    return {
        "id": row["id"],
        "source": row["source"],
        "category": category,
        "messages": row["messages"],
    }


def dump_line(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main data preparation
# ---------------------------------------------------------------------

def prepare(ds, out, revision=None):
    """
    Save all cleaned, relevant examples to data/full/.

    No train/validation splitting, pooling, sampling, or mixture
    construction is performed here.
    """
    full = out / "full"
    full.mkdir(parents=True, exist_ok=True)

    source_counts = Counter(ds["source"])
    mapping = {
        source: classify_source(source)
        for source in source_counts
    }

    print("Source mapping:")

    for source, n in source_counts.most_common():
        category = mapping[source]

        print(
            f"{n:>9,}  "
            f"{str(category):>8}  "
            f"{source}"
        )

    # Catch unexpected source identifiers rather than silently
    # omitting an intended training dataset.
    missing = set(SOURCE_MAPPING) - set(source_counts)

    if missing:
        print("\nWARNING: expected sources not found:")

        for source in sorted(missing):
            print(f"  {source}")

    # Avoid producing an empty dataset for a required category.
    available_categories = {
        category
        for category in mapping.values()
        if category is not None
    }

    missing_categories = set(CATEGORIES) - available_categories

    if missing_categories:
        raise ValueError(
            f"No matching sources for: {sorted(missing_categories)}. "
            "Check SOURCE_MAPPING against the printed source names."
        )

    seen = set()

    counts = Counter()
    kept_by_source = Counter()
    skipped = Counter()
    skipped_by_source = {
        source: Counter()
        for source in source_counts
        if mapping[source] is not None
    }

    with ExitStack() as stack:
        # all.jsonl contains the union of the cleaned category files,
        # not the complete original Tulu archive.
        all_file = stack.enter_context(
            (full / "all.jsonl").open("w", encoding="utf-8")
        )

        files = {
            cat: stack.enter_context(
                (full / f"{cat}.jsonl").open(
                    "w", encoding="utf-8"
                )
            )
            for cat in CATEGORIES
        }

        for i, row in enumerate(ds):
            source = row["source"]
            cat = mapping[source]

            if cat is None:
                skipped["excluded_source"] += 1

            else:
                reason = filter_sample(row, cat)

                if reason is not None:
                    skipped[reason] += 1
                    skipped_by_source[source][reason] += 1

                else:
                    key = prompt_hash(row["messages"])

                    if key in seen:
                        skipped["duplicate_prompt"] += 1
                        skipped_by_source[source][
                            "duplicate_prompt"
                        ] += 1

                    else:
                        seen.add(key)

                        clean_row = record(row, cat)

                        dump_line(all_file, clean_row)
                        dump_line(files[cat], clean_row)

                        counts[cat] += 1
                        counts["all"] += 1
                        kept_by_source[source] += 1

            if (i + 1) % 100000 == 0:
                print(
                    f"Processed {i + 1:,}/{len(ds):,}; "
                    f"kept {counts['all']:,}"
                )

    # The categorized files must partition all.jsonl.
    assert counts["all"] == sum(
        counts[cat] for cat in CATEGORIES
    )

    assert counts["all"] + sum(skipped.values()) == len(ds)

    manifest = {
        "dataset": DATASET,
        "revision": revision,
        "original_size": len(ds),

        "source_mapping": mapping,
        "source_counts": dict(source_counts),

        "cleaned_counts": {
            cat: counts[cat]
            for cat in ("all", *CATEGORIES)
        },

        "kept_by_source": {
            source: kept_by_source[source]
            for source in source_counts
            if mapping[source] is not None
        },

        "skipped_counts": dict(skipped),

        "skipped_by_source": {
            source: dict(reasons)
            for source, reasons in skipped_by_source.items()
        },

        "missing_expected_sources": sorted(missing),
    }

    write_json(full / "manifest.json", manifest)

    print("\nCleaned dataset counts:")

    for cat in ("all", *CATEGORIES):
        print(f"{cat:>8}: {counts[cat]:,}")

    print("\nRetention by source:")

    for source, original in source_counts.most_common():
        if mapping[source] is None:
            continue

        kept = kept_by_source[source]
        rate = kept / original

        print(
            f"{kept:>8,}/{original:<8,} "
            f"({rate:6.1%})  {source}"
        )

    print("\nSkipped counts:")

    for reason, n in skipped.most_common():
        print(f"{n:>9,}  {reason}")

    print(f"\nSaved cleaned data to {full}")
    print("Done.")

    return manifest


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from dotenv import load_dotenv

    load_dotenv()

    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
    )

    p.add_argument(
        "--revision",
        default=None,
        help="HF commit SHA; defaults to current revision",
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing existing generated data",
    )

    args = p.parse_args()

    full = args.output_dir / "full"

    if full.exists() and any(full.iterdir()) and not args.force:
        p.error(
            "Output already exists; use --force to regenerate "
            "or choose another --output-dir"
        )

    hf_token = os.environ.get("HF_TOKEN")

    revision = args.revision or HfApi().dataset_info(
        DATASET,
        token=hf_token,
    ).sha

    print(f"Dataset: {DATASET}")
    print(f"Revision: {revision}")

    ds = load_dataset(
        DATASET,
        split="train",
        revision=revision,
        token=hf_token,
    )

    prepare(
        ds,
        args.output_dir,
        revision=revision,
    )


if __name__ == "__main__":
    main()