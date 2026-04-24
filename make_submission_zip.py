"""
Build the Gradescope code-submission ZIP.

What goes in: exactly the files a TA needs to reproduce R16 end-to-end from
base Llama-3.1-8B. Nothing more.

What stays out: audit scripts, intermediate reports, markdown iteration logs,
task-planning files, stale submission JSONs, evaluation transcripts, virtual
envs, training data JSONLs (rebuildable), logs, secrets.

Run:
    python make_submission_zip.py
    # writes submission_code.zip at repo root

Inspect before uploading:
    unzip -l submission_code.zip
"""

from __future__ import annotations

import fnmatch
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Files needed to reproduce R16 exactly. Grouped by purpose.
# ----------------------------------------------------------------------------

INCLUDE_FILES = [
    # ----- Training pipeline -----
    # Package marker
    "training/__init__.py",

    # Dataset construction (r1_clean, r1_clean_v2 configs)
    "training/data_sources.py",
    "training/build_dataset.py",

    # SFT driver + configs (r11_clean_8b for R15 base, r9_bon_8b reused for R13/R14 BoN-SFT)
    "training/sft_train.py",

    # RL driver + 3-way GRPO configs (r12_grpo_3way_clean_8b, r16_grpo_3way_clean_v2_8b)
    "training/rl_train.py",

    # RL environment classes (each verifier)
    "training/rl_ifeval.py",
    "training/rl_math_env.py",
    "training/rl_code_env.py",

    # Best-of-N rejection-sampling distillation driver
    "training/bon_run.py",

    # Helper to pre-filter new candidate datasets (KodCode / MBPP-full) before training
    "training/prep_candidate_datasets.py",

    # Concatenates r1_clean.jsonl + 15K filtered KodCode rows -> r1_clean_v2.jsonl
    # (the exact SFT data used for R15 / R16)
    "training/make_r1_clean_v2.py",

    # ----- Evaluation (TA-provided, included so repro is self-contained) -----
    "evaluation/__init__.py",
    "evaluation/eval_all.py",
    "evaluation/eval_ifeval.py",
    "evaluation/eval_gsm8k.py",
    "evaluation/eval_code.py",
    "evaluation/run_eval.sh",
    "evaluation/train_and_publish.py",

    # ----- Submission artifact -----
    "evaluation/submission.json",

    # ----- Setup / entry -----
    "README.md",
    "PROJECT.md",

    # ----- Dataset reproduction documentation -----
    # Lists every raw HF source, every preprocessing/filtering decision, and
    # the exact end-to-end commands to rebuild every training JSONL from
    # scratch. Includes determinism notes and integrity-verification summary.
    "DATA_REPRODUCTION.md",
]

# Files the TA might appreciate for reproduction but not strictly necessary.
# Off by default; flip KEEP_REPRO_HELPERS = True to include.
KEEP_REPRO_HELPERS = False
OPTIONAL_REPRO_HELPERS = [
    # RL post-run orchestrator (automates the "pick best checkpoint, eval, update
    # submission.json" step). Not strictly needed — commands are in the report.
    "training/post_r9.py",
]

# Explicit ignore patterns (even if someone's Git adds them later).
NEVER_INCLUDE_GLOBS = [
    # Secrets / env
    "*.env", ".env", ".env.*",
    # Caches / compiled
    "**/__pycache__/**", "*.pyc", "*.pyo",
    # Virtual env
    ".venv/**", "venv/**",
    # Logs
    "logs/**", "**/logs/**",
    # Huge training datasets — rebuildable from training/build_dataset.py + sources
    "training/data/**",
    # Eval inspect transcripts (MBs each)
    "evaluation/inspect-logs/**",
    # Intermediate per-run submission JSONs — only the canonical submission.json ships
    "evaluation/submission_*.json",
    # Per-step eval JSONs from the old temperature sweep / step sweep
    "evaluation/submission_step*.json",
    # Old baseline / experimental eval jsons
    "evaluation/r*_checkpoint_eval.json",
    "evaluation/checkpoint_info.json",
    # IDE / OS
    ".vscode/**", ".idea/**", ".DS_Store", "Thumbs.db",
    # Claude Code config
    ".claude/**",
    # Git internals
    ".git/**", ".gitignore",
    # All iteration / task / audit markdown (keeping only README.md, PROJECT.md above)
    "419iter.md", "423iter.md", "R14_TECHNIQUES.md", "SUBMISSION_READINESS.md",
    "interactionabtdata.md", "Finalstuff.md", "usage.md",
    "tasks/**",
    # Contamination audit scripts + reports (the user asked to exclude these)
    "training/verify_no_test_leak.py",
    "training/verify_no_test_leak_deep.py",
    "training/ta_standard_audit.py",
    "training/final_audit.py",
    "training/inspect_matches.py",
    "training/inspect_ngram_matches.py",
    "training/trace_leaks.py",
    "training/data/contamination_report*.md",
    "training/data/FINAL_contamination_report.md",
    "training/data/TA_STANDARD_contamination_report.md",
    "training/data/CONTAMINATION_VERDICT.md",
    # Stale helper scripts not used by R16
    "training/compare_r1.py",
    "training/eval_checkpoints.py",
    "training/rft_run.py",  # RFT was experimented with in R8, not in R16 critical path
    # Configs dir (empty / unused in R16)
    "training/configs/**",
    # The final report LaTeX itself goes in a different Gradescope upload (the PDF)
    "final_report_572.tex",
    "final_report_572.pdf",
    # This script itself
    "make_submission_zip.py",
]


def _matches_any(path: str, globs: list[str]) -> bool:
    """True if `path` matches any of the glob patterns (`**` recursive)."""
    for g in globs:
        # fnmatch doesn't understand `**`. Special-case: `foo/**` matches anything
        # under foo/, and `**/foo/**` matches foo/ anywhere.
        if g.endswith("/**"):
            prefix = g[:-3]
            if path == prefix or path.startswith(prefix + "/"):
                return True
        elif g.startswith("**/") and g.endswith("/**"):
            inner = g[3:-3]
            if (f"/{inner}/" in "/" + path + "/") or path.startswith(inner + "/"):
                return True
        elif g.startswith("**/"):
            if fnmatch.fnmatch(path, g[3:]) or fnmatch.fnmatch(path.split("/")[-1], g[3:]):
                return True
        else:
            if fnmatch.fnmatch(path, g):
                return True
    return False


def build_file_list() -> tuple[list[Path], list[str]]:
    """Return (list of files to include, list of missing required files)."""
    wanted = list(INCLUDE_FILES)
    if KEEP_REPRO_HELPERS:
        wanted.extend(OPTIONAL_REPRO_HELPERS)

    found: list[Path] = []
    missing: list[str] = []
    for rel in wanted:
        p = REPO / rel
        if p.exists():
            found.append(p)
        else:
            missing.append(rel)
    return found, missing


def main():
    out_zip = REPO / "submission_code.zip"

    files, missing = build_file_list()

    print("=" * 72)
    print("SUBMISSION CODE-ZIP BUILDER")
    print("=" * 72)
    print(f"Repo: {REPO}")
    print(f"Output: {out_zip}")
    print()

    if missing:
        print("WARNING: the following REQUIRED files are missing. Aborting.")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # Safety check: each file we plan to include must NOT match any
    # NEVER_INCLUDE_GLOBS pattern. Protects against accidental additions to
    # INCLUDE_FILES that overlap with the excluded categories.
    final_files = []
    for p in files:
        rel = p.relative_to(REPO).as_posix()
        if _matches_any(rel, NEVER_INCLUDE_GLOBS):
            # This is a deliberate duplicate in both lists — the INCLUDE
            # wins for specific paths but we still print a warning.
            print(f"  NOTE: {rel} is in INCLUDE but also matches an exclude glob; including anyway.")
        final_files.append(p)

    # Write zip
    print("Packaging files:")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in final_files:
            rel = p.relative_to(REPO).as_posix()
            zf.write(p, arcname=rel)
            size_kb = p.stat().st_size / 1024
            print(f"  + {rel}  ({size_kb:.1f} KB)")

    # Summary
    size_mb = out_zip.stat().st_size / 1024 / 1024
    print()
    print(f"Wrote {out_zip.name} ({size_mb:.2f} MB, {len(final_files)} files).")
    print()
    print("Sanity check — listing zip contents with 'python -m zipfile -l submission_code.zip':")
    print("-" * 72)
    import zipfile as _z
    with _z.ZipFile(out_zip) as zf:
        for info in zf.infolist():
            print(f"  {info.file_size:>10} bytes  {info.filename}")

    print()
    print("IMPORTANT: this zip contains ONLY the code + submission.json.")
    print("Before uploading, verify that:")
    print("  1. No files under training/data/ are in the zip (datasets are rebuilt by build_dataset.py).")
    print("  2. No files under evaluation/inspect-logs/ are in the zip.")
    print("  3. No submission_r*.json intermediate files are in the zip (only evaluation/submission.json).")
    print("  4. No secrets (.env) are in the zip.")


if __name__ == "__main__":
    main()
