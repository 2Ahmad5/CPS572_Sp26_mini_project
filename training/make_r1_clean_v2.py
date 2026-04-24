"""
Assemble r1_clean_v2.jsonl = r1_clean.jsonl (base) + 15K filtered KodCode rows.

Used as the SFT base for R15 (and R16, which RL's on top of R15).

Usage:
    python -m training.make_r1_clean_v2
"""

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "training" / "data"

SEED = 42
N_KODCODE_ROWS = 15_000


def main():
    r1_clean = DATA / "r1_clean.jsonl"
    kodcode = DATA / "kodcode_candidate.jsonl"
    out = DATA / "r1_clean_v2.jsonl"

    assert r1_clean.exists(), f"missing: {r1_clean} — run `python -m training.build_dataset --config r1_clean --output {r1_clean}`"
    assert kodcode.exists(), f"missing: {kodcode} — run `python -m training.prep_candidate_datasets --which kodcode --kodcode_max 30000`"

    rows = [ln for ln in r1_clean.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"r1_clean: {len(rows)} rows")

    kod_rows = [ln for ln in kodcode.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"kodcode_candidate: {len(kod_rows)} rows")

    random.Random(SEED).shuffle(kod_rows)
    add = kod_rows[:N_KODCODE_ROWS]
    rows.extend(add)
    print(f"appending {len(add)} KodCode rows; total = {len(rows)}")

    random.Random(SEED + 1).shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        for line in rows:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
