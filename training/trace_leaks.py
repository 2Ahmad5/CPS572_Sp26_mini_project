"""Find which training rows contain specific HumanEval fingerprints, with context
so we can trace to the source dataset."""

import json

targets = [
    ">>> how_many_times('aaaa', 'aa')",
    "triples_sum_to_zero([1, 3, 5, 0])",
    ">>> encode('This is a message')",
    "Test if given string is a palindrome",
    "Return median of elements in the list l",
    "rolling_max([1, 2, 3, 2, 3, 4, 2])",
]

found = {t: [] for t in targets}
with open("training/data/r1_enhanced.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except:
            continue
        combined = " ".join(m.get("content", "") for m in row.get("messages", []))
        for t in targets:
            if t in combined and len(found[t]) < 2:
                found[t].append((i, row))

for t, hits in found.items():
    print(f"=== Pattern: {t!r} ===")
    print(f"  found in {len(hits)} rows")
    for i, row in hits[:1]:
        print(f"  Row {i}:")
        user_msg = next((m for m in row.get("messages", []) if m.get("role") == "user"), None)
        if user_msg:
            content = user_msg.get("content", "")
            idx = content.find(t)
            if idx >= 0:
                start = max(0, idx - 120)
                end = min(len(content), idx + len(t) + 120)
                print(f"    [user context]  ...{content[start:end]}...")
            else:
                print(f"    [user ({len(content)} chars)]: {content[:200]}")
    print()
