"""Print a clean comparison table of R1 checkpoint eval results."""
import json
import glob

files = sorted(glob.glob("evaluation/submission_step*.json"))
header = ("step", "IF_final", "IF_strict", "IF_loose", "GSM8K", "HumanEval", "avg_norm")
print("{:<8} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*header))
print("-" * 80)

results = []
for fp in files:
    sub = json.load(open(fp))
    ifm = sub.get("ifeval", {}).get("metrics", {})
    gsm = sub.get("gsm8k", {}).get("metrics", {})
    he = sub.get("humaneval", {}).get("metrics", {})
    if_final = next((v for k, v in ifm.items() if k.endswith("/final_acc")), None)
    if_strict = next((v for k, v in ifm.items() if k.endswith("/prompt_strict_acc")), None)
    if_loose = next((v for k, v in ifm.items() if k.endswith("/prompt_loose_acc")), None)
    gsm_acc = next((v for k, v in gsm.items() if k.endswith("/accuracy")), None)
    he_acc = next((v for k, v in he.items() if k.endswith("/accuracy")), None)
    norm = None
    if None not in (if_final, gsm_acc, he_acc):
        norm = (if_final / 0.45 + gsm_acc / 0.50 + he_acc / 0.30) / 3
    step = fp.split("step")[1].replace(".json", "")
    results.append((step, if_final, if_strict, if_loose, gsm_acc, he_acc, norm))
    vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in (if_final, if_strict, if_loose, gsm_acc, he_acc, norm)]
    print(f"{step:<8} " + " ".join(f"{v:<10}" for v in vals))

print()
best = max(results, key=lambda r: r[6] if r[6] is not None else -1)
print(f"WINNER: step {best[0]}")
print(f"   IF_final={best[1]:.3f}  GSM8K={best[4]:.3f}  HumanEval={best[5]:.3f}  avg_norm={best[6]:.3f}")
print()
print("Baselines: IFEval 0.45  GSM8K 0.50  HumanEval 0.30")
print("avg_norm 1.00 = meets all 3 baselines exactly")
