# DEEP contamination audit report

This audit extends the basic verify_no_test_leak.py by extracting multiple distinctive fingerprints per test item — each `def` signature, each `>>> example` call, each `>>> expected_output` line, each docstring sentence, each canonical-solution line — and checks each as a SUBSTRING against every training JSONL. Designed to catch derivative contamination (e.g. Evol-Instruct-paraphrased test problems that preserve distinctive internal elements).

Fingerprints extracted: IFEval=1623  GSM8K-test=5104  HumanEval=763.

## Summary: unique leaked test items per category per file

| File | Rows | IFEval items | IFEval matches | GSM8K-test items | GSM8K-test matches | HumanEval items | HumanEval matches |
|---|---|---|---|---|---|---|---|
| r8_bon.jsonl | 110 | 0 | 0 | 0 | 0 | 0 | 0 |
| r8_rft.jsonl | 5861 | 20 | 20 | 0 | 0 | 0 | 0 |

## r8_bon.jsonl

- No fingerprint matches.


## r8_rft.jsonl

- **IFEval / last_sent**: 20 matches across 20 unique test items. Items (first 10): ['IFEval/1019', 'IFEval/1148', 'IFEval/13', 'IFEval/1300', 'IFEval/1446', 'IFEval/1601', 'IFEval/1659', 'IFEval/1730', 'IFEval/1902', 'IFEval/2216']. Sample patterns: ['No capital letters are allowed', 'You can use markdown ticks such as ```', 'You can use markdown ticks such as ```']

## Verdict

- Unique leaked IFEval test items: **20** / 541
- Unique leaked GSM8K-test items: **0** / 1319
- Unique leaked HumanEval items: **0** / 164

**20 unique test items have fingerprint overlap** with training data. Review the per-file detail above to determine whether these are genuine derivative leaks or benign common-code matches. Ratio matters: ~1-5% of HumanEval is unavoidable (common utility functions); >10% warrants removing the offending training source.
