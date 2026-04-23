# DEEP contamination audit report

This audit extends the basic verify_no_test_leak.py by extracting multiple distinctive fingerprints per test item — each `def` signature, each `>>> example` call, each `>>> expected_output` line, each docstring sentence, each canonical-solution line — and checks each as a SUBSTRING against every training JSONL. Designed to catch derivative contamination (e.g. Evol-Instruct-paraphrased test problems that preserve distinctive internal elements).

Fingerprints extracted: IFEval=1623  GSM8K-test=5104  HumanEval=763.

## Summary: unique leaked test items per category per file

| File | Rows | IFEval items | IFEval matches | GSM8K-test items | GSM8K-test matches | HumanEval items | HumanEval matches |
|---|---|---|---|---|---|---|---|
| kodcode_candidate.jsonl | 30000 | 0 | 0 | 0 | 0 | 4 | 4 |
| acecode_candidate.jsonl | 30000 | 0 | 0 | 0 | 0 | 14 | 15 |

## kodcode_candidate.jsonl

- **HumanEval / docstring_sent_0**: 4 matches across 4 unique test items. Items (first 10): ['HumanEval/133', 'HumanEval/147', 'HumanEval/90', 'HumanEval/94']. Sample patterns: ['You are given a list of integers', 'You are given a list of integers', 'You are given a positive integer n']

## acecode_candidate.jsonl

- **HumanEval / solution_line_0**: 9 matches across 9 unique test items. Items (first 10): ['HumanEval/105', 'HumanEval/112', 'HumanEval/124', 'HumanEval/155', 'HumanEval/18', 'HumanEval/4', 'HumanEval/40', 'HumanEval/43', 'HumanEval/73']. Sample patterns: ['sorted_arr = sorted(arr, reverse=True)', 'for i in range(len(arr) // 2):', 'return (even_count, odd_count)']
- **HumanEval / solution_line_1**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/18']. Sample patterns: ['if string[i:i+len(substring)] == substring:']
- **HumanEval / docstring_sent_0**: 5 matches across 5 unique test items. Items (first 10): ['HumanEval/104', 'HumanEval/133', 'HumanEval/147', 'HumanEval/90', 'HumanEval/94']. Sample patterns: ['You are given a list of integers', 'You are given a list of integers', 'Given a list of positive integers x']

## Verdict

- Unique leaked IFEval test items: **0** / 541
- Unique leaked GSM8K-test items: **0** / 1319
- Unique leaked HumanEval items: **14** / 164

**14 unique test items have fingerprint overlap** with training data. Review the per-file detail above to determine whether these are genuine derivative leaks or benign common-code matches. Ratio matters: ~1-5% of HumanEval is unavoidable (common utility functions); >10% warrants removing the offending training source.
