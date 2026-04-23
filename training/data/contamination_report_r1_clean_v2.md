# DEEP contamination audit report

This audit extends the basic verify_no_test_leak.py by extracting multiple distinctive fingerprints per test item — each `def` signature, each `>>> example` call, each `>>> expected_output` line, each docstring sentence, each canonical-solution line — and checks each as a SUBSTRING against every training JSONL. Designed to catch derivative contamination (e.g. Evol-Instruct-paraphrased test problems that preserve distinctive internal elements).

Fingerprints extracted: IFEval=1623  GSM8K-test=5104  HumanEval=763.

## Summary: unique leaked test items per category per file

| File | Rows | IFEval items | IFEval matches | GSM8K-test items | GSM8K-test matches | HumanEval items | HumanEval matches |
|---|---|---|---|---|---|---|---|
| r1_clean_v2.jsonl | 332218 | 11 | 11 | 1 | 1 | 12 | 13 |

## r1_clean_v2.jsonl

- **IFEval / last_sent**: 11 matches across 11 unique test items. Items (first 10): ['IFEval/1367', 'IFEval/1897', 'IFEval/2097', 'IFEval/2239', 'IFEval/227', 'IFEval/2273', 'IFEval/2417', 'IFEval/281', 'IFEval/2825', 'IFEval/3536']. Sample patterns: ['Wrap your entire response in double quotation marks', '" Put your entire response in double quotation marks', 'Put your entire response in double quotation marks']
- **GSM8K-test / question_first_sent**: 1 matches across 1 unique test items. Items (first 10): ['GSM8K-test/1216']. Sample patterns: ['A bus has a capacity of 200 people']
- **HumanEval / docstring_sent_0**: 5 matches across 5 unique test items. Items (first 10): ['HumanEval/133', 'HumanEval/147', 'HumanEval/158', 'HumanEval/90', 'HumanEval/94']. Sample patterns: ['You are given a list of numbers', 'Write a function that accepts a list of strings', 'You are given a list of integers']
- **HumanEval / solution_line_0**: 5 matches across 5 unique test items. Items (first 10): ['HumanEval/105', 'HumanEval/155', 'HumanEval/18', 'HumanEval/4', 'HumanEval/55']. Sample patterns: ['mean = sum(numbers) / len(numbers)', 'for i in range(len(string) - len(substring) + 1):', 'return fib(n - 1) + fib(n - 2)']
- **HumanEval / sig**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/13']. Sample patterns: ['def greatest_common_divisor(a:']
- **HumanEval / docstring_sent_1**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/97']. Sample patterns: ['Assume the input is always valid']
- **HumanEval / solution_line_1**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/18']. Sample patterns: ['if string[i:i+len(substring)] == substring:']

## Verdict

- Unique leaked IFEval test items: **11** / 541
- Unique leaked GSM8K-test items: **1** / 1319
- Unique leaked HumanEval items: **12** / 164

**24 unique test items have fingerprint overlap** with training data. Review the per-file detail above to determine whether these are genuine derivative leaks or benign common-code matches. Ratio matters: ~1-5% of HumanEval is unavoidable (common utility functions); >10% warrants removing the offending training source.
