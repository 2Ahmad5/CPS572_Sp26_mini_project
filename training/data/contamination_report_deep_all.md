# DEEP contamination audit report

This audit extends the basic verify_no_test_leak.py by extracting multiple distinctive fingerprints per test item — each `def` signature, each `>>> example` call, each `>>> expected_output` line, each docstring sentence, each canonical-solution line — and checks each as a SUBSTRING against every training JSONL. Designed to catch derivative contamination (e.g. Evol-Instruct-paraphrased test problems that preserve distinctive internal elements).

Fingerprints extracted: IFEval=1623  GSM8K-test=5104  HumanEval=763.

## Summary: unique leaked test items per category per file

| File | Rows | IFEval items | IFEval matches | GSM8K-test items | GSM8K-test matches | HumanEval items | HumanEval matches |
|---|---|---|---|---|---|---|---|
| r8_bon.jsonl | 110 | 0 | 0 | 0 | 0 | 0 | 0 |
| r8_rft.jsonl | 5861 | 20 | 20 | 0 | 0 | 0 | 0 |
| r1_enhanced.jsonl | 377227 | 11 | 11 | 1 | 1 | 37 | 55 |
| r1_kitchen_sink.jsonl | 347346 | 11 | 11 | 1 | 1 | 37 | 54 |

## r8_bon.jsonl

- No fingerprint matches.


## r8_rft.jsonl

- **IFEval / last_sent**: 20 matches across 20 unique test items. Items (first 10): ['IFEval/1019', 'IFEval/1148', 'IFEval/13', 'IFEval/1300', 'IFEval/1446', 'IFEval/1601', 'IFEval/1659', 'IFEval/1730', 'IFEval/1902', 'IFEval/2216']. Sample patterns: ['Mark the beginning of each section with Section X', 'Mark the beginning of each section with SECTION X', 'Your answer must contain a title, wrapped in double angular brackets']

## r1_enhanced.jsonl

- **IFEval / last_sent**: 11 matches across 11 unique test items. Items (first 10): ['IFEval/1367', 'IFEval/1897', 'IFEval/2097', 'IFEval/2239', 'IFEval/227', 'IFEval/2273', 'IFEval/2417', 'IFEval/281', 'IFEval/2825', 'IFEval/3536']. Sample patterns: ['Do not use any commas in your response', 'Do not use any commas in your response', 'Do not use any commas in your response']
- **GSM8K-test / question_first_sent**: 1 matches across 1 unique test items. Items (first 10): ['GSM8K-test/1216']. Sample patterns: ['A bus has a capacity of 200 people']
- **HumanEval / sig**: 8 matches across 8 unique test items. Items (first 10): ['HumanEval/1', 'HumanEval/13', 'HumanEval/136', 'HumanEval/141', 'HumanEval/143', 'HumanEval/20', 'HumanEval/26', 'HumanEval/79']. Sample patterns: ['def separate_paren_groups(paren_string:', 'def find_closest_elements(numbers:', 'def words_in_sentence(sentence):']
- **HumanEval / solution_line_0**: 7 matches across 7 unique test items. Items (first 10): ['HumanEval/105', 'HumanEval/155', 'HumanEval/18', 'HumanEval/26', 'HumanEval/4', 'HumanEval/55', 'HumanEval/94']. Sample patterns: ['sorted_arr = sorted(arr, reverse=True)', 'return fib(n - 1) + fib(n - 2)', 'return (even_count, odd_count)']
- **HumanEval / docstring_sent_1**: 6 matches across 6 unique test items. Items (first 10): ['HumanEval/111', 'HumanEval/162', 'HumanEval/20', 'HumanEval/26', 'HumanEval/62', 'HumanEval/97']. Sample patterns: ["If 'text' is an empty string, return None", 'xs[0] + xs[1] * x + xs[2] * x^2 +', 'Keep order of elements left the same as in the input']
- **HumanEval / call_0**: 4 matches across 4 unique test items. Items (first 10): ['HumanEval/20', 'HumanEval/26', 'HumanEval/40', 'HumanEval/9']. Sample patterns: ['triples_sum_to_zero([1, 3, 5, 0])', 'remove_duplicates([1, 2, 3, 2, 4])', 'rolling_max([1, 2, 3, 2, 3, 4, 2])']
- **HumanEval / doctest**: 10 matches across 7 unique test items. Items (first 10): ['HumanEval/1', 'HumanEval/18', 'HumanEval/20', 'HumanEval/26', 'HumanEval/59', 'HumanEval/61', 'HumanEval/93']. Sample patterns: [">>> how_many_times('aaaa', 'aa')", '>>> correct_bracketing(")(()")', ">>> encode('This is a message')"]
- **HumanEval / call_3**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/40']. Sample patterns: ['triples_sum_to_zero([2, 4, -5, 3, 9, 7])']
- **HumanEval / call_1**: 4 matches across 4 unique test items. Items (first 10): ['HumanEval/116', 'HumanEval/40', 'HumanEval/42', 'HumanEval/47']. Sample patterns: ['sort_array([-2, -3, -4, -5, -6])', 'triples_sum_to_zero([1, 3, -2, 1])', 'median([-10, 4, 6, 1000, 10, 20])']
- **HumanEval / docstring_sent_0**: 12 matches across 12 unique test items. Items (first 10): ['HumanEval/10', 'HumanEval/104', 'HumanEval/109', 'HumanEval/133', 'HumanEval/147', 'HumanEval/158', 'HumanEval/30', 'HumanEval/38', 'HumanEval/47', 'HumanEval/85']. Sample patterns: ['Return median of elements in the list l', 'Test if given string is a palindrome', 'Given a list of positive integers x']
- **HumanEval / solution_line_1**: 2 matches across 2 unique test items. Items (first 10): ['HumanEval/18', 'HumanEval/64']. Sample patterns: ['if string[i:i+len(substring)] == substring:', "if s[-1] == 'y' or s[-1] == 'Y':"]
- **HumanEval / call_2**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/40']. Sample patterns: ['triples_sum_to_zero([1, 2, 3, 7])']

## r1_kitchen_sink.jsonl

- **IFEval / last_sent**: 11 matches across 11 unique test items. Items (first 10): ['IFEval/1367', 'IFEval/1897', 'IFEval/2097', 'IFEval/2239', 'IFEval/227', 'IFEval/2273', 'IFEval/2417', 'IFEval/281', 'IFEval/2825', 'IFEval/3536']. Sample patterns: ['Do not use any commas in your response', 'Do not use any commas in your response', 'Do not use any commas in your response']
- **GSM8K-test / question_first_sent**: 1 matches across 1 unique test items. Items (first 10): ['GSM8K-test/1216']. Sample patterns: ['A bus has a capacity of 200 people']
- **HumanEval / sig**: 8 matches across 8 unique test items. Items (first 10): ['HumanEval/1', 'HumanEval/13', 'HumanEval/136', 'HumanEval/141', 'HumanEval/143', 'HumanEval/20', 'HumanEval/26', 'HumanEval/79']. Sample patterns: ['def separate_paren_groups(paren_string:', 'def find_closest_elements(numbers:', 'def words_in_sentence(sentence):']
- **HumanEval / solution_line_0**: 7 matches across 7 unique test items. Items (first 10): ['HumanEval/105', 'HumanEval/155', 'HumanEval/18', 'HumanEval/26', 'HumanEval/4', 'HumanEval/55', 'HumanEval/94']. Sample patterns: ['sorted_arr = sorted(arr, reverse=True)', 'return fib(n - 1) + fib(n - 2)', 'return (even_count, odd_count)']
- **HumanEval / docstring_sent_1**: 6 matches across 6 unique test items. Items (first 10): ['HumanEval/111', 'HumanEval/162', 'HumanEval/20', 'HumanEval/26', 'HumanEval/62', 'HumanEval/97']. Sample patterns: ["If 'text' is an empty string, return None", 'xs[0] + xs[1] * x + xs[2] * x^2 +', 'Keep order of elements left the same as in the input']
- **HumanEval / doctest**: 10 matches across 7 unique test items. Items (first 10): ['HumanEval/1', 'HumanEval/18', 'HumanEval/20', 'HumanEval/26', 'HumanEval/59', 'HumanEval/61', 'HumanEval/93']. Sample patterns: [">>> how_many_times('aaaa', 'aa')", ">>> encode('This is a message')", '>>> largest_prime_factor(13195)']
- **HumanEval / call_0**: 4 matches across 4 unique test items. Items (first 10): ['HumanEval/20', 'HumanEval/26', 'HumanEval/40', 'HumanEval/9']. Sample patterns: ['triples_sum_to_zero([1, 3, 5, 0])', 'remove_duplicates([1, 2, 3, 2, 4])', 'rolling_max([1, 2, 3, 2, 3, 4, 2])']
- **HumanEval / call_3**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/40']. Sample patterns: ['triples_sum_to_zero([2, 4, -5, 3, 9, 7])']
- **HumanEval / call_1**: 4 matches across 4 unique test items. Items (first 10): ['HumanEval/116', 'HumanEval/40', 'HumanEval/42', 'HumanEval/47']. Sample patterns: ['sort_array([-2, -3, -4, -5, -6])', 'triples_sum_to_zero([1, 3, -2, 1])', 'median([-10, 4, 6, 1000, 10, 20])']
- **HumanEval / docstring_sent_0**: 12 matches across 12 unique test items. Items (first 10): ['HumanEval/10', 'HumanEval/104', 'HumanEval/109', 'HumanEval/133', 'HumanEval/147', 'HumanEval/158', 'HumanEval/30', 'HumanEval/38', 'HumanEval/47', 'HumanEval/85']. Sample patterns: ['Return median of elements in the list l', 'Test if given string is a palindrome', 'Given a list of positive integers x']
- **HumanEval / call_2**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/40']. Sample patterns: ['triples_sum_to_zero([1, 2, 3, 7])']
- **HumanEval / solution_line_1**: 1 matches across 1 unique test items. Items (first 10): ['HumanEval/64']. Sample patterns: ["if s[-1] == 'y' or s[-1] == 'Y':"]

## Verdict

- Unique leaked IFEval test items: **31** / 541
- Unique leaked GSM8K-test items: **1** / 1319
- Unique leaked HumanEval items: **37** / 164

**69 unique test items have fingerprint overlap** with training data. Review the per-file detail above to determine whether these are genuine derivative leaks or benign common-code matches. Ratio matters: ~1-5% of HumanEval is unavoidable (common utility functions); >10% warrants removing the offending training source.
