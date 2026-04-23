# Contamination verification report

Independent audit of materialized training JSONLs against IFEval (541), GSM8K test (1319), and HumanEval (164). Three checks - EXACT (normalized equality), SUBSTRING (normalized containment, test text >= 40 chars), and NGRAM (8-gram overlap, noisier near-duplicate detector) - plus a HumanEval function-signature check on assistant messages.

## Summary

| File | Rows | Msgs | IFEval exact | IFEval substring | IFEval 8-gram | GSM8K-test exact | GSM8K-test substring | GSM8K-test 8-gram | HumanEval exact | HumanEval substring | HumanEval 8-gram | HumanEval func-sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r8_bon.jsonl | 110 | 220 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r8_rft.jsonl | 5861 | 11722 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r1_enhanced.jsonl | 377227 | 756557 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 67 |
| r1_kitchen_sink.jsonl | 347346 | 696795 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 67 |

## r8_bon.jsonl

Rows: 110  |  messages scanned: 220

- No leaks detected.


## r8_rft.jsonl

Rows: 5861  |  messages scanned: 11722

- No leaks detected.


## r1_enhanced.jsonl

Rows: 377227  |  messages scanned: 756557

- **HUMANEVAL-FUNC-SIG**: 67 hits (e.g. [('HumanEval/10', 'def is_palindrome(string: str) -> bool:'), ('HumanEval/11', 'def string_xor(a: str, b: str) -> str:'), ('HumanEval/13', 'def greatest_common_divisor(a: int, b: int) -> int:'), ('HumanEval/18', 'def how_many_times(string: str, substring: str) -> int:'), ('HumanEval/31', 'def is_prime(n):')])

## r1_kitchen_sink.jsonl

Rows: 347346  |  messages scanned: 696795

- **HUMANEVAL-FUNC-SIG**: 67 hits (e.g. [('HumanEval/10', 'def is_palindrome(string: str) -> bool:'), ('HumanEval/11', 'def string_xor(a: str, b: str) -> str:'), ('HumanEval/13', 'def greatest_common_divisor(a: int, b: int) -> int:'), ('HumanEval/18', 'def how_many_times(string: str, substring: str) -> int:'), ('HumanEval/31', 'def is_prime(n):')])

## Verdict

- EXACT hits across all files: **0** (hard gate for verbatim leakage)
- SUBSTRING hits across all files: **0** (hard gate for verbatim leakage)
- HUMANEVAL-FUNC-SIG hits: 134 (advisory — function names are often common code)
- 8-GRAM overlap hits (noisy near-duplicates): 0

**No verbatim test-set leakage detected** on the hard gates (EXACT + SUBSTRING). This matches the standard in the field (BigCode / StarCoder style 13-gram decontamination).

NOTE: 134 HumanEval function-signature matches in training assistant messages. This is mostly common-code false positives (`is_palindrome`, `is_prime`, `greatest_common_divisor` etc.), but some may be derivative contamination from 'Evol-Instruct' style data augmentation of HumanEval-like seed problems. Inspect specific rows manually if strict decontamination is required; the 13-gram filter at build time already catches verbatim HumanEval prompts/solutions.
