# Contamination verification report

Independent audit of materialized training JSONLs against IFEval (541), GSM8K test (1319), and HumanEval (164). Three checks - EXACT (normalized equality), SUBSTRING (normalized containment, test text >= 40 chars), and NGRAM (8-gram overlap, noisier near-duplicate detector) - plus a HumanEval function-signature check on assistant messages.

## Summary

| File | Rows | Msgs | IFEval exact | IFEval substring | IFEval 8-gram | GSM8K-test exact | GSM8K-test substring | GSM8K-test 8-gram | HumanEval exact | HumanEval substring | HumanEval 8-gram | HumanEval func-sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r8_bon.jsonl | 110 | 220 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| r8_rft.jsonl | 5861 | 11722 | 0 | 0 | 69 | 0 | 0 | 21 | 0 | 0 | 0 | 0 |

## r8_bon.jsonl

Rows: 110  |  messages scanned: 220

- **NGRAM / HumanEval**: 5 leaked test items (e.g. ['HumanEval/55/solution', 'HumanEval/63/solution', 'HumanEval/78/prompt', 'HumanEval/107/prompt', 'HumanEval/107/solution'])

## r8_rft.jsonl

Rows: 5861  |  messages scanned: 11722

- **NGRAM / IFEval**: 69 leaked test items (e.g. ['IFEval/1019', 'IFEval/1128', 'IFEval/1148', 'IFEval/1180', 'IFEval/1219'])
- **NGRAM / GSM8K-test**: 21 leaked test items (e.g. ['GSM8K-test/a19', 'GSM8K-test/q32', 'GSM8K-test/a48', 'GSM8K-test/a214', 'GSM8K-test/q238'])

## Verdict

- EXACT hits across all files: **0**
- SUBSTRING hits across all files: **0**
- HUMANEVAL-FUNC-SIG hits: **0**
- 8-GRAM overlap hits (noisy near-duplicates): 95

**No verbatim test-set leakage detected** in any materialized training JSONL. 8-gram overlap may still be nonzero (common phrases like 'the answer is', 'how many', function names), but no EXACT, SUBSTRING, or function-signature match was found.
