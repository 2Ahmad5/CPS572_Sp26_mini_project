# TA-approved rigor: 8-gram training-vs-test audit

Applied the TA-approved 8-gram word overlap filter (reference: Piazza interaction where Prof. Bhuwan Dhingra approved this method as sufficient for contamination filtering against GSM8K-test). We extend it to all three held-out benchmarks (google/IFEval, openai/gsm8k test, openai/openai_humaneval) and run it against the full R16-lineage training artifacts.

Index sizes: IFEval=15388  GSM8K-test=123995  HumanEval=12780 unique 8-grams.

## Summary: training rows flagged by the TA-approved filter

| Artifact | Rows | Flagged IFEval | Flagged GSM8K-test | Flagged HumanEval |
|---|---|---|---|---|
| r1_clean.jsonl | 317218 | 2632 (0.83%) | 3753 (1.18%) | 4623 (1.46%) |
| r1_clean_v2.jsonl | 332218 | 2636 (0.79%) | 3753 (1.13%) | 5108 (1.54%) |
| r13_bon.jsonl | 113 | 0 (0.00%) | 0 (0.00%) | 12 (10.62%) |
| r14_rsft.jsonl | 355 | 0 (0.00%) | 0 (0.00%) | 32 (9.01%) |
| r17_bon.jsonl | 351 | 0 (0.00%) | 0 (0.00%) | 33 (9.40%) |
| kodcode_candidate.jsonl | 30000 | 7 (0.02%) | 0 (0.00%) | 975 (3.25%) |
| _audit_argilla_filtered.jsonl | 56339 | 49904 (88.58%) | 0 (0.00%) | 0 (0.00%) |
| _audit_mbpp_rl.jsonl | 120 | 0 (0.00%) | 0 (0.00%) | 15 (12.50%) |
| _audit_gsm8k_train.jsonl | 7473 | 0 (0.00%) | 812 (10.87%) | 1 (0.01%) |

## r1_clean.jsonl

- **IFEval**: 2632 rows flagged. First row indices: [375, 387, 491, 577, 610, 652, 866, 897, 909, 1058].
  - Sample matching 8-grams: ['please include the exact phrase in your response', 'please include the exact phrase in your response', 'what is the value of a b c', 'the number of words in all capital letters', 'the number of words in all capital letters']
- **GSM8K-test**: 3753 rows flagged. First row indices: [16, 116, 268, 563, 674, 752, 1014, 1036, 1231, 1265].
  - Sample matching 8-grams: ['we need to determine how many days it', 'calculate the total amount of money she spent', 'the number of days to find the total', 'since there are 12 months in a year', 'the total number of employees in the company']
- **HumanEval**: 4623 rows flagged. First row indices: [0, 77, 140, 145, 152, 184, 224, 266, 388, 396].
  - Sample matching 8-grams: ['1 2 3 4 5 6 7 8', 'j in range n if grid i j', 'should return true otherwise it should return false', 'if num i 0 return false return true', '3 for i in range 2 n 1']

## r1_clean_v2.jsonl

- **IFEval**: 2636 rows flagged. First row indices: [58, 156, 504, 777, 1011, 1029, 1105, 1116, 1121, 1301].
  - Sample matching 8-grams: ['can you give me some advice on how', 'wrapped in double angular brackets i e title', 'a postscript starting with p s at the', 'wrapped in double angular brackets i e title', 'wrapped in double angular brackets i e title']
- **GSM8K-test**: 3753 rows flagged. First row indices: [19, 32, 131, 183, 197, 408, 616, 620, 808, 968].
  - Sample matching 8-grams: ['what is the combined length of time in', 'in the first week in the second week', 'minutes how long will it take her to', 'there are 7 days in a week so', 'is 8 x 2 8 2 16 16']
- **HumanEval**: 5108 rows flagged. First row indices: [24, 29, 57, 232, 413, 441, 528, 530, 582, 674].
  - Sample matching 8-grams: ['the function should return a list of strings', '2 3 4 5 6 7 8 9', 'function that takes a string as input and', '1 2 3 4 5 6 7 8', '2 3 4 5 6 7 8 9']

## r13_bon.jsonl

- **HumanEval**: 12 rows flagged. First row indices: [3, 6, 20, 32, 37, 38, 40, 44, 53, 72].
  - Sample matching 8-grams: ['range 2 int n 0 5 1 if', 'i in range 1 n 1 if i', '1 2 3 4 5 6 7 8', '0 1 2 3 4 5 6 7', '1 2 3 4 5 6 7 8']

## r14_rsft.jsonl

- **HumanEval**: 32 rows flagged. First row indices: [2, 17, 29, 37, 40, 44, 49, 57, 68, 72].
  - Sample matching 8-grams: ['if num i 0 return false return true', '1 2 3 4 5 6 7 8', '0 1 2 3 4 5 6 7', '1 2 3 4 5 6 7 8', 'y for x y in zip a b']

## r17_bon.jsonl

- **HumanEval**: 33 rows flagged. First row indices: [3, 17, 24, 29, 35, 43, 53, 82, 115, 118].
  - Sample matching 8-grams: ['if num i 0 return false return true', '1 2 3 4 5 6 7 8', 'for i in range len lst for j', '0 1 2 3 4 5 6 7', '1 2 3 4 5 6 7 8']

## kodcode_candidate.jsonl

- **IFEval**: 7 rows flagged. First row indices: [10145, 11208, 15859, 15949, 17054, 20290, 22015].
  - Sample matching 8-grams: ['hoping you could help me out with a', 'can you give me some advice on how', 'is there anything else i can help with', 'is there anything else i can help with', 'can you give me some advice on how']
- **HumanEval**: 975 rows flagged. First row indices: [6, 105, 144, 169, 172, 189, 211, 265, 301, 308].
  - Sample matching 8-grams: ['your task is to write a function that', 'for i in range n for j in', '1 for j in range 1 i 1', 'your task is to write a function that', '1 2 3 4 5 6 7 8']

## _audit_argilla_filtered.jsonl

- **IFEval**: 49904 rows flagged. First row indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
  - Sample matching 8-grams: ['highlighted section your entire response should be in', 'response finish your response with this exact phrase', 'your answer must contain a title wrapped in', 'your answer must contain a title wrapped in', 'your answer must contain a title wrapped in']

## _audit_mbpp_rl.jsonl

- **HumanEval**: 15 rows flagged. First row indices: [6, 9, 18, 28, 29, 38, 40, 45, 55, 58].
  - Sample matching 8-grams: ['1 for j in range 1 i 1', '1 2 3 4 5 6 7 8', '1 2 3 4 5 6 7 8', 'n for j in range i 1 n', '0 1 2 3 4 5 6 7']

## _audit_gsm8k_train.jsonl

- **GSM8K-test**: 812 rows flagged. First row indices: [11, 13, 20, 35, 58, 67, 68, 75, 86, 95].
  - Sample matching 8-grams: ['because 35 7 35 7 5 5 5', 'cost 10 x 2 10 2 20 20', 'fewer rose stamps than truck stamps how many', 'x 60 100 60 60 100 36 36', 'x 2 3 60 2 3 40 40']
- **HumanEval**: 1 rows flagged. First row indices: [4610].
  - Sample matching 8-grams: ['1 2 3 4 1 2 3 4']

## Grand totals

- Total training rows across all audited artifacts: **744,187**
- Rows flagged on IFEval 8-gram overlap: **55179** (7.4147%)
- Rows flagged on GSM8K-test 8-gram overlap: **8318** (1.1177%)
- Rows flagged on HumanEval 8-gram overlap: **10799** (1.4511%)

**74296 rows flagged across all artifacts.** Each flagged row must be inspected to determine if it is real contamination or a coincidental benign 8-gram (e.g. common constraint phrases, generic math setup, standard Python idioms). See per-file detail above for sample matches.