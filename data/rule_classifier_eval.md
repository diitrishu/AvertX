# Rule Classifier Evaluation Report

**Generated:** 2026-09-04T10:51:08Z  
**Model:** `rule_classifier_lr_minilm v1.0.0`  
**Embedding:** `all-MiniLM-L6-v2` (384-dim)  
**Classifier:** `LogisticRegression` multinomial, `class_weight='balanced'`, `C=1.0`, `max_iter=1000`  

---

## 1. Data summary

| Split | Total rows | Valid-rule rows | Excluded (UNKNOWN/Unmapped/NaN) |
|---|---|---|---|
| train.csv | 927 | 450 | 477 |
| test.csv  | 199 | 24  | 175  |

### Training class distribution

| Life-Saving Rule | Train count | Warning |
|---|---|---|
| Bypassing Safety Controls | 30 |  |
| Confined Space | 44 |  |
| Driving | 27 |  |
| Energy Isolation | 90 |  |
| Hot Work | 55 |  |
| Line of Fire | 77 |  |
| Safe Mechanical Lifting | 43 |  |
| Work Authorisation | 33 |  |
| Working at Height | 51 |  |

> **Note on training data quality:**
> Of the 450 training rows, 418 (92%) are synthetic. The model relies heavily on synthetic data generated from templates. Real-world performance on genuinely novel incident narratives may be lower than these evaluation metrics suggest.

### Test class distribution

| Life-Saving Rule | Test count | Coverage |
|---|---|---|
| Bypassing Safety Controls | 1 | ⚠️ sparse (<5) |
| Confined Space | 0 | ❌ absent |
| Driving | 5 | ✅ present |
| Energy Isolation | 7 | ✅ present |
| Hot Work | 0 | ❌ absent |
| Line of Fire | 8 | ✅ present |
| Safe Mechanical Lifting | 0 | ❌ absent |
| Work Authorisation | 0 | ❌ absent |
| Working at Height | 3 | ⚠️ sparse (<5) |

> **Critical gap:** 4 rules have **zero** test examples: `Confined Space`, `Hot Work`, `Safe Mechanical Lifting`, `Work Authorisation`. Precision/recall/F1 for these rules cannot be estimated at all from this split. Do NOT treat their reported metrics as meaningful.

---

## 2. Overall test-set performance

| Metric | Value | Caveat |
|---|---|---|
| Overall accuracy | 0.542 | n=24 — very small sample |
| NEEDS_REVIEW fraction | 0.917 | (conf < 0.5 OR gap < 0.15) |
| Rows triggering low-confidence | 22/24 | — |
| Rows triggering low-gap (ambiguity) | 20/24 | — |

**Accuracy on confident predictions** (rows NOT flagged NEEDS_REVIEW, n=2): `0.500`

---

## 3. Per-class precision / recall / F1

| Life-Saving Rule | Precision | Recall | F1 | Test support | Reliability |
|---|---|---|---|---|---|
| Bypassing Safety Controls | 0.000 | 0.000 | 0.000 | 1 | ⚠️ unreliable (<5 test examples) |
| Confined Space | 0.000 | 0.000 | 0.000 | 0 | ❌ cannot estimate (0 test examples) |
| Driving | 1.000 | 0.600 | 0.750 | 5 | ✅ marginally reliable (≥5 examples) |
| Energy Isolation | 0.750 | 0.429 | 0.545 | 7 | ✅ marginally reliable (≥5 examples) |
| Hot Work | 0.000 | 0.000 | 0.000 | 0 | ❌ cannot estimate (0 test examples) |
| Line of Fire | 0.429 | 0.750 | 0.545 | 8 | ✅ marginally reliable (≥5 examples) |
| Safe Mechanical Lifting | 0.000 | 0.000 | 0.000 | 0 | ❌ cannot estimate (0 test examples) |
| Work Authorisation | 0.000 | 0.000 | 0.000 | 0 | ❌ cannot estimate (0 test examples) |
| Working at Height | 1.000 | 0.333 | 0.500 | 3 | ⚠️ unreliable (<5 test examples) |

---

## 4. NEEDS_REVIEW analysis

Contract thresholds: `CONFIDENCE_THRESHOLD = 0.5`, `AMBIGUITY_MARGIN = 0.15`  
(Both are **placeholders** pending calibration on a larger test set — see `backend/rule_output_contract.py` module docstring.)

| Trigger | Count | % of test |
|---|---|---|
| Low confidence only | 2 | 8.3% |
| Low gap only | 0 | 0.0% |
| Both triggers | 20 | 83.3% |
| **Total NEEDS_REVIEW** | **22** | **91.7%** |

Among the 22 NEEDS_REVIEW predictions, the top-1 class was correct in **12/22** cases (54.5%), indicating that the model often has the right answer but with insufficient certainty — consistent with the expected behaviour of balanced-weight LR on an imbalanced dataset.

---

## 5. Honest assessment of classifier reliability

> [!CAUTION]  
> **The test set is too small (n=24) to support confident claims about this classifier's real-world performance.** The overall accuracy figure of 0.542 and all per-class metrics should be treated as indicative only. A minimum of ~20 examples per class (180 total for 9 classes) would be needed for reliable per-class metric estimation. Do NOT cite these numbers as evidence that the classifier works well.

> [!WARNING]  
> **4 rules have zero test coverage**: Confined Space, Hot Work, Safe Mechanical Lifting, Work Authorisation. The classifier may have learned useful representations for these classes (they are all represented in training), but we have no empirical evidence either way.

> [!WARNING]  
> **2 rules have sparse test coverage (<5 examples)**: Bypassing Safety Controls, Working at Height. Per-class F1 for these rules is unreliable.

> [!IMPORTANT]  
> **Training data is 92% synthetic.** Synthetic narratives were generated from templates — they may not capture the linguistic diversity of real incident reports. The classifier's learned decision boundaries reflect template language patterns as much as genuine semantic differences between IOGP rules. Real-world accuracy is likely lower than these metrics suggest.

### What the classifier IS reliable enough for right now

- **Triage / pre-filtering**: flagging *candidate* rules for human review, not making final determinations. The NEEDS_REVIEW contract enforces this discipline.
- **High-confidence, unambiguous cases**: when `confidence > 0.5` AND `gap > 0.15`, the model's top prediction is worth showing to a reviewer as a starting hypothesis.
- **Ruling out clearly irrelevant rules**: a very low probability for a rule is informative even if the top prediction is uncertain.

### What the classifier is NOT reliable enough for right now

- **Autonomous rule assignment** without human sign-off.
- **Per-rule performance comparison** between rules with sparse test coverage.
- **Threshold calibration**: the `CONFIDENCE_THRESHOLD=0.5` and `AMBIGUITY_MARGIN=0.15` values in `backend/rule_output_contract.py` are placeholders and have not been derived from a calibration curve on this data.

### Next steps to improve reliability

1. **More labelled real incident data** — the single biggest lever. Even 20 real examples per class would transform this evaluation.
2. **Cross-validation on training set** — with n=450 training rows and k=5 folds, CV would give a more stable accuracy estimate than a single 24-row test set.
3. **Threshold calibration** via Platt scaling or isotonic regression on a held-out calibration split once the dataset is larger.
4. **Error analysis** on the NEEDS_REVIEW rows to understand which rule pairs the model most frequently confuses — may reveal taxonomy overlap issues.

---

*Report generated by `train_rule_classifier.py` at 2026-09-04T10:51:08Z.*