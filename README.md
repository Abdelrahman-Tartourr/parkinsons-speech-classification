# parkinsons-speech-classification

Binary classification of Parkinson's Disease (PD) vs. healthy controls from acoustic features of sustained vowel phonation, using a leakage-free scikit-learn pipeline with subject-grouped cross-validation.

**Best result:** Random Forest on the top 50 features — **74.2% macro F1 / 72.2% balanced accuracy** (5-fold subject-grouped cross-validation).

## Dataset

[UCI Parkinson's Disease Classification Data Set](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification) (Sakar et al., 2019, *Applied Soft Computing*).

- 756 voice recordings from **252 subjects** (188 PD patients, 64 healthy controls), each subject recorded **3 times**
- 754 acoustic features per recording: jitter, shimmer, MFCCs, wavelet-based (TQWT) measures, and more
- Class imbalance: ~3:1 PD to healthy

## Why this isn't a standard tabular classification problem

Each subject contributes 3 rows. Recordings from the same subject are highly correlated (same voice, same physiology), so a plain random train/test split can let recordings from one subject appear in *both* sets — the model partly "recognizes the subject" instead of learning genuine disease signal, and every reported metric ends up inflated. Every split and cross-validation fold in this project is done **by subject (`id`)**, not by row, using `GroupShuffleSplit` and `StratifiedGroupKFold`.

## Methodology

1. **Cleaning:** dropped 1 duplicate row, fixed two columns that were read as strings due to single corrupted characters (`'1.13#4'`, `'66.13744a406'`), dropped one column that was ~70% missing, and dropped 5 rows with remaining nulls (2 of which were corrupted across most of their columns). Final shape: 750 recordings × 752 features across 252 subjects.
2. **Split:** 80/20 subject-grouped train/test split (201 vs. 51 subjects, zero subject overlap).
3. **Pipeline:** `StandardScaler → SelectKBest(f_classif, k) → Classifier`, fit fresh on every training fold so no preprocessing step ever sees validation or test data.
4. **Model & feature-count comparison:** 4 algorithms (Logistic Regression, SVM-RBF, Random Forest, KNN) × 4 feature-selection sizes (50, 100, 200, all features), scored with 5-fold `StratifiedGroupKFold` on accuracy, balanced accuracy, and macro F1.
5. **Final evaluation:** best pipeline refit on the full training set, evaluated once on the held-out (unseen-subject) test set.

## Results

| k features | Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---|---|---|
| **50** | **Random Forest** | **0.831** | **0.722** | **0.742** |
| all | Random Forest | 0.833 | 0.712 | 0.739 |
| all | KNN | 0.826 | 0.712 | 0.733 |
| 200 | Random Forest | 0.824 | 0.708 | 0.732 |
| 100 | Random Forest | 0.814 | 0.697 | 0.717 |

(full table in the notebook)

Random Forest with 50 features is essentially tied with using all 752 features, so the smaller, more interpretable set was chosen as the final pipeline.

**Held-out test set** (51 unseen subjects, 152 recordings): 78.9% accuracy, 68.5% balanced accuracy.

![Confusion Matrix]([images/confusion_matrix.png](https://github.com/Abdelrahman-Tartourr/parkinsons-speech-classification/blob/main/confusion_matrix.png))

The gap between the 5-fold CV estimate (~72% balanced accuracy) and the single held-out split (~69%) is expected sampling noise from a small test set (51 subjects) — the CV number is the more reliable estimate of real-world performance.

![Feature Importance](images/feature_importance.png)

The most predictive features are dominated by delta/delta-delta energy and MFCC statistics — consistent with the literature on voice-based PD detection, which links reduced articulatory precision and energy-modulation control to PD-related motor symptoms.

## Limitations

- Single-source dataset (one recording protocol, one clinic) — results may not generalize to other microphones, languages, or recording conditions.
- No hyperparameter tuning was performed; a `GridSearchCV`/`RandomizedSearchCV` using the same grouped CV scheme would likely improve on these numbers.
- Held-out test set is small (51 subjects), so the single-split numbers carry real sampling noise.
- This is a methodology demonstration, **not a validated diagnostic tool**.

## Reproduction

```bash
git clone <this-repo-url>
cd parkinsons-speech-classification
pip install -r requirements.txt
jupyter notebook parkinsons-speech-classification.ipynb
```

Download `pd_speech_features.csv` from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification) and place it in the project root before running.


## Tech stack

Python · pandas · scikit-learn · matplotlib · seaborn
