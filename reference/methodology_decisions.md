# Methodology decision log

| Topic | Stated in the paper | Our current choice | Unknown / limitation |
|---|---|---|---|
| Dataset | Pima Indians Diabetes Dataset; 768 records; 8 predictors + Outcome | Use the public CSV mirror maintained by Jason Brownlee; validate shape, class counts, and file hash | This is a mirror of the standard dataset, not a direct download from the UCI repository. The exact source-file version/hash used by the paper's authors is not stated. |
| Zero values | Zero values meaningless for some attributes; means/medians used by diabetic/non-diabetic group | Convert zero to missing in Glucose, BloodPressure, SkinThickness, Insulin, BMI; fit median imputation on training data only | Exact columns and choice of mean versus median are not fully specified |
| Engineered features | F9: BP > 80 AND Glucose > 105; F10: BP > 80 | Compute after training-fitted imputation | Whether thresholds were evaluated before or after imputation is unspecified |
| Normalization | Dataset was normalized | StandardScaler for KNN, SVM, ANN; none for RF/DT | Paper's scaler and target range/parameters are unspecified |

| Evaluation split | 70% training/validation and 30% testing | Use a stratified 70/30 split with a fixed seed; perform stratified cross-validation on the training partition only; reserve the held-out test partition for final evaluation | Exact split seed and validation procedure used by the authors are unspecified; our CV protocol is an implementation choice |
| ANN configuration | The paper does not report sufficient ANN hyperparameters to reproduce the exact network configuration | Use `alpha=0.01` and `tol=1e-3`, with the existing `max_iter=1000` and fixed random state; selected after comparing five configurations by 5-fold CV on the training partition | The selection is based on reliable convergence without a clear accuracy penalty relative to the alternatives tested; differences are within fold-to-fold variability, and changing alpha and tolerance together does not isolate their individual effects |
| ANN convergence validation | Not specified | Add a regression test that treats `ConvergenceWarning` as an error for each feature configuration on the fixed 70/30 training split; also verify convergence behavior in the app's CV workflow | Passing the regression test does not guarantee convergence on every possible split; the warning was absent in the latest observed app run |

| CV for stability | The exact split seed and validation procedure are unspecified | Stratified cross-validation on the training partition, with a configurable outer fold count (5 by default). Threshold selection uses training-only out-of-fold scores. Random Forest and ANN hyperparameters are selected separately using stratified cross-validation, with F1 as the refit metric. | The authors' exact CV protocol is unknown. App CV metrics are computed without hyperparameter tuning, whereas Random Forest and ANN held-out test metrics use separately tuned parameters. These values are therefore not directly comparable for those models. Hyperparameter tuning uses at most 3 folds. Threshold selection uses 5 folds for held-out test evaluation and up to 5 folds during outer CV. The 100-split imputation diagnostic is a separate sensitivity analysis. |
| Metrics | Accuracy, precision, recall, F1 are reported | Label positive-class and macro averages explicitly | Averaging convention is unspecified |
| Preprocessing sensitivity analysis | Class-wise means/medians are computed separately for diabetic and non-diabetic cases | Completed a diagnostic comparing a leakage-safe baseline, training-class-median imputation, and global label-informed mean/median variants across 100 identical stratified 70/30 splits per variant | The paper's prose does not fully establish whether or how true labels influenced held-out-row preprocessing. The diagnostic measures sensitivity and is not a valid estimate of deployment performance. |



## Paper Accuracy Investigation: Label-Informed Imputation

**Date:** 2026-10-21  
**Status:** Diagnostic completed; paper's exact preprocessing protocol remains unresolved.

### Research question

The paper by Nahzat and Yağanoğlu (2021) reports 88.31% accuracy for its Random Forest experiment. Our leakage-safe pipeline achieves substantially lower accuracy under repeated stratified train/test splits. This diagnostic investigates whether class-conditional imputation could explain part of that discrepancy.

### Experimental protocol

The diagnostic was run with:

```powershell
python -m scripts.paper_protocol_diagnostic --seeds 100 --config 10 --out results/paper_protocol_diagnostic.csv
```

The experiment used the Pima Indians Diabetes dataset (768 observations), Random Forest with scikit-learn defaults, the 10-feature configuration, and 100 stratified 70/30 train/test splits. Each variant was evaluated on the same split for a given seed.

### Results

| Variant | Mean accuracy | Standard deviation | 2.5th percentile | 97.5th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| `clean` | 0.7613 | 0.0233 | 0.7163 | 0.8054 | 0.8139 |
| `train_class_median` | 0.7431 | 0.0236 | 0.7013 | 0.7902 | 0.7965 |
| `leaky_median_5` | 0.8815 | 0.0178 | 0.8442 | 0.9177 | 0.9264 |
| `leaky_mean_5` | 0.8730 | 0.0179 | 0.8376 | 0.9091 | 0.9177 |
| `leaky_median_8` | 0.8833 | 0.0167 | 0.8571 | 0.9157 | 0.9221 |

The paired mean accuracy differences relative to `clean`, using identical splits, were:

| Variant | Mean difference |
|---|---:|
| `train_class_median` | -0.0182 |
| `leaky_median_5` | +0.1202 |
| `leaky_mean_5` | +0.1117 |
| `leaky_median_8` | +0.1220 |

The paper's Figure 4 reports the confusion matrix `[[139, 11], [16, 65]]`, corresponding to 204 correct predictions out of 231 and an accuracy of 88.31%. Its macro precision, recall, and F1 reconstructed from this matrix are approximately 0.876, 0.865, and 0.870, respectively.

The `leaky_median_5` and `leaky_median_8` variants achieve mean accuracy close to the paper's reported value. However, the paper reports a single test-set result, whereas this diagnostic summarizes 100 different splits. Similar accuracy alone does not establish replication.

### Where the accuracy gain occurs

Mean test accuracy, grouped by whether a row has at least one zero in the five commonly imputed columns:

| Variant | Rows with at least one zero | Rows without zeros |
|---|---:|---:|
| `clean` | 0.7459 | 0.7764 |
| `train_class_median` | 0.7169 | 0.7681 |
| `leaky_median_5` | 0.9946 | 0.7725 |
| `leaky_mean_5` | 0.9959 | 0.7547 |
| `leaky_median_8` | 0.9913 | 0.7791 |

Approximately 49.07% of test rows contained at least one zero in the five commonly imputed columns.

The global label-informed variants achieve almost perfect accuracy on rows with such zeros, while their performance on rows without zeros remains close to the clean baseline. This strongly suggests that the artificial advantage is associated with the imputation procedure rather than a general improvement in the Random Forest classifier.

### Interpretation and limitations

The `clean` variant is the primary reference for evaluating the leakage-safe prediction workflow. Its imputation statistics are learned from training data without using the target labels to choose replacement values.

The global label-informed variants calculate class-conditional imputation values using the true outcome labels across the entire dataset before splitting. Consequently, preprocessing of test observations depends on their actual outcomes. These variants introduce target leakage and are invalid for estimating performance on unseen observations.

The `leaky_median_8` variant additionally applies zero replacement to all eight original features, including `Pregnancies`, where zero is valid. It therefore differs from the project's normal missing-value convention in another respect.

The `train_class_median` variant uses class-specific medians for training rows but overall training medians for test rows. This inconsistent preprocessing makes it an exploratory diagnostic rather than a faithful or valid replication protocol.

The results demonstrate that label-informed preprocessing can produce a large artificial increase in accuracy. They are consistent with one possible explanation for the paper's reported performance, but they do **not** establish that the authors used target leakage. The paper's description does not specify enough implementation details to determine its exact imputation procedure.

The standard deviations and percentiles above describe variation across the evaluated splits; they are not confidence intervals for the paper's result.

### Decision

1. Retain the leakage-safe production pipeline as the authoritative prediction workflow.
2. Keep the label-informed variants confined to diagnostic investigations and clearly identify them as invalid for generalization estimates.
3. Do not tune the production models or alter preprocessing merely to reach the paper's reported accuracy.
4. Treat the paper's result as a reference point, not as a reproduction claim.
5. Preserve the per-seed results at `results/paper_protocol_diagnostic.csv` for future inspection and reproducibility.

**Conclusion:** The discrepancy between the paper's reported accuracy and the leakage-safe baseline remains unresolved. Label-informed imputation provides a plausible mechanism for obtaining similar accuracy, but the available evidence is insufficient to identify the paper's actual methodology or claim a successful replication.

### Expanded Diagnostic Follow-up: Feature Configurations and Shuffled Labels

**Date:** 2026-10-22  
**Status:** Completed; findings support the sensitivity analysis but do not establish the paper's actual preprocessing procedure.

#### Feature-configuration comparison

A subsequent diagnostic compared the leakage-safe pipeline across the 6-, 8-, and 10-feature configurations using Random Forest with default scikit-learn parameters. Each configuration was evaluated over 100 stratified 70/30 train/test splits using seeds 0–99, with the same split for corresponding variants.

The leakage-safe pipeline achieved mean accuracy of approximately:

| Feature configuration | Mean accuracy |
|---|---:|
| 6 features | 0.760 |
| 8 features | 0.759 |
| 10 features | 0.761 |

The small differences indicate that adding the additional features did not materially improve mean accuracy in this experiment. These results describe this specific model and evaluation protocol; they do not establish that the configurations are equivalent under every setting.

#### Shuffled-label canary

A separate diagnostic shuffled the target labels to remove their original relationship with the predictors. The clean pipeline achieved approximately 0.603 accuracy, compared with a majority-class baseline of approximately 0.651. The label-informed variants nevertheless achieved approximately 0.78–0.79 accuracy.

This is a leakage canary: when preprocessing uses outcome labels from the full dataset, it can create an apparent predictive advantage even when the original predictor–target relationship has been disrupted. Accuracy above the clean baseline in this experiment is not evidence of genuine predictive signal.

#### Interpretation

The expanded diagnostics reinforce the earlier finding that global class-conditional imputation can produce artificially high accuracy. They also show that the small differences between the leakage-safe 6-, 8-, and 10-feature configurations do not explain the discrepancy with the paper on their own.

These experiments do not establish that the paper's authors used target leakage. The paper does not provide enough implementation detail to identify the exact preprocessing, split, normalization, or model configuration used to obtain its reported results.

The shuffled-label experiment is a methodological stress test, not an estimate of real-world model performance. Its results should be interpreted alongside the leakage-safe baseline and the original sensitivity analysis, not as a new production-model result.

#### Decision

1. Retain the leakage-safe pipeline as the authoritative prediction workflow.
2. Keep label-informed preprocessing confined to clearly labeled diagnostic experiments.
3. Report the 6-, 8-, and 10-feature results as evidence that feature-count changes alone did not materially improve accuracy under this protocol.
4. Do not claim that the paper's authors used leakage or that the paper has been exactly reproduced.
5. Preserve the diagnostic CSV outputs and the scripts used to generate them so the results can be inspected and reproduced.
### Reference

Nahzat, S., & Yağanoğlu, M. (2021). *Diabetes Prediction Using Machine Learning Classification Algorithms*. European Journal of Science and Technology, 24, 53–59. https://doi.org/10.31590/ejosat.899716
