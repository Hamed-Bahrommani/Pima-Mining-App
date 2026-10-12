# Methodology decision log

| Topic | Stated in the paper | Our current choice | Unknown / limitation |
|---|---|---|---|
| Dataset | Pima Indians Diabetes Dataset; 768 records; 8 predictors + Outcome | Use a public mirror of the standard headerless 768-row CSV; validate shape and class counts | Exact source-file version/hash used by the authors is not stated |
| Zero values | Zero values meaningless for some attributes; means/medians used by diabetic/non-diabetic group | Convert zero to missing in Glucose, BloodPressure, SkinThickness, Insulin, BMI; fit median imputation on training data only | Exact columns and choice of mean versus median are not fully specified |
| Engineered features | F9: BP > 80 AND Glucose > 105; F10: BP > 80 | Compute after training-fitted imputation | Whether thresholds were evaluated before or after imputation is unspecified |
| Normalization | Dataset was normalized | StandardScaler for KNN, SVM, ANN; none for RF/DT | Paper's scaler and target range/parameters are unspecified |
| Evaluation split | 70% training/validation and 30% testing | Use a stratified 70/30 split with fixed seed, then repeated stratified CV for stability | Exact split seed and validation procedure are unspecified |
| Metrics | Accuracy, precision, recall, F1 are reported | Label positive-class and macro averages explicitly | Averaging convention is unspecified |
| Preprocessing sensitivity analysis | Class-wise means/medians are computed separately for diabetic and non-diabetic cases | Planned diagnostic protocol will make the label-informed nature explicit and compare it with the leakage-safe protocol on identical splits | The paper's prose does not fully pin down whether or how true labels influenced held-out-row preprocessing; this analysis measures sensitivity and is not a valid estimate of deployment performance |
