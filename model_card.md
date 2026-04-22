# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained using scikit-learn's `RandomForestClassifier` with `random_state=42`. It was built as part of a machine learning pipeline to predict whether an individual's annual income exceeds $50,000 based on census data. 

## Intended Use
The purpose of this model is to classify individuals into two income brackets (greater than $50K or less than or equal to $50K) based on demographic and employment features from census data. 

## Training Data
The model was trained on the Census Bureau dataset (`census.csv`), which contains demographic and employment information for approximately 32,000 individuals. The data includes continuous features  (age, education-num, capital-gain, capital-loss, hours-per-week) and categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country). The dataset was split 80/20 into training and test sets using `train_test_split` with `random_state=42`. Categorical features were encoded using `OneHotEncoder` and the target label was binarized using `LabelBinarizer`.

## Evaluation Data
The model was evaluated on the 20% test split of the Census Bureau dataset, which was approximately 6,500 samples. The same `OneHotEncoder` and `LabelBinarizer` fitted on the training data were applied to the test data to avoid data leakage.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model was evaluated using precision, recall, and F1 score on the binary classification 
task of predicting whether income exceeds $50K.

Overall model performance on the test set:
- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

Slice performance was also computed across all categorical features. A sample of results from `slice_output.txt`:
- workclass: Federal-gov — Precision: 0.7971 | Recall: 0.7857 | F1: 0.7914
- workclass: Private — Precision: 0.7376 | Recall: 0.6404 | F1: 0.6856
- workclass: Without-pay — Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000

## Ethical Considerations
The dataset contains sensitive demographics including race, sex, and native country. These features are used as inputs for the model, which introduces the risk of the model learning and perpetuating historical biases present in the census data. Slice analysis shows performance varies across demographic groups, which should be carefully considered before any real-world deployment.

## Caveats and Recommendations
Several slices with very small sample sizes (e.g., workclass: Without-pay, Count: 4) show misleading perfect scores due to insufficient data. The training data reflects historical census patterns and may not generalize well to current population distributions.








