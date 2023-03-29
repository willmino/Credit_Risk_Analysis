# Credit_Risk_Analysis
Supervised Machine Learning Model to Predict Credit Risk

## Purpoe
Several supervised machine learning (ML) models were ran to predict the credit risk of loans.

### Overview
Supervised ML models, such as LogisticRegression from the Scikit-learn library, and BalancedRandomForestClassifier and EasyEnsembleClassifier from the Imbalanced-Learn library, were compared for the accuracy, precision, and sensitivity of their predictions of loan credit risk.

Data Cleaning was performed on a raw data set. The data was split into a testing and training set. A variety of resampling methods including Oversampling, Undersampling, and 
a combination method of oversampling and undersampling (SMOTEENN) were employed to balance the data. ML models (like LogisticRegression, BalancedRandomForestClassifier, and EasyEnsembleClassifier) were fit to the resampled data and each model made predictions regarding loan credit risk.

The dataset in question is imbalanced. Out of 68817 loans in the data set, 68470 of the loans were low credit risk and 347 of the loans were high credit risk.
Problems may arise when ML models make predictions using an imbalanced data set. Thus, the data was stratified into subgroups during the splitting of the data into training and testing sets.

This code can be described by the following block::

`from sklearn.model_selection import train_test_split`

`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)`


The stratification process ensured that the proportion of each class of the target variable y, "high_risk" and "low_risk" is maintained in the training and testing datasets.
This helps prevent bias in the model to improve ML prediction accuracy.

## Results

Describe the balanced accuracy scores and the precision and recall scores of all six machine learning models.
Use a bulleted list for each resampling technique in the resampling .ipynb file.
Use a bulleted list for each ensemble ML model in the second .ipynb file.


### Naive RandomOversampling: Oversampling method with LogisticRegression
The training dataset was resampled with the RandomOverSampling ML method from Scikit-learn. This method takes a small data set like the "high_risk" credit group of loans,
and it randomly samples all the data. Thus, it is reusing datapoints to make the minority data set the same size as the majority data set (the "low_risk" credit loan group).
Finally, when equally sizes data sets emerge, we can begin to make predictions.
Here are the results from the RandomOverSampling technique with LogistcRegression ML model.

- Balanced Accuracy Score: 0.65
- Precision: 0.01
- Recall: 0.61
(Note that precision and recall mentioned above are in reference to the ML model predictions made for "high_risk" loans.)

The accuracy score of the random oversampling method was above 0.5, at a value of 0.65. This is not the best but can be valid in some situations.
In order for the model to be favorable, we need to also observe its Precision and Recall in making successful predictions for high_risk loans.
Precision is calculated as `Precision = tp/(tp + fp)`. The Precision in this case is 0.01 which is very low. The model makes 5335 false positive predictions.
The sheer number of false positives causes precision to be dramatically low.
Recall is calculated as `Recall = tp/(tp + fn)`. The Recall of the oversampling model is 0.61. This value is relatively low. Its more important 
to capture all of the truly high_risk loans. A high risk loan prediction model would be favorable if its Precision and Recall were high.
However, the Precision and Recall of the Oversampling model are low. Thus, this model is not favorable.


### SMOTE Oversampling model with LogisticRegression
The training dataset was resampled with the SMOTE oversamplingML method from Scikit-learn. SMOTE oversamling takes a small data set like the "high_risk" credit group of loans, and it creates new interpolated data points based on the existing small data set. This method may be favorable compared to RandomOverSampling because it does reuse data. Howeverm it may introduce problems with outlier data. For example, a data outlier from the "low_risk" group might actually meet the criteria of the "high_risk" group using the LogisticRegression classifier. Let't look at the SMOTE oversampling method in practice.
After applying SMOTE oversampling, our data sets achieved the same size and more favorable for the LogisticRegression model predictions.
Here are the results from the SMOTE Oversampling technique with LogistcRegression ML model.

- Balanced Accuracy Score: 0.62
- Precision: 0.01
- Recall: 0.59

The accuracy score of the SMOTE OVersampling method was above 0.5, at a value of 0.62. This is also not a favorable quality.
In order for the model to be favorable, we need to observe its Precision and Recall in making successful predictions for high_risk loans.
The Precision in this case is 0.01 which is very low. The model makes 6065 false positive predictions.
The sheer number of false positives causes Precision to be dramatically low.
The Recall of the oversampling model is 0.59. This value is relatively low. Again, Its more important 
to capture all of the truly high_risk loans. A high risk loan prediction model would be favorable if its Precision and Recall were high.
However, the Precision and Recall of the SMOTE Oversampling model are low. Thus, this model is not favorable. In fact, its less favorable than the RandomOverSampling model.


### ClusterCentroids Undersampling method with LogisticRegression
The training dataset was resampled with the ClusterCentroids ML method from Scikit-learn. This method takes a large data set like the "low_risk" credit group of loans,
and it randomly resamples the data to be smaller data set. Thus, the low_risk credit group will achieve the same sample size as the smaller high_risk credit loan group.
Finally, when equally sizes data sets emerge, we can begin to make predictions with the LogisticRegression model.
Here are the results from the ClusterCentroids undersampling technique with LogistcRegression ML model.

- Balanced Accuracy Score: 0.51
- Precision: 0.01
- Recall: 0.59

The accuracy score of the ClusterCentroids undersampling method LogisticRegression model was 0.51. This is very low with a similar accuracy score to the SMOTE undersampling method. The Precision of the dataset is very low again at 0.01. The Recall is also very low again at 0.59. 
The sensitivity of the model is above 50% at 0.59, being able to correctly predict over half of the high_risk loan groups.
However, the accuracy score, precision, and recall are still very low. This model is thus not favorable.


Use screenshots of your outputs tosupport your results.


## Summary

Summarize the results of the machine learning models.
Recommend a model to predict loan credit risk. If you choose to not recommend an ML mode, justify your reasoning.


## Sources
1. This code, which can be found in both .ipynb files:

`from sklearn.model_selection import train_test_split`

`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)`

comes from this webpage:
https://courses.bootcampspot.com/courses/2638/pages/18-dot-3-1-overview-of-logistic-regression?module_item_id=836986


2. Imbalanced-Learn  BalancedRandomForestClassifier

`from imblearn.ensemble import BalancedRandomForestClassifier`

`brf = BalancedRandomForestClassifier(random_state=1, n_estimators = 100)`

`brf.fit(X_train, y_train)`

`y_pred = brf.predict(X_test)`

source: https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html

3. Imbalanced-learn EasyEnsembleClassifier

`from imblearn.ensemble import EasyEnsembleClassifier`

`eec = EasyEnsembleClassifier(n_estimators=100,random_state=1)`

`eec.fit(X_train, y_train)`

`y_pred = eec.predict(X_test)`

source: https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html


