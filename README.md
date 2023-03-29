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




### Undersampling method with LogisticRegression
### Naive RandomOversampling: Oversampling method with LogisticRegression
The training dataset was resampled with the RandomOverSampling ML method from Scikit-learn. This method takes a small data set like the "high_risk" credit group of loans,
and it randomly samples all the data. Thus, it is reusing datapoints to make the minority data set the same size as the majority data set (the "low_risk" credit loan group).
Finally, when equally sizes data sets emerge, we can begin to make predictions.
Here are the results from the RandomOverSampling technique with LogistcRegression ML model.

- Balanced Accuracy Score: 
- Precision: 
- Recall: 



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


