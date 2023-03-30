# Credit_Risk_Analysis
Supervised Machine Learning Model to Predict Credit Risk

## Purpoe
Several supervised machine learning (ML) models were ran to predict the credit risk of loans.

### Overview
Supervised ML models, such as LogisticRegression from Scikit-learn, RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN, BalancedRandomForestClassifier and EasyEnsembleClassifier from the Imbalanced-Learn library, were compared for the accuracy, precision, and sensitivity of their predictions of loan credit risk.

Data Cleaning was performed on a raw data set. The data was split into a testing and training set. A variety of resampling methods were employed to balance the data. ML models were fit to the resampled data and each model made predictions regarding loan credit risk.

The dataset in question is imbalanced. Out of 68817 loans in the data set, 68470 of the loans were low credit risk and 347 of the loans were high credit risk.
Problems may arise when ML models make predictions using an imbalanced data set. Thus, the data was stratified into subgroups during the splitting of the data into training and testing sets.

This code can be described by the following block:

`from sklearn.model_selection import train_test_split`

`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)`


The stratification process ensured that the proportion of each class of the target variable y, "high_risk" and "low_risk" is maintained in the training and testing datasets.
This helps prevent bias in the model to improve ML prediction accuracy.

## Results
A variety of resampling machine learning algorithms were employed to make our low_risk and high_risk credit loan groups suitable for model predictions.
Some models were used in conjunction with LogisticRegression (RandomOverSampler, SMOTE Oversampling, ClusterCentroids Undersampling, and SMOTE_ENN).
Two additional Ensemble models resampled our data and performed their own predictions. (BalancedRandomForestClassifier and EasyEnsembleClassifier).


### Naive RandomOversampling: Oversampling method with LogisticRegression
The training dataset was resampled with the RandomOverSampling ML method from Scikit-learn. This method takes a small data set like the "high_risk" credit group of loans,
and it randomly samples all the data. Thus, it is reusing datapoints to make the minority data set the same size as the majority data set (the "low_risk" credit loan group).
Finally, when equally sized data sets emerge, we can begin to make predictions.
Here are the results from the RandomOverSampling technique with LogistcRegression ML model.

![RandomOverSampling](https://github.com/willmino/Credit_Risk_Analysis/blob/main/images/RandomOverSampling.png)

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
The training dataset was resampled with the SMOTE oversampling ML method from Scikit-learn. SMOTE oversampling takes a small data set like the "high_risk" credit group of loans, and it creates new interpolated data points based on the existing small data set. This method may be favorable compared to RandomOverSampling because it does not reuse data. However it may introduce problems with outlier data. For example, a data outlier from the "low_risk" group might actually meet the criteria of the "high_risk" group using the LogisticRegression classifier. Let't look at the SMOTE oversampling method in practice.
After applying SMOTE oversampling, our data sets achieved the same size and more favorable for the LogisticRegression model predictions.
Here are the results from the SMOTE Oversampling technique with LogistcRegression ML model.

![SMOTE](https://github.com/willmino/Credit_Risk_Analysis/blob/main/images/SMOTE.png)

- Balanced Accuracy Score: 0.62
- Precision: 0.01
- Recall: 0.59

The accuracy score of the SMOTE oversampling method was above 0.5, at a value of 0.62. This is also not a favorable quality.
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

![ClusterCentroids](https://github.com/willmino/Credit_Risk_Analysis/blob/main/images/ClusterCentroids.png)

- Balanced Accuracy Score: 0.51
- Precision: 0.01
- Recall: 0.59

The accuracy score of the ClusterCentroids undersampling method LogisticRegression model was 0.51. This is very low and unfavorable.
The Precision of the dataset is very low again at 0.01. The Recall is also very low again at 0.59. 
The only potentially favorable aspect of this model is the Recall at a value of 0.59, being able to correctly predict over half of the high_risk loan groups. But, its still low enough to be unfavorable. The accuracy score, precision, and recall are all still very low. This model is not favorable.


### SMOTEENN Combination Oversampling and Undersampling method with LogisticRegression
The training dataset was resampled with the SMOTEENN ML method from Scikit-learn. This method combines the SMOTE oversampling method with the Edited Nearest Neighbors (ENN)
algorithms. The minority class is oversampled with SMOTE. The resulting data is then cleaned with ENN. Since SMOTE creates interpolated data points depending on its nearest neighbors, sometimes the two nearest neighbors of a data point belong to two different classes. Thus, theis data point is dropped by the ENN component of SMOTE-ENN.

Here is an example of SMOTEENN in action. Let's say the majority low_risk credit loan group is the larger purple dataset, and the minority high_risk smaller dataset is the yellow group. The yellow group was oversampled with SMOTE. Note that there is an area of overlap in the dataset with yellow and purple points. This area is a mixture of low_risk and high_risk loans in the data set. These two different classes of loans have similar characteristics. Thus, it will make the regular SMOTE algorithm inaccurate.

To account for the mixed dataset, the ENN component of SMOTE-ENN drops the points with two nearest neighbors from different classes. Now, there are less mixed data points
in the trouble region which was highlighted earlier. The LogisticRegression model will now have a better chance of making more reliable predictions.

Thus, the high_risk credit group will achieve a similar sample size to the low_risk credit group due to oversampling, but not quite the same since the mixed data points are dropped due to ENN.
Finally, when comparable sized data sets emerge, we can begin to make predictions with the LogisticRegression model.
Here are the results from the SMOTEENN undersampling technique with LogistcRegression ML model.

![SMOTEENN](https://github.com/willmino/Credit_Risk_Analysis/blob/main/images/SMOTEENN.png)

- Balanced Accuracy Score: 0.62
- Precision: 0.01
- Recall: 0.70

The accuracy score of the SMOTEENN combination oversampling and undersampling method with the LogisticRegression model was 0.62. This is not the lowest out of all the accuracy scores of the previous methods. The Precision of the dataset is very low again at 0.01. However, the Recall is slightly higher this time at 0.70.
This means that the model detected 70% of the high_risk loans. 
Despite a medium-low level accuracy score of 0.62 and a vew low Precision value of 0.01, the SMOTEENN model is more favorable because of its Recall value of 0.70.


### BalancedRandomForestClassifier
The training dataset was resampled and predictions were made with the BalancedRandomForestClassifier ML method. This method makes small decision trees compared to the larger decision trees made by other algorithms. Small datasets are generated from randomly sampled data. This process is called bootstrapping. Then, a small decision tree is made based on each new dataset. A different subset of features is chosen for each small decision tree. So none of these small decision trees are intended to have the same feature data inputs (independent variables). The algorithm makes predictions by passing a single data point through each tree. Each tree will make a vote on what the prediction will be. The final prediction is determined by whatever label is classified as the majority of the votes. This voting process is called Aggregation. This overall process is called Bagging (Bootstrapping + Aggregation).

![BalancedRandomForest](https://github.com/willmino/Credit_Risk_Analysis/blob/main/images/BalancedRandomForest.png)

- Balanced Accuracy Score: 0.79
- Precision: 0.04
- Recall: 0.67

The BalancedRandomForestClassifier exhbits an accuracy score of 0.79. This is a good score. The precision is very low, but at a value of 0.04, its about 4 times higher
than the precision scores of any other model previously mentioned. The Recall is relatively high at 0.67. This model has some advantages over theothers in that 
it has a high accuracy score and has relatively high Recall. This model could be considered favorable for credit loan risk predictions.


### EasyEnsembleClassifier
The EasyEnsembleClassifer ML model was used to make credit loan risk predictions. This method takes the majority class of data and allocates it into subsets of equal size. The size of each new subgroup of the split-up majority class of data is equal to the minority class of data. The classifier is trained on each subset of the split-up majority data class and the minority data class. Thus, an ensemble of classifiers is generated and predictions can be made by the ensemble. The final prediction is made by taking the weighted average of all of the predictions made by each ensemble. 

![EasyEnsembleClassifier](https://github.com/willmino/Credit_Risk_Analysis/blob/main/images/EasyEnsembleClassifier.png)

- Balanced Accuracy Score: 0.93
- Precision: 0.07
- Recall: 0.91

The EasyEnsembleClassifer exhibited an accuracy score of 0.93. This is very high and favorable. 93% of all credit loan predictions were accurate.
The Precision of the model is low because there were 979 false predictions made. Because the dataset is so large, this does not have much of an impact on the overall accuracy of the model. Yet the Precision for this model is higher than any of there other ML models because its Precision value was 0.07. Again this is about 7 times higher than any other model. The Recall of the EasyEnsembleClassifier model i 0.91. This value was very high and favorable. This means that 91% of all high_risk loans were correctly detected by the EasyEnsembleClassifier model. This model could be considered favorable.

## Summary

I ran several ML models to predict credit risks associated with given out particular loans. Its useful for a bank or credit company to know
the history of an individuals credit before they loan money to them. Sometimes the bank will only have access to limited information and might want to predict whether
loans from certain customers could be constituted as either low_risk or high_risk. These machine learning models took input feature data (independent variables)
in order to fit data to each model, and subsequently make predictions. Again, this dataset was highly imbalanced. Most of the records were for low_risk loans. Thus, to handle the imbalance of the dataset, several methods for resampling the data were made. In order to make accurate predictions with ML models, the data sets should
have similar sizes. When resampling yielded similarly sized data sets, the ML models began to make as accurate of predictions as possible.

The most inaccurate predictions were made by the initial 4 models. These models, in conjunction with LogisticRegression, were either resampled with
RandomOverSampler, SMOTE Oversampling, ClusterCentroids Undersampling, and SMOTEENN combination oversampling and undersampling. These models were considered
inaccurate and not favorable because they all had very low Precision values, each at about a value of 0.01. Also, their Recall values were generally low. 
Going from left to right with the order of ML models previosuly mentioned, their respective Recall values were 0.61, 0.59, 0.59, and 0.70.

Recall is a valuable figure for credit risk detection as we want to minimize as many false negatives as possible. If banks minimize false negatives in this context, they will avoid as many bad loans as possible.
Precision is also valuable in credit risk detection because we want to minimize the number of false positives. Doing so will allow banks to maximize their opportunities for good lending opportunities. At the same time, having low Precision (large number of false positives for high_risk loans) will cause banks to pass up on these valuable opportunities. Thus, there is a balance in credit loan risk prediction by taking in to account the overall accuracy, Precision, and Recall of the prediction model.

Even through a range of the models exhibited medium to high levels of Recall, meaning the percentage of correct high_risk predictions out of all truly high risk loans, and medium to high accuracy, most of the models exhibited dismal Precision ranging from values of 0.01 to 0.07 This means out of all the high_risk loan predictions made, only 7% of these predictions were accurate. This specific model was the EasyEnsembleClassifier. This means that 93% of the high_risk credit loan predictions were false positives. There were 979 false positive predictions made by this model. The bank would have to balance the fact that they are passing on about 979 low_risk and potentially profitable loans. Out of the 17205 predictions made by the model, the 979 false positive but acutally low_risk loans comprise 6% of all the total potential loans. If the bank is ok with losing about 6% of all of its good quality loans, with the trade off of avoiding 91% of all the high_risk loans available (Recall level of 0.91), then I would see this model as actually a viable option for the bank. Who knows how much money banks or credit companies lose each year due to bad loans, but I can see the EasyEnsembleClassifier ML model as a valuable tool for capitalizing upon as many high quality loans as possible. Thus, I would recommend the EasyEnsembleClassifer for credit loan risk predictions.

## Sources
1. This code, which can be found in both .ipynb files:

`from sklearn.model_selection import train_test_split`

`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)`

comes from this webpage:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


2. Imbalanced-Learn  BalancedRandomForestClassifier
The code for this model and the explanation for the model in the results section was obtained from info in the documentation of the model on the Imbalanced-Learn website.

`from imblearn.ensemble import BalancedRandomForestClassifier`

`brf = BalancedRandomForestClassifier(random_state=1, n_estimators = 100)`

`brf.fit(X_train, y_train)`

`y_pred = brf.predict(X_test)`

source: https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html

3. Imbalanced-learn EasyEnsembleClassifier
The code for this model and the explanation for the model in the results section was obtained from info in the documentation of the model on the Imbalanced-Learn website.


`from imblearn.ensemble import EasyEnsembleClassifier`

`eec = EasyEnsembleClassifier(n_estimators=100,random_state=1)`

`eec.fit(X_train, y_train)`

`y_pred = eec.predict(X_test)`

source: https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html


