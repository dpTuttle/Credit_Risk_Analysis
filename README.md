# Credit_Risk_Analysis

## Overview
The objective of the analysis is to use credit card usage dataset from LendingClub, a peer-to-peer lending services company, to predict credit risk. The data will be both over and undersampled, as well as a combinatorial approach used to fit models to generate predictions. Finally, the performance of these models will be evaluated.

### Resources

- Data sources: XXXXXX
- Tools: Jupyter Notebook

## Results

### Naive Random Oversampling

The Naive Random Oversampling method used **RandomOverSampler** to resample the data, where instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. The logistic regression model was then fitted on training dataset to evaluate testing dataset. The following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

[INSERT IMAGE URL HERE]

- The balanced accuracy score for this model is around XXX%, meaning that the model predicted the credit risk accurately XXX% of the time. This is a fairly positive score, but not great.
- The precision scores for this model are very skewed toward the low-risk loans (XXX) as almost all of the low-risk loans were correctly predicted, but very few of the high-risk loans (0.01) were accurately predicted. Therefore, this model is not good for identifying high-risk loans.
- The recall scores for this model show that the model is merginally better at identifying positive low-risk loans (XXX%) than positive high-risk loans (XXX%) and the recall scores are not great for either.
- The f1 scores for low and high risk loans are XXX and XXX resprctively, indicating a bad model for identifying high_risk loans compared to low-risk ones.

### SMOTE Oversampling

In **SMOTE**, like random oversampling, the size of the minority class is also increased, but instances of the minority class are selected based on the neighboring data instead of randomly.

The logistic regression model was then fitted to get balanced accuracy score, confusion matrix, and classification report.

[INSERT IMAGE URL HERE]

- The balanced accuracy score for this model is around XXX%, meaning that the model predicted the credit risk accurately XXX% of the time. This is also a fairly good score, but not excellent.
- The precision scores for this model are similar to Random Oversampling method, almost all of the low-risk loans(XXXX) were correctly predicted but very few of the high-risk(XXXX) ones were. This model is thus not good for identifying high-risk loans.
- The recall for both low and high risk loans are XXX% and 66% repsectively.
- The f1 scores for low and high risk loans are XXXX and XXXX respectively, indicating a bad model for identifying high_risk loans compared to low-risk ones.

### Undersampling

The Undersampling method used **Cluster Centroids** to resample the data and reduce the majority class of training data to use in a logistic regression model. The logistic regression model was fitted and the following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

[INSERT IMAGE URL HERE]

- The balanced accuracy score for the undersampling technique was around XXX%, meaning that only 53% of the testing data was accurately classified. This is the lowest accuracy score of all models.
- The precision scores for this model are very skewed toward the low-risk loans(XXX) as almost all of the low-risk loans were correctly predicted, but nearly none of the high-risk loans(0.01) were. This model is not optimal for identifying high-risk loans.
- The recall scores for the high and low risk loans were around XXX% and XXX%, respectively. These show that many of the positive cases were not accurately predicted.
- The f1 scores for low and high risk loans are XXX and XXX resprctively, indicating that the model failed to identify high_risk loans.

### Combination (Over and Under) Sampling

The Combination method(over and under sampling) used **SMOTEENN** to remove outliers and resample the data to use in a logistic regression model. The logistic regression model was fitted to get balanced accuracy score, confusion matrix, and classification report.

[INSERT IMAGE URL HERE]

- The balanced accuracy score for the SMOTEEN method is around 62%, meaning that XXX% of the testing data was accurately classified. 
- The precision scores for this model are very skewed toward the low-risk loans(XXX) compared to high-risk loans(XXX).
- The high-risk recall score for this model is higher at 69% than the the low-risk score at XXX%. Compared to the previous methods, this model is relatively better at identifying true high-risk loans.
- The f1 scores for low and high risk loans are XXX and XXX resprctively, indicating the model is not great for identifying high_risk loans.

### Balanced Random Forest Classifier

The Balanced Random Forest Classifier was used to resample the training data using the **BalancedRandomForestClassifier** algorithm with XXX estimators to classify the testing data.The following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

[INSERT IMAGE URL HERE]

- The balanced accuracy score for this model is comparatively high at XXX%, meaning that XXX% of the testing credit data was accurately classified.
- The precision for the high-risk loans(0.04) is low compared to low-risk loan(XXX), indicating a large number of false positives, which indicates an unreliable positive classification.
- The recall score for low-risk loans is very high at XXX% and for the high-risk loans is fairly high too at XXX%. This shows that the classifier is good at predicting true positives for low-risk loans.
- The high f1 score for low-risk loans is XXX indicating a good model at classifying low_risk loan.

### Easy Ensemble AdaBoost Classifier

The Balanced Random Forest Classifier was used to resample the training data using the **EasyEnsembleClassifier** algorithm with 100 estimators to classify the testing data.The following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

[INSERT IMAGE URL HERE]

- The balanced accuracy score for this model is comparatively high at XXX%, meaning that XXX% of the testing credit data was accurately classified.
- The precision for the high-risk loans(XXX) is low compared to low-risk loan(XXX), indicating a large number of false positives, which indicates an unreliable positive classification.
- The recall score for low and high risk loans are quite high at XXX% and XXX%, respectively. This shows that the classifier is good at predicting true positives for both cases.
- The high f1 score for low-risk loans is XXX indicating a good model at classifying low_risk loan compared to high-risk loans at XXXX.

## Summary

- From the results, it is evident that the **XXXXXXXXXXXXXXXXXX** is the winner among all the models fitted. It has the highest accuracy score and was able to correctly classify more high and low risk loans than the other models.
- However, none of these models were a good fit when it came to predicting high-risk loans. The best we could get was from the XXXXXXXXXXXXXXXXXXX which produced a mere XXX% of high-risk loans being correctly predicted. So this model can be used only if we're not interested in predicting high-risk loans, which is unlikely.
- This might be a problem of overfitting and/or not having enough useful features in the dataset. We could inspect which features are really correlated with the target variable and fit a model using only those. If that model too fails to predict both low and high-risk loans correctly, we will have to search for more data. 
