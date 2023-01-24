# Credit_Risk_Analysis

## Overview
The objective of the analysis is to use credit card usage dataset from LendingClub, a peer-to-peer lending services company, to predict credit risk. The data will be both over and undersampled, as well as a combinatorial approach used to fit models to generate predictions. Finally, the performance of these models will be evaluated.

### Resources

- Data sources: LoanStats2019Q1.csv
- Tools: Jupyter Notebook

## Results

### Naive Random Oversampling

The Naive Random Oversampling method used **RandomOverSampler** to resample the data, where instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. The logistic regression model was then fitted on training dataset to evaluate testing dataset. The following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

![Oversampling](https://github.com/dpTuttle/Credit_Risk_Analysis/blob/main/Resources/Oversampling.png)

- The balanced accuracy score for this model is around 64%, meaning that the model predicted the credit risk accurately 64% of the time. This is a fairly positive score, but not great.
- The precision scores for this model are very skewed toward the low-risk loans (1.00) as all of the low-risk loans were correctly predicted, but very few of the high-risk loans (0.01) were accurately predicted. **Therefore, this model is not good for identifying high-risk loans.**
- The recall scores for this model show that the model is merginally better at identifying positive high-risk loans (70%%) than positive low-risk loans (59%) and the recall scores are not great for either.
- The f1 scores for low and high risk loans are XXX and XXX resprctively, indicating a bad model for identifying high_risk loans compared to low-risk ones.

### SMOTE Oversampling

In **SMOTE**, like random oversampling, the size of the minority class is also increased, but instances of the minority class are selected based on the neighboring data instead of randomly.

The logistic regression model was then fitted to get balanced accuracy score, confusion matrix, and classification report.

![SMOTE](https://github.com/dpTuttle/Credit_Risk_Analysis/blob/main/Resources/SMOTE.png)

- The balanced accuracy score for this model is around 66%, meaning that the model predicted the credit risk accurately 66% of the time. This is also a fairly good score, but not excellent.
- The precision scores for this model are similar to Random Oversampling method, all of the low-risk loans(1.00) were correctly predicted but very few of the high-risk(.01) ones were. **This model is thus not good for identifying high-risk loans.**
- The recall for both low and high risk loans are 69% and 63% repsectively.
- The f1 scores for low and high risk loans are 82% and 2% respectively, indicating a bad model for identifying high_risk loans compared to low-risk ones.

### Undersampling

The Undersampling method used **Cluster Centroids** to resample the data and reduce the majority class of training data to use in a logistic regression model. The logistic regression model was fitted and the following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

![Clustered_Centroids](https://github.com/dpTuttle/Credit_Risk_Analysis/blob/main/Resources/Clutered_Centroid.png)

- The balanced accuracy score for the undersampling technique was around 66%, meaning that only 66% of the testing data was accurately classified. This is in line with accuracy score of the previous models.
- The precision scores for this model are very skewed toward the low-risk loans(1.00) as all of the low-risk loans were correctly predicted, but nearly none of the high-risk loans(0.01) were. **This model is not optimal for identifying high-risk loans.**
- The recall scores for the high and low risk loans were around 69% and 40%, respectively. These show that many of the positive cases were not accurately predicted.
- The f1 scores for low and high risk loans are 57% and 1% resprctively, indicating that the model failed to identify high_risk loans.

### Combination (Over and Under) Sampling

The Combination method(over and under sampling) used **SMOTEENN** to remove outliers and resample the data to use in a logistic regression model. The logistic regression model was fitted to get balanced accuracy score, confusion matrix, and classification report.

![SMOTEEN](https://github.com/dpTuttle/Credit_Risk_Analysis/blob/main/Resources/SMOTEEN.png)

- The balanced accuracy score for the SMOTEEN method is around 54%, meaning that 54% of the testing data was accurately classified. 
- The precision scores for this model are very skewed toward the low-risk loans(1.00) compared to high-risk loans(.01).
- The high-risk recall score for this model is higher at 73% than the the low-risk score at 57%. Compared to the previous methods, this model is relatively better at identifying true high-risk loans.
- The f1 scores for low and high risk loans are .73 and .02 resprctively, indicating **the model is not great for identifying high_risk loans.**

### Balanced Random Forest Classifier

The Balanced Random Forest Classifier was used to resample the training data using the **BalancedRandomForestClassifier** algorithm with XXX estimators to classify the testing data.The following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

![Random_Forrest](https://github.com/dpTuttle/Credit_Risk_Analysis/blob/main/Resources/RandomForest.png)

- The balanced accuracy score for this model is comparatively high at 68%, meaning that 68% of the testing credit data was accurately classified.
- The precision for the high-risk loans(0.93) is high compared to low-risk loan(1.00), indicating a large number of false positives, which indicates an unreliable positive classification.
- The recall score for low-risk loans is very high at 37% and for the high-risk loans is high too at 100%. This shows that the classifier is good at predicting true positives for low-risk loans.
- The high f1 score for low-risk loans is 1.00 indicating a good model at classifying low_risk loan.

### Easy Ensemble AdaBoost Classifier

The Balanced Random Forest Classifier was used to resample the training data using the **EasyEnsembleClassifier** algorithm with 100 estimators to classify the testing data.The following image and information refers to the resulting balanced accuracy score, confusion matrix, and classification report.

![Easy_Ensemble](https://github.com/dpTuttle/Credit_Risk_Analysis/blob/main/Resources/Easy_Ensemble.png)

- The balanced accuracy score for this model is comparatively high at 68%, meaning that 68% of the testing credit data was accurately classified.
- The precision for the high-risk loans(93%) is low compared to low-risk loan(1.00%), indicating a large number of false positives, which indicates an unreliable positive classification.
- The recall score for low and high risk loans are quite high at 37% and 100%, respectively. This shows that the classifier is ok at predicting true positives for both cases.
- The high f1 score for low-risk loans is 1.00 indicating a good model at classifying low_risk loan compared to high-risk loans at .52.

## Summary

- From the results, it is evident that the **Easy Ensemble AdaBoost Classifier** or the **Random Forrest** model are the winners among all the models fitted.They had the highest accuracy scores and were able to correctly classify more high and low risk loans than the other models.
- However, none of these models were a good fit when it came to predicting high-risk loans. The best we could get was from the **SMOTEEN** model which produced a mere 57% of high-risk loans being correctly predicted. So this model can be used only if we're not interested in predicting high-risk loans, which is unlikely.
- This might be a problem of overfitting and/or not having enough useful features in the dataset. We could inspect which features are really correlated with the target variable and fit a model using only those. If that model too fails to predict both low and high-risk loans correctly, we will have to search for more data. 
