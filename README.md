# Stroke-Prediction

Built various ML Models such as DecisionTree Classifier and XGBoost Classifier with Undersampling, Oversampling, and SMOTE sampling technique to predict stroke in patient based on gender, age, smoking habit, and other existing conditions.

## Introduction
As per the Centre for Disease Control and Prevention website a stroke, sometimes also known as brain attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts. Which can result into permanent or lasting damage into the brain, sometimes it also causes long-term disability, or even death. According to the World Health Organization stroke is 2nd leading cause of death globally, responsible for around 11% of total deaths.

Dataset: [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

In this project we are going to analyze and predict weather a patient is likely to get a stroke or not based on following input parameters: age, gender, hypertension, heart disease, ever married, work type, residence type, average glucose level, bmi, smoking status. The goal is to identify potential risk factors and provide a risk assessment for the individual, which can be used by healthcare professionals to take preventative measures and reduce the risk of stroke. It is important to note that the prediction should not be considered as a definite diagnosis and that a comprehensive medical evaluation by a healthcare professional is necessary to determine the presence of stroke or any other medical conditions.

## Data Description
![alt text](https://github.com/kpola009/Stroke-Prediction/blob/main/Images/df.png "Figure 1")

Number of records: 5110

Number of features: 12

Categorical features:['gender', 'ever_married', 'work-type', 'Residence_type', 'smoking_status']

## Data Preparation
- After analyzing the dataset, it was found that 201 records for the 'bmi' feature had null values. To handle these null values, the mean of all 'bmi' values was calculated where the 'stroke' feature had a value of 0. This mean value was then used to replace the null values in the corresponding records for the 'bmi' feature and 'stroke' value of 0. The same approach was used for replacing the null values of the 'bmi' feature where the 'stroke' value was 1.
- Additionally, it was discovered that the 'gender' feature had only one record with a value of 'other.' This record was removed as it acted as an outlier in the dataset and would not have any significant impact on further analysis.
- Furthermore, the dataset contained 5 categorical features which were transformed into numerical values using the Label Encoding technique.
- Additionally, it was observed that the target feature 'stroke' was highly imbalanced. To address this issue, various sampling techniques such as SMOTE, undersampling, and oversampling were applied with predefined weights for the machine learning algorithm.

## EDA
In the Exploratory Data Analysis (EDA), the relationships between the features were analyzed. The correlation between the non-categorical features was examined using a correlation matrix, while the relationships between the categorical features were analyzed using a chi-square test.

![alt text](https://github.com/kpola009/Stroke-Prediction/blob/main/Images/corr.png "Figure 2")

From the above graph, we can observe that all the continuous features (excluding 'bmi') appear to have some degree of correlation with the 'stroke' outcome. However, 'bmi' does not seem to have a linear correlation with 'stroke'.

![alt text](https://github.com/kpola009/Stroke-Prediction/blob/main/Images/catcorr.png "Figure 2")

The results of the chi-square test indicate that there is a relationship between two variables if the p-value is less than or equal to 0.5. In this case, all features (excluding residence type) have a p-value less than or equal to 0.5, indicating a relationship between the feature and 'stroke'. This includes a relationship between 'gender' and 'stroke', 'ever_married' and 'stroke', 'work_type' and 'stroke', and 'smoking status' and 'stroke'.

## Machine Learning Algorithm Selection
For this project, the Decision Tree Classifier and XGBoost Classifier models were selected. The reason for choosing these models is that they are both tree-based classifiers, which have been known to perform well on imbalanced datasets.

## Results
To better understand results first I will define precision and recall

- Precision: Truly predicting class 'stroke (1)'/'non-stroke (0)' upon all the class 'stroke (1)'/'non-stroke (0)' preciditions.
- Recall: Correctly classifying class 'stroke (1)'/'non-stroke (0)' in case of class 'stroke (1)'/'non-stroke (0)'
From above analysis and with model building using Decision tree classifier and XGBClassifier, with sampling methods such as SMOTE, Oversampling and Undersampling resulted in following metrics.

- DT with SMOTE: Model overfitted with good training accuracy but moderate test accuracy. Precision and recall for class stroke - poor
- DT with oversampling: Good training and test accuracy but precision and recall for class 'stoke' - poor
- DT with undersampling: Model overfitted with poor recall for class 'non-stroke' and poor precision for class 'stroke'
- XGB without sampling: Perfect Model fitting and accuracy but poor recall for class 'stroke'
- XGB with SMOTE: Good Model fitting but poor recall and precision for class 'stroke'
- XGB with oversampling: Poor recall and precision for class 'stroke' with model overfitting
- XGB with undersampling: Moderate Model fitting and accuracy but poor precision for class 'stroke'

## Algorithm Selection Reasoning
After evaluating the performance of all the models, it was determined that the best models were XGBoost without sampling and XGBoost with undersampling. The other models were discarded because they either overfitted the data or had poor precision and recall scores. The choice between these two models illustrates the trade-off between precision and recall. Let's examine this in more detail.

- For model XGB without sampling we have precision of 1.00 and recall of 0.11 for class 'stoke' meaning when our model predicts 'stroke' is correct 100% whereas it correctly identifies 11% of all cases being 'stroke' when it is 'stroke'.
- For model XGB with undersampling we have precision of 0.15 and recall of 0.81 for class 'stroke' meaning when our model predicts 'stroke' is correct 15% of the times whereas it correctly identiifies 81% of cases being 'stroke' when it is 'stroke'

For this project, the XGBoost model with undersampling is considered to be the best model. This is because incorrect labeling of a 'non-stroke' case does not have a significant impact on the patient, compared to incorrect labeling of a 'stroke' case as 'non-stroke.' Additionally, the XGBoost model with undersampling correctly identifies 81% of cases as 'stroke' when it is actually 'stroke.'



