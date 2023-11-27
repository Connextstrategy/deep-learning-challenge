# Python Rutgers Bootcamp Challenge - Venture Capital Funding 

This activity is broken down into Python code which acts as a tool that can help it select the applicants for funding with the best chance of success in their ventures

## Description

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

![Screenshot 2023-11-27 124825](https://github.com/Connextstrategy/deep-learning-challenge/assets/18508699/94cc902d-c875-43df-8a6f-ab938a7607ee)

## Instructions

The instructions for this Challenge are divided into the following subsections:

1. Split the Data into Training and Testing Sets

2. Create a Logistic Regression Model with the Original Data

3. Write a Credit Risk Analysis Report

## Split the Data into Training and Testing Sets

Open the starter code notebook and use it to complete the following steps:

1. Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

2. Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

![Screenshot 2023-11-20 185139](https://github.com/Connextstrategy/credit-risk-classification/assets/18508699/9d3c40eb-fafc-42cb-ae16-e843327365a1)

3. Split the data into training and testing datasets by using train_test_split.

## Create a Logistic Regression Model with the Original Data

Use your knowledge of logistic regression to complete the following steps:

1. Fit a logistic regression model by using the training data (X_train and y_train).

2. Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.

   * Evaluate the model’s performance by doing the following:

   * Generate a confusion matrix.

3. Print the classification report.

4. Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?
 
 ## Write a Credit Risk Analysis Report

Write a brief report that includes a summary and analysis of the performance of the machine learning models that you used in this homework. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, ensuring that it contains the following:

1. An overview of the analysis: Explain the purpose of this analysis.

2. The results: Using a bulleted list, describe the accuracy score, the precision score, and recall score of the machine learning model.

3. A summary: Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning.

 ## Final & Complete Written Report 

### Module 12 Report Template

### Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

The purpose of this analysis was a ascertain what supervised machine learning model would be best to predict loan classification for credit card users. The financial information is based on the CSV file combined with a splitting of data (around whether it's a healthy (0) versus high risk (1) loan. It was tested after that. Variables outside of the type of loan include loan interest rate, borrower income, debt to income ratio, number of acconts, and total debt. 

The model attempted see the difference between the healthy versus high risk loans. Two separate types of supervised machine learning models were deployed to include: 

* Logistical regression: Used when the outcome is binary; Simple, scalable, and has clear metrics
* Logistical regression with Random Sampler: Used to randomly select a subset of data from a larger dataset; Helps prevent over fitting

### Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

  * Machine Learning Model 1: Logistical Regression  

The original logistical regression model data tested with 99.184% accuracy with the following confusion matrix below: 

![Screenshot 2023-11-20 210555](https://github.com/Connextstrategy/credit-risk-classification/assets/18508699/e8abbca7-5768-48fb-8eb1-f5c96406d31c)

This confusion matrix has a slightly higher precision due to fewer false positives, meaning it is more accurate when it predicts a positive instance.

Testing report shows a high level of accuracy and an excellent balance between precision and recall for both classes, particularly excelling in class 0 predictions. The decision to deploy or further refine this model would depend on the specific requirements of the task, especially considering the slightly lower (but still strong) performance on the minority class (class 1).

![Screenshot 2023-11-20 211529](https://github.com/Connextstrategy/credit-risk-classification/assets/18508699/351b9a7e-c8bf-4153-b872-5b69ce9e02fd)


  * Machine Learning Model 2: Logistical Regression with Random Sampler
  
The random sampler logistical regression model data tested with 99.38% accuracy with the following confusion matrix below: 

![Screenshot 2023-11-20 205643](https://github.com/Connextstrategy/credit-risk-classification/assets/18508699/2fa4740b-9138-41d8-bc14-8e0d03203fd7)

This confusion matrix has a higher recall since it has fewer false negatives. This is crucial in scenarios where missing a positive instance can be critical. 

Testing report shows that the model is almost perfect at predicting class 0, with exceptional precision, recall, and F1-score. The model also has very high recall but slightly lower precision. This indicates that the model is very good at identifying class 1 instances, but when it predicts something as class 1, there's a 16% chance it might be wrong.

![Screenshot 2023-11-20 212214](https://github.com/Connextstrategy/credit-risk-classification/assets/18508699/e73837c7-9e6f-43c4-8c48-6ffef78b1e23)


### Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

The random sampler logistical regression model performed better, but the issues with the prediction of the high risk is an issue when it comes to choosing this model. That being said, if you had to choose which mdoel to use, the choice should be the model with random sampler and more balance between classes. If slightly higher precision in the positive class is more important (like in medical diagnosis), then the original logistical model might be the better choice.
 

## Dependencies

* Using Jupyter notebooks for the analysis of the models. 

## Installing

* No modifications needed to be made to files/folders

## Help

No help required. 
