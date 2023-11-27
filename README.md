# Python Rutgers Bootcamp Challenge - Venture Capital Funding 

This activity is broken down into Python code which acts as a tool that can help it select the applicants for funding with the best chance of success in their ventures

## Description

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

![Screenshot 2023-11-27 124825](https://github.com/Connextstrategy/deep-learning-challenge/assets/18508699/94cc902d-c875-43df-8a6f-ab938a7607ee)

## Step 1: Preprocess the Data

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
  * What variable(s) are the target(s) for your model?
  * What variable(s) are the feature(s) for your model?

2. Drop the EIN and NAME columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

6. Use pd.get_dummies() to encode categorical variables.

7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

## Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

## Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

  * Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
  * Add more neurons to a hidden layer.
  * Add more hidden layers.
  * Use different activation functions for the hidden layers.
    
  Add or reduce the number of epochs to the training regimen.
  
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.
 
 ## Step 4: Write a Report on the Neural Network Model

For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. Overview of the analysis: Explain the purpose of this analysis.

2. Results: Using bulleted lists and images to support your answers, address the following questions:

  * Data Preprocessing
    * What variable(s) are the target(s) for your model?
    * What variable(s) are the features for your model?
    * What variable(s) should be removed from the input data because they are neither targets nor features?

  * Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take in your attempts to increase model performance?
   
3. Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

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
