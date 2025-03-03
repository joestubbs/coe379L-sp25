Project 02 - 20 Points
======================

**Date Assigned:** Thursday, Feb.27, 2025

**Due Date:** Thursday, March 13, 2025, 5 pm CST.

**Individual Assignment:** Every student should work independently and submit their own project.
You are allowed to talk to other students about the project, but please do not copy any code 
for the notebook or text for the report.

If you use ChatGPT, please state exactly how you used it. For example, state which parts of the 
code it helped you generate, which errors it helped you debug, etc. Please do not use ChatGPT to 
generate the report for part 3. 

**Late Policy:**  Late projects will be accepted at a penalty of 1 point per day late, 
up to five days late. After the fifth late date, we will no longer be able to accept 
late submissions. In extreme cases (e.g., severe illness, death in the family, etc.) special 
accommodations can be made. Please notify us as soon as possible if you have such a situation. 

**Project Description:**

You will be using a dataset for predicting house prices that are above the median in California. 
This data set has 8 attributes/features and 1 dependent/target variable to 
determine houses that are priced above the median value. The 8 independent features are as follows:

MedInc: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
HouseAge: Age of a house within a block; a lower number is a newer building
AveRooms: Average number of rooms within a block
AveBedrms: Average number of bedrooms within a block
Population: Total number of people residing within a block
AveOccup:  Average number of people residing within a block
Latitude: A measure of how far north a house is; a higher value is farther north
Longitude: A measure of how far west a house is; a higher value is farther west

One dependent variable that is to be predicted is:
price_above_median: Median house value for households within a block (measured in US Dollars)

It can be downloaded `here <https://raw.githubusercontent.com/joestubbs/coe379L-sp25/refs/heads/master/datasets/unit02/california_housing.csv>`_.

**Part 1 : (5 points)** Exploratory Data Analysis

* Identify shape and size of the data (1 point)
* Get information about datatypes. Comment if any of the variables need datatype conversion. Check for duplicate rows and treat them if required. (1 point)
* Get the statistical information (mean, median, etc.) for all variables and derive meaniful insights from it. Comment if you see any anamolies in the data. (1 point)
* Visualize the dataset through different univariate analysis and comment on your observations. (2)


**Part 2 : (10 points)** Classification techniques

* Split the data into training and test datasets. Make sure your split is reproducible and 
  that it maintains roughly the proportion of each class of dependent variable. (1 points) 
* Perform classification using below supervised learning techniques. When appropriate, use 
  a hyperparameter space search to find optimal hyperparameter setting. 
  Consider using other techniques from class lectures, such as data standardization. 
  At a minimum, you should try the following model algorithms. We will base the grading on the 
  quality of your model(s) you develop. (5 points) 
    * K-nearnest neighbor
    * Decision Tree Classifier
    * Random Forest Classifier
    * AdaBoost Classifier
* Print report showing accuracy, recall, precision and f1-score for each classification model on all 
  data (training, testing, etc.). Which 
  metric is most important for this problem? (You will explain your answer in the report in Part 3). ( 2 points)
* Print confusion matrix for each model. (2 points)


**Part 3: (5 Points)**  Submit a 2-3 page report summarizing your findings. Be sure to include the following: 

* Which techniques did you use to train the models?  (1 point)
* Explain any techniques used to optimize model performance? (1 point)
* Compare the performance of all models to predict the dependent variable? (1 point)
* Which model would you recommend to be used for this dataset (1 point)
* For this dataset, which metric is more important, why? (1 point)

