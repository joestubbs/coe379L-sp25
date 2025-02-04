Project 01 - 20 Points
======================

**Date Assigned:** Tuesday, Feb.4, 2025

**Due Date:** Tuesday, Feb.25, 2025, 5 pm CST. 

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
For this project, you will use a dataset about breast cancer available from the class git repository.
It can be downloaded here: `Project 1 Dataset <https://raw.githubusercontent.com/joestubbs/coe379L-sp25/master/datasets/unit01/project1.csv>`_

This data set has 9 attributes/features and 1 dependent/target variable to determine 
recurrence of breast cancer in patients. The 9 independent features are: ``age``, ``menopause``,
``tumor-size``, ``inv-nodes``, ``node-caps``, ``deg-malig``, ``breast``, ``breast-quad``, and 
``irradiat``. The dependent variable to be predicted is ``class``. 

**Part 1 (6 points):** Your objective is to perform Exploratory data analysis on the dataset.
Complete the following:

* Identify shape, size of the raw data (1 point)
* Get information about datatypes. Comment if any of the variables need datatype conversion. Check for duplicate rows and treat them. (1 point)
* Identify missing data and/or invalid values and treat them with suitable mean, median, mode or other method  (1 point)
* Visualize the dataset through different univariate analysis and comment on your observations (2)
* Perform one-hot encoding on categorical variables (1 point)

**Part 2 (9 points):** Fit Regression models on the data to predict the fuel efficiency of cars:

* Split the data into training and test datasets. Make sure your split is reproducible and 
  that it maintains roughly the proportion of each class of dependent variable. (1 point)
* Perform classification using  (6 points) 
    * K-Nearest Neighbor Classifier 
    * K-Nearest Neighbor Classifier using Grid search CV
    * Linear classification
* Print report showing accuracy, recall, precision and f1-score for each classification model. Which 
  metric is most important for this problem? (You will explain your answer in the report in Part 3). ( 2 points)

**Part 3 (5 points):** Submit a 2 page report with the following: 

* What did you do to prepare the data?
* What insights did you get from your data preparation?
* What procedure did you use to train the model? 
* How does the model perform to predict the fuel efficiency?
* How confident are you in the model?

**Submission Guidelines:**
Part 1 and Part 2 should be submitted as one notebook file. Part 3 should be submitted as a PDF file. 
Both the files should be committed to a personal GitHub repo. 

To submit your project, send an email with the following information:

.. code-block:: bash 

    Subject: COE 379L Project 1 Submission
    To: jstubbs@tacc.utexas.edu, ajamthe@tacc.utexas.edu, hainguyen@utexas.edu

    Body: Please include the following: 
      1) GitHub Repo Link 
      2) Any other details needed to access the repository (e.g., file locations)
    
Please make sure the repository is either public or shared with the following GitHub accounts: 

* Joe Stubbs, GitHub account: ``joestubbs`` 
* Anagha Jamthe, GitHub account: ``ajamthetacc``
* Van Hai Nguyen, GitHub account: ``nguyenvanhaibk92``

Projects will be considered late if an email is not received by the due date. 
We will reply with an acknowledgement that we received and were able to pull the GitHub repo.
I recommend that everyone create the git repository, either share it with us more make it public, 
and then send us the email above ASAP. 


**Evaluation:**
We will git pull all repos on the due date at or after 5 pm. This is the version of your submission 
that we will evaluate unless we receive a message that you would like an extension (with a 1 point 
per day penalty). 