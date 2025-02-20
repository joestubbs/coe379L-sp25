Ensemble Techniques and Boosting 
================================
In this module we introduce the concept of ensemble methods in Machine Learning and describe 
in detail boosting and the Adaptive Boosting algorithm. We then discuss boosting implementations
in scikit-learn. 

By the end of this module, students should be able to:

1. Describe ensemble methods at a high level and when they can be effective. 
2. Implement boosting algorithms such as AdaBoost and Histogram-based Gradient Boosting in 
   scikit-learn. 

Introduction 
------------

Ensemble techniques in Machine Learning are methods that combine multiple, individual models or 
algorithms to produce a single model. We have already seen one example of an ensemble method: the 
Random Forrest Classifier, which combined multiple decision trees into a single estimator. 

Ensemble methods can lead to more robust, accurate and stable 
models. Some of the advantages of ensemble techniques are:

1. *Improved accuracy*: When we combine mutliple models the bias and variance can be reduced, thus 
   leading to overall accurate predictions.

2. *Reduce overfitting*: Complex models are more prone to overfitting. Bagging techniques such as 
   Random forest, which we
   studied in the previous lecture, reduces overfitting by averaging predictions of multiple models on 
   different subsets of data.

3. *Robustness*: Ensemble methods are more robust to noise in the data, as they rely on multiple models that may each perform differently but help each other in terms of generalization.

There are several broad categories of ensemble techniques, including:

1. **Bagging (Bootstrap Aggregation)**: Involves training multiple instances of a model on 
   random subsets of the data and averaging the predictions (e.g., Random Forest).

2. **Boosting**:  Sequentially builds models where each model tries to correct the errors 
   made by the previous one (e.g., AdaBoost, Gradient Boosting).

3. **Stacking** : Combines the predictions of multiple models by training a meta-model to 
   combine them in an optimal way (often with different types of base learners).


.. note:: 

    The general idea of ensemble methods is that the group of weak learners (whose predictions might not be the most accurate) can
    outperform a single strong learner by combining their strengths and minimizing the weakness.


Bias and Variance
-------------------

In machine learning, bias and variance are two fundamental sources of error that affects the performance of a model.
Both the errors are seen when the model tries to generalize to predict on the unseen data. The goal is minimize
both bias and variance, to achieve a model that generalizes well.

A model with high *bias* makes strong assumptions about the data and tries to oversimplify the underlying patterns.
This usually leads to **underfitting** as the model fails to capture the underlying patterns in data.

On the other hand, *variance* refers to errors introduced by the model's sensitivity to small 
fluctuations in the training data.
A model with high variance is highly flexible and can fit the training data very well, but it often captures 
noise or random fluctuations in the data rather than the true underlying patterns. This leads to overfitting, 
where the model performs well on the training data but poorly on unseen test data.

Bias-Variance Tradeoff. The relationship between bias and variance is inverse: reducing bias often 
increases variance, and reducing variance often increases bias. 
This is called the bias-variance tradeoff. The key is to find the right balance between 
bias and variance that allows the model to generalize well to new, unseen data. 

Bagging techniques reduces variance, where as boosting reduces both bias.


When to Use Ensemble Techniques:
-----------------------------------

**High Bias (Underfitting)**: Boosting can help by iteratively correcting errors, reducing bias.

**High Variance (Overfitting)**: Bagging (e.g., Random Forest) helps by averaging multiple models trained on different subsets of data, reducing variance.

**Outliers or Noisy Data**: Ensemble methods are often robust to outliers or noisy data because they aggregate the predictions of multiple models.


.. note:: 
    
    Ensemble Methods can be used for various reasons, mainly to:

    Decrease Variance (Bagging)
    Decrease Bias (Boosting)
    Improve Predictions (Stacking)


Boosting techniques
--------------------

The general idea of boosting is to build multiple models sequentially, where each model tries to correct the errors 
made by previous models. The models are trained sequentially and each subesequent model gives more weights to 
the data points that were previously missclassified by the previous model.
The final prediction is weighted combination of all individual models.

Some of the common boosting techniques are:

**AdaBoost (Adaptive Boosting)**: In the paper, "Experiments with a New Boosting Algorithm" by Yoav Freund and Robert Schapire (1996) a novel
boosting algortihm was proposed, known as  the AdaBoost (Adaptive Boosting) which improved the accuracy of weak classifiers.
This algorithm was shown to be effective in combining multiple weak learners to create a stronger, more accurate classifier.
A weak learner is defined as a classifier that performs slightly better than random guessing on a given task.

The central idea of boosting is to improve the performance of weak learners by re-weighting the training data
and giving more emphasis to the data points that are misclassified in each iteration.

AdaBoost trains a series of weak learners in sequence, with each learner focusing on the instances that were misclassified by the previous ones.
After each classifier is trained, AdaBoost adjusts the weights of the training instances. Misclassified instances get higher weights, so the next weak learner focuses more on these "hard" examples.
After each weak learner is trained, the final prediction is made by a weighted combination of all the weak learners' predictions, with more accurate learners receiving higher weights.


.. code-block:: python3 

    # Import necessary libraries
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
