# L03 SVM

Download the repository as a zip folder and begin an R project for this lab. The zip folder will contain instructions (repeated below) and a template to get an Rmd file started.

# Overview

The main goal of this lab is to continue practicing the application of support vector machines (SVMs).

# Datasets 

We have split the `wildfires.csv` dataset into a training dataset (`wildfires_train.csv`) and test dataset (`wildfires_test.csv`). They are contained in the **data** subdirectory along with a codebook.  

# Exercise

Please complete the following exercise. Be sure your solutions are clearly indicated and that the document is neatly formatted.

#### Exercise 1
Located in the northeast of the wilderness area is a wildlife protection zone. It is home to several rare and endangered species, and thus conservationists and park rangers are very interested in whether a given wildfire is likely to reach it. 

Our goal is to predict whether a wildfire will reach the wildlife protection zone, as determined by the indicator variable `wlf`. Previously we utilized boosting, bagging, and random forests methods to build a candidate classification trees that predict whether a wildfire will reach this zone. We also benchmarked those methods against a multiple linear logistic model that utilized all predictors for comparison. For this lab will will want to use a support vector classifier and support vector machines.

1. Use the `tune()` function to select an optimal `cost` for a support vector classifier (linear kernel). Consider values in the range 0.01 to 10. Compute the training and test error rates using this new value for cost.

2. Use the `tune()` function to select an optimal cost for a support vector machine (radial kernel). Use the default value for `gamma` and try to determine your own range of values for `cost` to explore. *Alternatively, you could use `tune()` to search for the optimal `cost` and `gamma`.*

3. Use the `tune()` function to select an optimal cost for a support vector machine (polynomial kernel). Use the default value for `degree` and try to determine your own range of values for `cost` to explore. *Alternatively, you could use `tune()` to search for the optimal `cost` and `degree`.* 

Construct a table displaying the test error for these 3 candidate classifiers, the 3 candidate classifiers from L02 CART, and the multiple logistic regression fit in L02 CART.

Calculate the test error for each of the 3 candidate classifiers selected in parts (1) - (3). Which classifier is the best? 

<br><br>

### Challenge (Not Mandatory)

Use the `ggroc()` function from the `pROC` package to construct ROC curves for the 3 SVM candidate classifiers from above. Do this for each classifier on both the training and testing sets.    
<br><br>
