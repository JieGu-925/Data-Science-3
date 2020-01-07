# L02 CART

Download the repository as a zip folder and begin an R project for this lab. The zip folder will contain instructions (repeated below) and a template to get an Rmd file started.

## Overview

The main goal of this lab is to continue practicing the application of tree-based methods (i.e., classification and regression trees).

## Datasets

We have split the `wildfires.csv` dataset into a training dataset (`wildfires_train.csv`) and test dataset (`wildfires_test.csv`). They are contained in the **data** subdirectory along with a codebook.  

## Exercises

Please complete the following exercises. The document should be neatly formatted. 

#### Exercise 1 
The total area burned by a wildfire is of great concern to government planners. This is captured by the variable `burned` in the `wildfires` dataset, which is a continuous variable. In this exercise, you will train models to predict `burned` using other variables in the data (**exclude `wlf` as a predictor** ). Train the following candidate models:

* boosting
* bagging
* random forests 
* linear regression
* ridge regression 

Compare the estimated test errors for each model to determine which is best. 

<br><br>

#### Exercise 2
Located in the northeast of the wilderness area is a wildlife protection zone. It is home to several rare and endangered species, and thus conservationists and park rangers are very interested in whether a given wildfire is likely to reach it. In the data, fires that reach the wildlife protection zone are denoted by the indicator variable `wlf`. 
In this exercise, you will train models to predict `wlf` using other variables in the data (**there is no exclusion on which varibles to use as predictors**). Train the following candidate models:

* boosting
* bagging 
* random forests
* logistic regression
* ridge logistic regression
     
Compare the estimated test errors for each model to determine which is best. 
