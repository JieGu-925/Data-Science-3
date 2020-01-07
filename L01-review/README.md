# L01 Review

Download the repository as a zip folder and begin a R project for this lab. The zip folder will contain instructions - seen below - and template to get a Rmd file started.

## Overview

The goal of this lab is to review concepts and techniques from previous quarters. 

## Datasets 

We will be utilizing `wildfires.csv` dataset contained in the **data** subdirectory.  

## Exercises

Please complete the following exercises. Be sure your solutions are clearly indicated and that the document is neatly formatted.

#### Exercise 1 
The total area burned by a wildfire is of great concern to government planners. This is captured by the variable `burned` in the `wildfires` dataset. The `starter_scipt.R` file provides two candidate models for estimating the total area burned by a wildfire (`burned`). 

Using tidyverse techniques, preform 10-fold cross validation to estimate each model's test error (e.g., test MSE, test RMSE) in order to determine which model might be superior. **Hint: Remember to set a seed.**
<br><br>

#### Exercise 2
Explain the concept of bias-variance trade-off when choosing between the validation set approach, leave-one-out cross-validation (LOOCV), or k-fold cross-validation for estimating test MSE.  
<br><br>

#### Exercise 3
A wildlife protection area is located in the park from which this data was collected, and is denoted by `wlf` in the dataset. The `starter_scipt.R` file provides two candidate models for predicting whether fires reach this wildlife area. 

Using tidyverse techniques, preform 10-fold cross validation to estimate each model's misclassification rate (e.g., test misclassification rate) in order to determine which model might be superior. **Hint: Remember to set a seed.**
<br><br>

### Challenge (Not Mandatory)

Use lasso regression to select a candidate model for estimating `burned` in  **Exercise 1**. Repeat **Exercise 1**, now with 3 candidate models.
<br><br>
