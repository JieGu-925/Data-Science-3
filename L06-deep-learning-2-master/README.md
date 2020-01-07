# L06 Deep Learning II

Download the repository as a zip folder and begin an R project for this lab. The zip folder will contain instructions (repeated below) and a template to get an Rmd file started.

# Overview

The main goal of this lab is to continue practicing the application of deep learning techniques using neural networks as implemented through the use of the `keras` and `tensorflow` packages in R. 

When creating the Rmd file for this assignment, you won't want all your code chunks to evaluate because it would take too long to run. You'll want to produce results, store them in a subdirectory, and then reference them in the Rmd file. You may need to use hidden chunks of code to read in results or use Rmd code to insert graphics that you want to display.

# Datasets 

We have split the `wildfires.csv` dataset into a training dataset (`wildfires_train.csv`) and test dataset (`wildfires_test.csv`). They are contained in the **data** subdirectory along with a codebook.  

# Exercises

Please complete the following exercises. Be sure your solutions are clearly indicated and that the document is neatly formatted.

<br>

#### Exercise 1
The total area burned by a wildfire is of great concern to government planners. This is captured by the variable `burned` in the `wildfires` dataset. Fit a small neural network, say 3 layers (2 hidden), to predict the total area burned by a wildfire. Consider exploring the use of a smaller or larger network. **Exclude `wlf` as a predictor.**

We previously applied boosting, bagging, and random forests methods to build candidate regression trees to predict the total area burned by a wildfire (`burned`). We also fit a multiple linear regression model that used all appropriate predictors for comparison.

Construct a table displaying the RMSE for the your fitted network(s) and the 3 tree-based models and the multiple regression model fit in L02 CART. Which model is the best?

<br>

#### Exercise 2
Our goal is to predict whether a wildfire will reach the wildlife protection zone, as determined by the indicator variable `wlf`. Fit a small neural network, say 3 layers (2 hidden), to predict whether or not a wildfire with reach the wildfire protection zone. Consider exploring the use of a smaller or larger network.

Previously we utilized boosting, bagging, and random forests methods to build a candidate classification trees that predict whether a wildfire will reach this zone. We also benchmarked those methods against a multiple linear logistic model that utilized all predictors for comparison --- L02 CART. We also previously fit a support vector classifier and 2 support vector machines in L03 SVM.

Construct a table displaying the test error for the your fitted network(s), the 3 tree-based models and the multiple regression model fit in L02 CART, and the 3 models fit in L03 SVM. Which model is the best?
