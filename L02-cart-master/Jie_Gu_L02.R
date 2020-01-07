
# Loading package(s)
library(modelr)
library(tidyverse)
library(randomForest)
library(xgboost)
library(gbm)
library(glmnet) 
library(glmnetUtils)


# Read in data
wildfire_train <- read_csv("data/wildfires_train.csv") %>%
  mutate(winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
         traffic = factor(traffic, levels = c("lo", "med", "hi")))
wildfire_test <- read_csv("data/wildfires_test.csv") %>%
  mutate(winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
         traffic = factor(traffic, levels = c("lo", "med", "hi")))

set.seed(1)
wild_validation <- wildfire_train %>%
  crossv_kfold(10, id = "fold")


#### Exercise 1
##### Bagging & Random Forest

# create a tibble of mtry values
model_def <- tibble(mtry = 1:(ncol(wildfire_train) - 2))
wild_rf <- wild_validation %>% 
  crossing(model_def)
# returns a random forest where burned is the outcome
fitRF <- function(data, mtry){
  data = as_tibble(data)
  return(randomForest(burned ~ . - wlf, data = data, mtry = mtry, importance = TRUE))
}
wild_rf <- wild_rf %>%
  mutate(model_fit = map2(train, mtry, fitRF),
         test_mse = map2_dbl(model_fit, test, mse))
# the best mtry for random forest
wild_rf_err <- wild_rf %>%
  group_by(mtry) %>%
  summarise(mean_mse = mean(test_mse)) %>% 
  arrange(mean_mse)
rf_tune <- wild_rf_err %>%
  pluck("mtry", 1)

##### Boosting

# return a boosting model where burned is the outcome
fitBoosting <- function(dat){
  dat = as_tibble(dat)
  mat = wildfire_train %>% dplyr::select(-c(burned, wlf)) %>% # encode on full df
    onehot::onehot() %>% # use onehot to encode variables
    predict(dat) # get OHE matrix
  return(xgb.DMatrix(data = mat, label = dat$burned))
}
# helper function to fit model
fit_wild <- function(train_data, learning_rate, depth, nrounds, silent = 1){
  return(
    xgb.train(params = list(eta = learning_rate, max_depth = depth, silent = silent),  
              train_data,  nrounds = nrounds) )
}
# function to get mse
xg_mse <- function(model, test){
  preds = predict(model, test)
  vals = getinfo(test, "label")
  return(mean((preds - vals)^2))
}

wild_boosting_data <- wild_validation %>%
  mutate(train_dg = map(train, fitBoosting), 
         test_dg = map(test, fitBoosting)) %>%
  dplyr::select(train_dg, test_dg)

wild_boosting <- wild_boosting_data %>%
  crossing(tibble(learning_rate = c(0.001, 0.01, 0.1, 0.2)))

wild_boosting <- wild_boosting %>%
  mutate(model_fit = map2(train_dg, learning_rate, fit_wild, depth = 10, nrounds = 5000), 
         test_mse = map2_dbl(model_fit, test_dg, xg_mse))

# 0.01 is the best learning rate for boosting
wild_boosting_err <- wild_boosting %>%
  group_by(learning_rate) %>%
  summarise(mean_mse = mean(test_mse)) %>%
  arrange(mean_mse)
boosting_tune <- wild_boosting_err %>%
  pluck("learning_rate", 1)

##### Ridge

# lambda grid to search -- use for ridge regression
lambda_grid <- 10^seq(-2, 10, length = 200)
# ridge regression
ridge_cv <- wildfire_train %>% 
  cv.glmnet(formula = burned ~ . - wlf, 
            data = ., alpha = 0, nfolds = 10, lambda = lambda_grid)

# ridge's best lambdas
ridge_lambda_min <- ridge_cv$lambda.min

##### Linear Regression

wild_lr <- function(data){
  lm(burned ~ . - wlf, data = data)
}

##### Compare mse to find the best model

wildfire <- tibble(wild_train = wildfire_train %>% list(),
                   wild_test = wildfire_test %>% list())
# mse for random forests, bagging and linear regression
mse_part1 <- wildfire %>%
  mutate(RandomForest = map2(wild_train, rf_tune, fitRF),
         Bagging = map2(wild_train, 15, fitRF),
         Linear = map(wild_train, wild_lr)) %>%
  gather(key = "model", value = "fit", RandomForest, Bagging, Linear) %>%
  mutate(test_mse = map2_dbl(fit, wild_test, mse))%>%
  select(model, test_mse)
# mse for boosting
wild <- wildfire_train %>%
  bind_rows(wildfire_test)

fitBoosting2 <- function(dat){
  dat = as_tibble(dat)
  mat = wild %>% dplyr::select(-c(burned, wlf)) %>% # encode on full df
    onehot::onehot() %>% # use onehot to encode variables
    predict(dat) # get OHE matrix
  return(xgb.DMatrix(data = mat, label = dat$burned))
}

mse_boosting <- wildfire %>%
  mutate(train_dg = map(wild_train, fitBoosting2),
         test_dg = map(wild_test, fitBoosting2)) %>%
  mutate(Boosting = map2(train_dg, boosting_tune, fit_wild, depth = 10, nrounds = 5000),
         test_mse = map2_dbl(Boosting, test_dg, xg_mse)) %>%
  gather(key = "model", value = "fit", Boosting) %>%
  select(model, test_mse)
# mse for ridge
mse_ridge <- wildfire%>%
  mutate(Ridge = map(wild_train, ~glmnet(burned ~ .- wlf, data = .x, alpha = 0, lambda = ridge_lambda_min))) %>%
  mutate(pred = map2(Ridge, wild_test, predict),
         test_mse = map2_dbl(wild_test, pred, ~ mean((.x$burned - .y)^2))) %>%
  gather(key = "model", value = "fit", Ridge) %>%
  select(model, test_mse)

mse_part1 %>%
  bind_rows(mse_boosting) %>%
  bind_rows(mse_ridge) %>%
  arrange(test_mse)


  
  #### Exercise 2
 
# function to calculate mean test error
test_err <- function(mod_fit, df){
  df = as_tibble(df)
  prob <- predict(mod_fit, newdata = df, type = "response")
  pred <- factor(if_else(prob > 0.5, 1, 0))
  mean(df$wlf != pred)
}


##### Bagging & Random Forest

# create a tibble of mtry values
wlf_rf <- wild_validation %>% 
  crossing(tibble(mtry = 1:(ncol(wildfire_train) - 1)))
# returns a random forest where wlf is the outcome
fitRF <- function(data, mtry){
  data = as_tibble(data)
  return(randomForest(wlf ~ ., data = data, mtry = mtry))
}
wlf_rf <- wlf_rf %>%
  mutate(model_fit = map2(train, mtry, fitRF),
         test_merr = map2_dbl(model_fit, test, test_err)) 
# the best mtry for random forest
wlf_rf_err <- wlf_rf %>%
  group_by(mtry) %>%
  summarise(mean_merr = mean(test_merr)) %>% 
  arrange(mean_merr)
rf_tune2 <- wlf_rf %>%
  pluck("mtry", 1)

##### Boosting

# return a boosting model where wlf is the outcome
fitBoosting <- function(dat){
  dat = as_tibble(dat)
  mat = wildfire_train %>% dplyr::select(- wlf) %>% # encode on full boston df
    onehot::onehot() %>% # use onehot to encode variables
    predict(dat) # get OHE matrix
  return(xgb.DMatrix(data = mat, label = dat$wlf))
}

fit_wlf <- function(train_data, learning_rate, depth, nrounds, silent = 1){
  return(xgb.train(params = list(eta = learning_rate, 
                                 max_depth = depth, 
                                 silent = silent), 
                   train_data, 
                   nrounds = nrounds)
  )
}

xg_mse <- function(model, test){
  prob = predict(model, test, type = "response")
  vals = getinfo(test, "label")
  pred <- factor(if_else(prob > 0.5, 1, 0))
  mean(vals != pred)
}

wlf_boosting_data <- wild_validation %>%
  mutate(train_dg = map(train, fitBoosting), 
         test_dg = map(test, fitBoosting))

wlf_boosting <- wlf_boosting_data %>%
  crossing(tibble(learning_rate = c(0.001, 0.01, 0.1, 0.2)))

wlf_boosting <- wlf_boosting %>%
  mutate(model_fit = map2(train_dg, learning_rate, fit_wlf, depth = 10, nrounds = 5000), 
         test_merr = map2_dbl(model_fit, test_dg, xg_mse))
# 0.001 is the best learning rate
# the best learning rate for boosting
wlf_boosting_err <- wlf_boosting %>%
  group_by(learning_rate) %>%
  summarise(mean_merr = mean(test_merr)) %>%
  arrange(mean_merr)
boosting_tune2 <- wlf_boosting %>%
  pluck("learning_rate", 1)


##### Ridge

# ridge regression
ridge_cv <- wildfire_train %>% 
  cv.glmnet(formula = wlf ~ ., data = ., alpha = 0, nfolds = 10, lambda = lambda_grid)

# ridge's best lambdas
ridge_lambda_min <- ridge_cv$lambda.min


##### Logistic Regression

wlf_glm <- function(data){
  glm(wlf ~ ., data = data, family = binomial)
}


##### Compare mean test error to find the best model

# mean test error for random forests, bagging, logistic regression and ridge regression
merr_part1 <- wildfire %>%
  mutate(RandomForest = map2(wild_train, rf_tune2, fitRF),
         Bagging = map2(wild_train, 16, fitRF),
         Logistic = map(wild_train, wlf_glm),
         Ridge = map(wild_train, ~glmnet(wlf ~ ., data = .x, alpha = 0, lambda = ridge_lambda_min))) %>%
  gather(key = "model", value = "fit", RandomForest, Bagging, Logistic, Ridge) %>%
  mutate(test_merr = map2_dbl(fit, wild_test, test_err))%>%
  select(model, test_merr)
# mean test error for boosting
fitBoosting2 <- function(dat){
  dat = as_tibble(dat)
  mat = wild %>% dplyr::select(- wlf) %>% # encode on full boston df
    onehot::onehot() %>% # use onehot to encode variables
    predict(dat) # get OHE matrix
  return(xgb.DMatrix(data = mat, label = dat$wlf))
}
merr_boosting <- wildfire %>%
  mutate(train_dg = map(wild_train, fitBoosting2),
         test_dg = map(wild_test, fitBoosting2)) %>%
  mutate(Boosting = map2(train_dg, boosting_tune2, fit_wlf, depth = 10, nrounds = 5000)) %>%
  mutate(test_merr = map2_dbl(Boosting, test_dg, xg_mse)) %>%
  gather(key = "model", value = "fit", Boosting) %>%
  select(model, test_merr)

merr_part1 %>%
  bind_rows(merr_boosting) %>%
  arrange(test_merr)
