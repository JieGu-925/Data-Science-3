
#### Load Packages


# Loading package(s)
library(modelr)
library(tensorflow)
use_python("/usr/bin/python")
library(keras)
library(ggplot2)
library(e1071)
library(randomForest)
library(xgboost)
library(glmnet) 
library(glmnetUtils)
library(tidyverse)


# read in data
wildfires_train <- read_csv("data/wildfires_train.csv") %>%
  mutate(winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
         traffic = factor(traffic, levels = c("lo", "med", "hi")))
wildfires_test <- read_csv("data/wildfires_test.csv") %>%
  mutate(winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
         traffic = factor(traffic, levels = c("lo", "med", "hi")))
wildfires_all <- bind_rows(wildfires_train, wildfires_test)
wildfires <- tibble(train = wildfires_train %>% list(),
                    test = wildfires_test %>% list())


#### Exercise 1

  ##### Neural Network

# one-hot encode
burned_onehot <- function(dat){
  dat = as_tibble(dat)
  mat = wildfires_all %>%
    dplyr::select(-c(burned, wlf)) %>%
    onehot::onehot() %>%
    predict(dat)
  return(mat)
}
# prepare the data
burned_train_data <- burned_onehot(wildfires_train)
burned_train_targets <- as.matrix(wildfires_train[,"burned"])
burned_test_data <- burned_onehot(wildfires_test)
burned_test_targets <- as.matrix(wildfires_test[,"burned"])
# scale the data
burned_train_data <- scale(burned_train_data)
burned_test_data <- scale(burned_test_data)

# build the network
build_burned_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 16, activation = "relu", input_shape = dim(burned_train_data)[[2]]) %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 1) 
  model %>% 
    compile(optimizer = "rmsprop", 
            loss = "mse", 
            metrics = c("mse"))
}

# k-fold cross validation
set.seed(1)
k <- 4
indices <- sample(1:nrow(burned_train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 
all_mse_histories <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE) 
  val_data <- burned_train_data[val_indices,]
  val_targets <- burned_train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- burned_train_data[-val_indices,]
  partial_train_targets <- burned_train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  burned_model <- build_burned_model()
  
  # Train the model (in silent mode, verbose=0)
  burned_history <- burned_model %>% 
    fit(partial_train_data, partial_train_targets,
        validation_data = list(val_data, val_targets),
        epochs = 100, batch_size = 1, verbose = 0)
  mse_history <- burned_history$metrics$val_mean_squared_error
  all_mse_histories <- rbind(all_mse_histories, mse_history)
}
# compute the average of the per-epoch MSE scores for all folds
average_mse_history <- data.frame(
  epoch = seq(1:ncol(all_mse_histories)),
  validation_mse = apply(all_mse_histories, 2, mean))
# get a clearer picture
ggplot(average_mse_history, aes(x = epoch, y = validation_mse)) + geom_smooth()

# Build the final model
burned_model <- build_burned_model()
# Train the model (in silent mode, verbose=0)
burned_model %>% fit(burned_train_data, burned_train_targets,
                     epochs = 27, batch_size = 1, verbose = 0)
# Evaluate the model on the testing data
burned_results <- burned_model %>% evaluate(burned_test_data, burned_test_targets, verbose = 0)
burned_nn_rmse <- sqrt(burned_results$mean_squared_error)
burned_nn_rmse


##### Other models - Tree

# helper function
xg_rmse <- function(model, test){
  preds = predict(model, test)
  vals = getinfo(test, "label")
  return(sqrt(mean((preds - vals)^2)))
}
# boosting
burned_Boosting <- tibble(train = xgb.DMatrix(data = burned_train_data, label = burned_train_targets)
                          %>% list(),
                          test = xgb.DMatrix(data = burned_test_data, label = burned_test_targets)
                          %>% list()) %>%
  mutate(model = map2(train, 5000, xgb.train, 
                      params = list(eta = 0.01, max_depth = 10)),
         Boosting = map2_dbl(model, test, xg_rmse)) %>%
  gather(key = "Model", value = "RMSE", Boosting) %>%
  select(Model, RMSE)
# bagging
burned_Bagging <- wildfires %>%
  mutate(model = map2(.x = train, .y = 15, ~randomForest(burned ~ . - wlf, data = .x, mtry = .y)),
         Bagging = map2_dbl(model, test, rmse)) %>%
  gather(key = "Model", value = "RMSE", Bagging) %>%
  select(Model, RMSE)
# random forests
burned_RandomForest <- wildfires %>%
  mutate(model = map2(.x = train, .y = 6, ~randomForest(burned ~ . - wlf, data = .x, mtry = .y)),
         RandomForest = map2_dbl(model, test, rmse)) %>%
  gather(key = "Model", value = "RMSE", RandomForest) %>%
  select(Model, RMSE)

##### Other models - Ridge & Linear regression

# ridge
burned_Ridge <- wildfires %>%
  mutate(model = map(train, ~glmnet(burned ~ .- wlf, data = .x, alpha = 0, lambda = 3.45)),
         pred = map2(model, test, predict),
         Ridge = map2_dbl(test, pred, ~sqrt( mean((.x$burned - .y)^2)))) %>%
  gather(key = "Model", value = "RMSE", Ridge) %>%
  select(Model, RMSE)
# linear regression
Linear <- wildfires %>%
  mutate(model = map(train, ~lm(burned ~ .-wlf, data = .x)),
         Linear = map2_dbl(model, test, rmse)) %>%
  gather(key = "Model", value = "RMSE", Linear) %>%
  select(Model, RMSE)

##### Compare the RMSE

burned_NN <- tibble(Model = "NeuralNetwork",
                    RMSE = burned_nn_rmse)
compare_rmse <- burned_Bagging %>%
  bind_rows(burned_RandomForest) %>%
  bind_rows(burned_Boosting) %>%
  bind_rows(burned_Ridge) %>%
  bind_rows(Linear) %>%
  bind_rows(burned_NN) %>%
  arrange(RMSE)
compare_rmse

  #### Exercise 2

  ##### Neural network

# one-hot encode
wlf_onehot <- function(dat){
  dat = as_tibble(dat)
  mat = wildfires_all %>%
    dplyr::select(-c(wlf)) %>%
    onehot::onehot() %>%
    predict(dat)
  return(mat)
}
# prepare the data
wlf_train_data <- wlf_onehot(wildfires_train)
wlf_train_label <- as.matrix(wildfires_train[,"wlf"])
wlf_test_data <- wlf_onehot(wildfires_test)
wlf_test_label <- as.matrix(wildfires_test[,"wlf"])
# scale the data
wlf_train_data <- scale(wlf_train_data)
wlf_test_data <- scale(wlf_test_data)

# build the network
build_wlf_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 16, activation = "relu", input_shape = dim(wlf_train_data)[[2]]) %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid") 
  model %>% 
    compile(optimizer = "rmsprop", 
            loss = "binary_crossentropy", 
            metrics = c("accuracy"))
}

# k-fold cross validation
set.seed(1)
k <- 4
indices <- sample(1:nrow(wlf_train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 
all_err_histories <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE) 
  val_data <- wlf_train_data[val_indices,]
  val_label <- wlf_train_label[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- wlf_train_data[-val_indices,]
  partial_train_label <- wlf_train_label[-val_indices]
  
  # Build the Keras model (already compiled)
  wlf_model <- build_wlf_model()
  
  # Train the model (in silent mode, verbose=0)
  wlf_history <- wlf_model %>% 
    fit(partial_train_data, partial_train_label,
        validation_data = list(val_data, val_label),
        epochs = 100, batch_size = 1, verbose = 0)
  err_history <- 1 - wlf_history$metrics$val_acc
  all_err_histories <- rbind(all_err_histories, err_history)
}
# compute the average of the per-epoch MSE scores for all folds
average_err_history <- data.frame(
  epoch = seq(1:ncol(all_err_histories)),
  validation_err = apply(all_err_histories, 2, mean))
# get a clearer picture
ggplot(average_err_history, aes(x = epoch, y = validation_err)) + geom_smooth()

# Build the final model
wlf_model <- build_wlf_model()
# Train the model (in silent mode, verbose=0)
wlf_model %>% fit(wlf_train_data, wlf_train_label,
                  epochs = 25, batch_size = 1, verbose = 0)
# Evaluate the model on the testing data
wlf_results <- wlf_model %>% evaluate(wlf_test_data, wlf_test_label, verbose = 0)
wlf_nn_error <- 1 - wlf_results$acc
wlf_nn_error


##### Other models - SVM

wildfire_train <- wildfires_train
wildfire_test <- wildfires_test
wildfire_train$wlf <- as.factor(wildfires_train$wlf)
wildfire_test$wlf <- as.factor(wildfires_test$wlf)
wildfire <- tibble(train = wildfire_train %>% list(),
                   test = wildfire_test %>% list())
# linear SVM
svm_linear <- wildfire %>%
  mutate(model_fit = map(.x = train, # fit the model
                         .f = function(x) svm(wlf ~ ., data = x,  kernel = "linear", 
                                              cost = 0.5, probability = TRUE)), 
         test_pred = map2(model_fit, test, 
                          ~ predict(.x, .y, probability = TRUE)), # get predictions on test set
         confusion_matrix = map2(.x = test, .y = test_pred,  # get confusion matrix
                                 .f = function(x, y) caret::confusionMatrix(x$wlf, y)),
         SVM_Linear = 1- confusion_matrix[[1]]$overall[[1]]) %>%
  gather(key = "Model", value = "ERROR", SVM_Linear) %>%
  select(Model, ERROR)
# radial SVM
svm_radial <- wildfire %>%
  mutate(model_fit = map(.x = train, 
                         .f = function(x) svm(wlf ~ ., data = x, kernel = "radial", 
                                              cost = 8, gamma = 0.05, probability = TRUE)),
         test_pred = map2(model_fit, test, ~ predict(.x, .y, probability = TRUE)), # test predcitions
         confusion_matrix = map2(.x = test, .y = test_pred, # test confusion matrix
                                 .f = function(x, y) caret::confusionMatrix(x$wlf, y)),
         SVM_Radial = 1- confusion_matrix[[1]]$overall[[1]]) %>%
  gather(key = "Model", value = "ERROR", SVM_Radial) %>%
  select(Model, ERROR)
# polynomial SVM
svm_polynominal <- wildfire %>%
  mutate(model_fit = map(.x = train, 
                         .f = function(x) svm(wlf ~ ., data = x, kernel = "polynomial", 
                                              cost = 0.01, gamma = 4, degree = 2, probability = TRUE)),
         test_pred = map2(model_fit, test, ~ predict(.x, .y, probability = TRUE)), # test predcitions
         confusion_matrix = map2(.x = test, .y = test_pred, # test confusion matrix
                                 .f = function(x, y) caret::confusionMatrix(x$wlf, y)),
         SVM_Polynomial = 1- confusion_matrix[[1]]$overall[[1]]) %>%
  gather(key = "Model", value = "ERROR", SVM_Polynomial) %>%
  select(Model, ERROR)


##### Other models - Tree

# helper function
xg_err <- function(model, test){
  prob = predict(model, test, type = "response")
  vals = getinfo(test, "label")
  pred <- factor(if_else(prob > 0.5, 1, 0))
  mean(vals != pred)
}
test_err <- function(mod_fit, df){
  prob <- predict(mod_fit, newdata = df, type = "response")
  pred <- factor(if_else(prob > 0.5, 1, 0))
  return(mean(df$wlf != pred))
}
# boosting
wlf_Boosting <- tibble(train = xgb.DMatrix(data = wlf_train_data, label = wlf_train_label) %>% list(),
                       test = xgb.DMatrix(data = wlf_test_data, label = wlf_test_label) %>% list()) %>%
  mutate(model = map2(train, 5000, xgb.train, 
                      params = list(eta = 0.09, max_depth = 10)),
         Boosting = map2_dbl(model, test, xg_err)) %>%
  gather(key = "Model", value = "ERROR", Boosting) %>%
  select(Model, ERROR)
# bagging
wlf_Bagging <- wildfires %>%
  mutate(model = map2(.x = train, .y = 16, ~randomForest(wlf ~ ., data = .x, mtry = .y)),
         Bagging = map2_dbl(model, test, test_err)) %>%
  gather(key = "Model", value = "ERROR", Bagging) %>%
  select(Model, ERROR)
# random forests
wlf_RandomForest <- wildfires %>%
  mutate(model = map2(.x = train, .y = 7, ~randomForest(wlf ~ ., data = .x, mtry = .y)),
         RandomForest = map2_dbl(model, test, test_err)) %>%
  gather(key = "Model", value = "ERROR", RandomForest) %>%
  select(Model, ERROR)


##### Other models - ridge & logistic

# ridge
wlf_Ridge <- wildfires %>%
  mutate(model = map(.x = train, ~glmnet(wlf ~ ., data = .x, alpha = 0, lambda = 3.37)),
         Ridge = map2_dbl(model, test, test_err)) %>%
  gather(key = "Model", value = "ERROR", Ridge) %>%
  select(Model, ERROR)
# logistic regression
Logistic <- wildfires %>%
  mutate(model = map(.x = train, ~glm(wlf ~., data = .x, family = binomial)),
         Logistic = map2_dbl(model, test, test_err)) %>%
  gather(key = "Model", value = "ERROR", Logistic) %>%
  select(Model, ERROR)


##### Compare the RMSE

wlf_NN <- tibble(Model = "NeuralNetwork",
                 ERROR = wlf_nn_error)
compare_error <- wlf_Bagging %>%
  bind_rows(wlf_RandomForest) %>%
  bind_rows(wlf_Boosting) %>%
  bind_rows(wlf_Ridge) %>%
  bind_rows(Logistic) %>%
  bind_rows(wlf_NN) %>%
  bind_rows(svm_linear) %>%
  bind_rows(svm_radial) %>%
  bind_rows(svm_polynominal) %>%
  arrange(ERROR)
compare_error
