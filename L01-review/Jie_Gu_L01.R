# load packages
library(tidyverse)
library(modelr)

# read in data
wildfire_dat <- read_csv("data/wildfires.csv") %>%
  mutate(winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
         traffic = factor(traffic, levels = c("lo", "med", "hi")))

#### Exercise 1 
# Model 1
mod01_burned_fn <- function(df){
  lm(burned ~ . - wlf, data = df)
}

# Model 2
mod02_burned_fn <- function(df){
  lm(burned ~ poly(windspd, 3) + poly(rain, 3) + poly(vulnerable, 3) + heli, data = df)
}

# 10-fold cross validation
set.seed(1)
trained_reg_model <- wildfire_dat %>%
  crossv_kfold(10, id = "fold")

# Reshape data -- gather model
trained_reg_model <- trained_reg_model %>%
  mutate(model1 = map(train, mod01_burned_fn),
         model2 = map(train, mod02_burned_fn)) %>%
  gather(key = "model", value = "fit", model1, model2)

# Calculate MSE
trained_reg_model <- trained_reg_model %>%
  mutate(mse_mod = map2_dbl(fit, test, mse))

# Present mean test MSE
trained_reg_model %>%
  group_by(model) %>%
  summarise(mean_mse = mean(mse_mod))

#### Exercise 3
# Critical functions
# calculate test error
test_err <- function(df, pred){
  mean(df$wlf != pred)
}

# calculate predicted prob of default
predict_fn <- function(mod_fit, df){
  predict(mod_fit, newdata = df, type = "response")
}

# calculate pred default 
predict_default <- function(prob, prior_prob = 0.5){
  factor(if_else(prob > prior_prob, 1, 0))
}

# Model 1
mod01_wlf_fn <- function(df){
  glm(wlf ~ . - wlf, data = df, family = binomial)
}

# Model 2
mod02_wlf_fn <- function(df){
  glm(wlf ~ poly(windspd, 2) + winddir + poly(rain, 2) + poly(vulnerable, 3) + x*y, data = df)
}

# 10-fold cross validation
set.seed(1)
trained_log <- wildfire_dat %>%
  crossv_kfold(10, id = "fold")

# reshape data -- gather models
trained_log <- trained_log %>%
  mutate(mod01 = map(train, mod01_wlf_fn),
         mod02 = map(train, mod02_wlf_fn)) %>%
  gather(key = "model", value = "fit", mod01, mod02)

# Calculate fold_Err
trained_log <- trained_log %>%
  mutate(pred_prob = map2(fit, test, predict_fn),
         pred_default = map(pred_prob, predict_default),
         test = map(test, as_tibble),
         fold_Err = map2_dbl(test, pred_default, test_err))

# Calculate test_Err  
trained_log %>%
  group_by(model) %>%
  summarise(test_Err = mean(fold_Err))

#### Challenge (Not Mandatory)
set.seed(1)
# Lasso regression
lasso_cv <- wildfire_dat %>% 
  cv.glmnet(formula = burned ~ . -name, 
            data = ., alpha = 1, nfolds = 10)
# lasso's best lambdas
lasso_lambda_min <- lasso_cv$lambda.min

trained_reg_model <- tibble(train = wildfire_dat %>% list(),
                            test  = wildfire_dat %>% list()) %>%
  mutate(model1 = map(train, mod01_burned_fn),
         model2 = map(train, mod02_burned_fn),
         lasso_min = map(train, ~ glmnet(burned ~ . -name, data = .x,
                                         alpha = 1, lambda = lasso_lambda_min)))

xg_mse <- function(model, test){
  preds = map2(model, test, predict)
  test_mse = map2_dbl(test, preds, ~ mean((.x$burned - .y)^2))
  return(test_mse)
}
# Calculate MSE
trained_reg_model <- trained_reg_model %>%
  mutate(mse_mod1 = map2_dbl(model1, test, mse),
         mse_mod2 = map2_dbl(model2, test, mse),
         mse_lasso = xg_mse(lasso_min, test))
# Present mean test MSE
trained_reg_model %>%
  gather(key = "mse", value = "mse_mod", mse_mod1, mse_mod2, mse_lasso) %>%
  group_by(mse) %>%
  summarise(mean_mse = mean(mse_mod))