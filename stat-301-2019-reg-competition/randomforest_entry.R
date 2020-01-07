# load package
library(modelr)
library(tensorflow)
use_python("/usr/bin/python")
library(keras)
library(ggplot2)
library(randomForest)
library(glmnet) 
library(glmnetUtils)
library(tidyverse)


# read in data
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
# unify the level
test_data <- cbind(total_funding_usd = train[1:1572,1], test[,2:8])

outlier <- subset(subset(train, total_funding_usd == 0), number_of_funding_rounds != 0)
train_new <- setdiff(train, outlier)

# helper function to refill missing data
refill <- function(dat) {
  dat <- cbind(dat, na_number = rep(0),nrow(dat))
  for (i in 1:nrow(dat)){
    dat[i,9] = sum(is.na(dat[i,]))}
  dat = as_tibble(dat)
  part1 = dat[,c(2,8)]
  part1[is.na(part1)] = "unknown"
  part2 = dat[,3:7]
  part2 = scale(part2)
  part2[is.na(part2)] = 0
  complete <- cbind(dat[,1], part1, part2, dat[,9])
  complete$category <- as.factor(complete$category)
  complete$region <- as.factor(complete$region)
  return(complete)
}

# prepare the data
all_data <- bind_rows(train_new, test_data)
all_data_refill <- refill(all_data)
train_refill <- all_data_refill[1:15181,]
test_refill <- all_data_refill[15182:16753,]

refill2 <- function(dat) {
  dat <- cbind(dat, na_number = rep(0),nrow(dat))
  for (i in 1:nrow(dat)){
    dat[i,9] = sum(is.na(dat[i,]))}
  dat = as_tibble(dat)
  part1 = dat[,c(2,8)]
  part1[is.na(part1)] = "unknown"
  part2 = dat[,4:7]
  part2 = scale(part2)
  part2[is.na(part2)] = 0
  complete <- cbind(dat[,1], part1, dat[,3], part2, dat[,9])
  complete$category <- as.factor(complete$category)
  complete$region <- as.factor(complete$region)
  return(complete)
}
all_data_refill2 <- refill2(all_data)
train_refill2 <- all_data_refill2[1:15181,]
test_refill2 <- all_data_refill2[15182:16753,]

# build random forests model
fitRF_mtry <- function(data, mtry){
  data <- as_tibble(data)
  return(randomForest(total_funding_usd ~ ., data = data, mtry = mtry))
}
fitRF_nodesize <- function(data, nodesize){
  data <- as_tibble(data)
  return(randomForest(total_funding_usd ~ ., data = data, mtry = 2, nodesize = nodesize))
}
set.seed(1)
train_validation <- train_refill %>%
  crossv_kfold(2, id = "fold")
# cross validation to choose mtry
set.seed(1)
RandomForest_mtry <- train_validation %>% 
  crossing(tibble(mtry = 1:8)) 
RandomForest_mtry <- RandomForest_mtry %>%
  mutate(model = map2(train, mtry, fitRF_mtry),
         mse = map2(model, test, mse)) 
RandomForest_mtry_mse <- RandomForest_mtry %>%
  group_by(mtry) %>%
  summarise(mean_mse = mean(as.numeric(mse))) %>% 
  arrange(mean_mse) 
# cross validation to choose nodesize
set.seed(1)
RandomForest_nodesize <- train_validation %>% 
  crossing(tibble(nodesize = 10:40)) 
RandomForest_nodesize <- RandomForest_nodesize %>%
  mutate(model = map2(train, nodesize, fitRF_nodesize),
         mse = map2(model, test, mse)) 
RandomForest_nodesize_mse <- RandomForest_nodesize %>%
  group_by(nodesize) %>%
  summarise(mean_mse = mean(as.numeric(mse))) %>% 
  arrange(mean_mse) 
ggplot(RandomForest_nodesize_mse, aes(x = nodesize, y = mean_mse)) + geom_smooth()
# final model
RandomForest_model <- randomForest(total_funding_usd ~ ., data = train_refill, mtry = 2, nodesize = 30)
pred <- predict(RandomForest_model, newdata = test_refill)

# write data
for (i in 1:length(pred)){
  if (test_refill2$number_of_funding_rounds[i] == 0)
  {
    pred[i] = 0
  }
}
output <- bind_cols(id = test[1:1572,1], pred %>% list()) %>%
  rename("total_funding_usd" = "V1")

write.csv(output, "output.csv", row.names = FALSE)