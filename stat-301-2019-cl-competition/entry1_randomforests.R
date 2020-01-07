# Load package
library(modelr)
library(utils)
library(skimr)
library(janitor)
library(ggfortify)
library(tidyverse)
library(randomForest)

# read in data
competition_train <- read.csv("competition_train.csv", stringsAsFactors = FALSE)
competition_test <- read.csv("competition_test.csv", stringsAsFactors = FALSE)
# unify the level
test_data <- cbind(status = competition_train[1:636,1], competition_test[,2:9])

# helper function to refill missing data
refill <- function(dat) {
  dat <- cbind(dat, na_number = rep(0),nrow(dat))
  for (i in 1:nrow(dat)){
    dat[i,10] = sum(is.na(dat[i,]))}
  dat = as.tibble(dat)
  part1 = dat[,1:3]
  part1[is.na(part1)] = "unknown"
  part2 = dat[,4:9]
  part2 = scale(part2)
  part2[is.na(part2)] = 0
  complete <- cbind(part1, part2, dat[,10])
  complete$status <- as.factor(complete$status)
  complete$category <- as.factor(complete$category)
  complete$region <- as.factor(complete$region)
  return(complete)
}

# prepare the data
competition_all <- bind_rows(competition_train, test_data)
competition_all_refill <- refill(competition_all)
competition_train_refill <- competition_all_refill[1:11158,]
test_data_refill <- competition_all_refill[11159:11794,]

# cross validation to choose mtry parameter
# helper function to choose mtry
fitRF_mtry <- function(data, mtry){
  data <- as_tibble(data)
  return(randomForest(status ~ ., data = data, mtry = mtry))
}
test_err <- function(model, df){
  df <- as_tibble(df)
  pred <- predict(model, newdata = df, type = "response")
  return (mean(df$status != pred))
}
set.seed(1)
train_validation_mtry <- competition_train_refill %>%
  crossv_kfold(5, id = "fold")
set.seed(1)
RandomForest_mtry <- train_validation_mtry %>% 
  crossing(tibble(mtry = 1:9)) 
RandomForest_mtry <- RandomForest_mtry %>%
  mutate(model = map2(train, mtry, fitRF_mtry),
         error = map2(model, test, test_err)) 
RandomForest_mtry_error <- RandomForest_mtry %>%
  group_by(mtry) %>%
  summarise(mean_error = mean(as.numeric(error))) %>% 
  arrange(mean_error) 
# the best mtry for random forest: 2
RandomForest_mtry_error

# helper function to choose nodesize
fitRF_nodesize <- function(data, nodesize){
  data <- as_tibble(data)
  return(randomForest(status ~ ., data = data, mtry = 2, nodesize = nodesize))
}
set.seed(1)
train_validation_nodesize <- competition_train_refill %>%
  crossv_kfold(5, id = "fold")
set.seed(1)
RandomForest_nodesize <- train_validation_nodesize %>% 
  crossing(tibble(nodesize = 6:20)) 
RandomForest_nodesize <- RandomForest_nodesize %>%
  mutate(model = map2(train, nodesize, fitRF_nodesize),
         error = map2(model, test, test_err)) 
RandomForest_nodesize_error <- RandomForest_nodesize %>%
  group_by(nodesize) %>%
  summarise(mean_error = mean(as.numeric(error))) %>% 
  arrange(mean_error) 
# the best nodesize for random forest: 17
RandomForest_nodesize_error

# the final model
RandomForest_model <- randomForest(status ~ ., data = competition_train_refill, mtry = 2, nodesize = 17)
pred <- predict(RandomForest_model, newdata = test_data_refill, type = "response")

# write data
output <- bind_cols(ID = competition_test[,1], pred %>% list()) %>%
  rename("status" = "V1")
write.csv(output, "output.csv", row.names = FALSE)