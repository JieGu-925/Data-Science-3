# load package
library(e1071)
library(tidyverse)

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
competition <- tibble(train = competition_train_refill %>% list(),
                      test = test_data_refill %>% list())

# cross validation to finde the best parameters
set.seed(1)
radial_svm <- competition %>%
  mutate(model = map(.x = train, 
                     .f = function(x) tune(svm, status ~ ., data = x, kernel = "radial", 
                                           ranges = list(cost = seq(0.01, 10, length = 20), 
                                                         gamma = seq(0.001, 1, length = 20)))))
# find the best tuning parameters  
radial_svm$model[[1]]$best.parameters

# final model
svm_radial_model <- competition %>%
  mutate(model = map(.x = train, 
                     .f = function(x) svm(status ~ ., data = x, kernel = "radial", cost =1 , 
                                          gamma = 0.3, probability = TRUE)),
         pred = map2(model, test, ~ predict(.x, .y, probability = TRUE)))

# write data
output <- bind_cols(ID = competition_test[,1], svm_radial_model$pred %>% list()) %>%
  rename("status" = "V1")
write.csv(output, "output.csv", row.names = FALSE)
