### Review Lab (L01)
### Data Science III (STAT 301-3) - Spring 2018

# Load packages
library(tidyverse)
library(modelr)

# Read in data
wildfire_dat <- read_csv("data/wildfires.csv") %>%
  mutate(winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
         traffic = factor(traffic, levels = c("lo", "med", "hi")))

###----------------------------------###
# Candidate models for Exercise 1
###----------------------------------###

# Model 1
mod01_burned_fn <- function(df){
  lm(burned ~ . - wlf, data = df)
}

# Model 2
mod02_burned_fn <- function(df){
  lm(burned ~ poly(windspd, 3) + poly(rain, 3) + poly(vulnerable, 3) + heli, data = df)
}


###----------------------------------###
# Candidate models for Exercise 3
###----------------------------------###

# Model 1
mod01_wlf_fn <- function(df){
  glm(wlf ~ . - wlf, data = df, family = binomial)
}

# Model 2
mod02_wlf_fn <- function(df){
  glm(wlf ~ poly(windspd, 2) + winddir + poly(rain, 2) + poly(vulnerable, 3) + x*y, data = df)
}
