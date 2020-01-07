# Once Connected students must run installations

# tensorflow
install.packages("tensorflow")
library(tensorflow)
use_python("/usr/bin/python")
install_tensorflow()

# Ignore any warnings about Python dependencies.
# R Session will restart after installation.

# keras
install.packages("keras")
library(keras)
install_keras()

# Ignore depreciation warnings and CPU related warnings.
# R Session will restart after installation.

# Must run this code at the top of all scripts when using 
# RStudio on analytics node
library(tensorflow)
use_python("/usr/bin/python")
library(keras)