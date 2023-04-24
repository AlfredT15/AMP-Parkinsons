# Load the required library
library(data.table)

# Read in the train_clinical_data.csv and supplemental_clinical_data.csv files
train <- fread("/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv")
sup <- fread("/kaggle/input/amp-parkinsons-disease-progression-prediction/supplemental_clinical_data.csv")

# Combine the two data tables using the rbind() function
data <- rbind(train, sup)
# Load the required libraries
library(dplyr)
library(tidyr)
library(purrr)
library(stats)

# Define the target variables
target <- c("updrs_1", "updrs_2", "updrs_3", "updrs_4")

# Define a function to train the linear regression model
train_model <- function(data, target_col) {
  
  # Drop rows with missing values in the target column
  data <- data %>% drop_na({{target_col}})
  
  # For updrs_3, drop rows with a value of 0
  if (target_col == "updrs_3") {
    data <- data %>% filter({{target_col}} != 0)
  }
  
  # Extract the predictor and response variables
  X <- data$visit_month
  y <- data[[target_col]]
  
  # Train linear regression model
  trained <- lm(y ~ X)
  
  # Return the trained model
  return(trained)
}

# Train a linear regression model for each target variable
models <- map(setNames(target, target), ~ train_model(data, .x))
summary(models)
# Calculate R-squared for each trained model
rsquared <- map_dbl(models, ~ summary(.x)$r.squared)

# Print the R-squared values for each target variable
cat(paste0("R-squared for ", target, ": ", round(rsquared, 4), "\n"))
