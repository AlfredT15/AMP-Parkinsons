# Load necessary libraries
library(dplyr)
library(tidyr)
library(caret)
library(glmnet)
library(tidyverse)
library(magrittr)

setwd("C:/Users/nmarx/Downloads/R STUDIO 441/FINAL PROJECT")

# Load the data
train_peptides <- read.csv("train_peptides.csv")
train_proteins <- read.csv("train_proteins.csv")
train_clinical_data <- read.csv("train_clinical_data.csv")

# Merge the data
merged_data <- train_peptides %>%
  left_join(train_proteins, by = c("visit_id", "visit_month", "patient_id", "UniProt")) %>%
  left_join(train_clinical_data, by = c("visit_id", "visit_month", "patient_id"))

# Prepare the data for modeling
predictors <- merged_data %>%
  select(-c(visit_id, visit_month, patient_id, UniProt, Peptide)) %>%
  na.omit() %>%
  select(-upd23b_clinical_state_on_medication)
response <- merged_data %>%
  select(updrs_1, updrs_2, updrs_3, updrs_4) %>%
  na.omit()

# Create the multivariate regression model
model <- glmnet(predictors, response, alpha = 1)

# Make predictions on the test data
test_peptides <- read.csv("test_peptides.csv")
test_proteins <- read.csv("test_proteins.csv")
test_clinical_data <- read.csv("test_clinical_data.csv")
test_merged_data <- test_peptides %>%
  left_join(test_proteins, by = c("visit_id", "visit_month", "patient_id", "UniProt")) %>%
  left_join(test_clinical_data, by = c("visit_id", "visit_month", "patient_id"))
test_predictors <- test_merged_data %>%
  select(-c(visit_id, visit_month, patient_id, UniProt, Peptide)) %>%
  na.omit() %>%
  select(-upd23b_clinical_state_on_medication)
test_response <- test_merged_data %>%
  select(updrs_1, updrs_2, updrs_3, updrs_4) %>%
  na.omit()
predictions <- predict(model, newx = test_predictors)

# Evaluate the model performance
rmse <- sqrt(mean((predictions - test_response)^2))
