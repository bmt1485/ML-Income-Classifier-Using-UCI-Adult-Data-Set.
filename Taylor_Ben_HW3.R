##################################################
# ECON 418-518 Homework 3
# Ben Taylor
# The University of Arizona
# bmt1485@arizona.edu 
# 28 November 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)

# Set sead
set.seed(418518)

# Set working directory 
setwd("/Users/bentaylor/Desktop/ECON 418 R/Data")

# Load data
data <- read.csv("ECON_418-518_HW3_Data.csv")

#####################
# Problem 1
#####################


#################
# Question (i)
#################

# Drop  columns
data <- subset(data, select = -c(fnlwgt, occupation, relationship, capital.gain, capital.loss, educational.num))



#################
# Question (ii)
#################

##############
# Part (a)
##############

# Convert "income" to binary
data$income <- ifelse(data$income == ">50K", 1, 0)

##############
# Part (b)
##############

# Convert "race" to binary
data$race <- ifelse(data$race == "White", 1, 0)

##############
# Part (c)
##############

# Convert "gender" to binary
data$gender <- ifelse(data$gender == "Male", 1, 0)

##############
# Part (d)
##############

# Convert "workclass" to binary
data$workclass <- ifelse(data$workclass == "Private", 1, 0)

##############
# Part (e)
##############

# Convert "native.country" to binary
data$native.country <- ifelse(data$native.country == "United-States", 1, 0)

##############
# Part (f)
##############

# Convert "marital.status" to binary
data$marital.status <- ifelse(data$marital.status == "Married-civ-spouse", 1, 0)

##############
# Part (g)
##############

# Convert "education" to binary
data$education <- ifelse(data$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)

##############
# Part (h)
##############

# Create "age sq" variable
data$age.sq <- data$age^2

##############
# Part (i)
##############

# Standardize variables
data$age <- scale(data$age)
data$age.sq <- scale(data$age.sq)
data$hours.per.week <- scale(data$hours.per.week)

#################
# Question (iii)
#################

##############
# Part (a)
##############

# see the proportion with income >50k, 
income_over_50k <- mean(data$income ==1)
income_over_50k

##############
# Part (b)
##############

# see the proportion in the private sector
private_sector <- mean(data$workclass ==1)
private_sector

##############
# Part (c)
##############

# see the proportion that are married
married <- mean(data$marital.status ==1)
married

##############
# Part (d)
##############

# see proportion that are female
female <- mean(data$gender ==0)
female

##############
# Part (e)
##############

# see the number of NAs in the data
num_NAs <- sum(is.na(data))
num_NAs

##############
# Part (f)
##############

# change income to a factor
data$income <- as.factor(data$income)

#################
# Question (iv)
#################

##############
# Part (a)
##############

# find the last observation in the training set
last_training <- floor(nrow(data)*.70)
last_training

##############
# Part (b)
##############

# create training set, check size
training <- data[1:last_training,]
nrow(training)

##############
# Part (c)
##############

# create testing set, check size
testing <- data[(last_training + 1): nrow(data),]
nrow(testing)

##############


#################
# Question (v)
#################

##############
# Part (A)
##############

# ANSWER IN MY NOTES

##############
# Part (b)
##############
#download libraries 
install.packages("caret")
install.packages("glmnet")

# Load required libraries
library(caret)
library(glmnet)

# Define the predictors (all columns except 'income') and outcome ('income')
predictors <- training[, -which(names(training) == "income")]
outcome <- training$income

# Train  Lasso regression model
lasso_grid <- expand.grid(alpha = 1, lambda = seq(10^5, 10^-2, length = 50))
lasso_model <- train(
  x = as.matrix(predictors), y = outcome,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = lasso_grid
)

# Display the best lambda value
cat("Best lambda for Lasso:", lasso_model$bestTune$lambda, "\n")


##############
# Part (c)
##############

# Classification accuracy of the best Lasso model
lasso_accuracy <- max(lasso_model$results$Accuracy)
cat("Classification accuracy of best Lasso model:", lasso_accuracy, "\n")


##############
# Part (d)
##############

# Extract coefficients for the best lambda
lasso_coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
cat("Non-zero coefficients:\n")
print(lasso_coefficients[lasso_coefficients[, 1] != 0, ])

##############
# Part (e)
##############

# Get names of non-zero coefficient variables
non_zero_vars <- rownames(lasso_coefficients)[lasso_coefficients[, 1] != 0]
non_zero_vars <- non_zero_vars[non_zero_vars != "(Intercept)"]

# Create new training data with selected variables
selected_training <- training[, c(non_zero_vars, "income")]

# Train Lasso model with selected variables
lasso_selected_model <- train(
  x = as.matrix(selected_training[, -which(names(selected_training_data) == "income")]),
  y = selected_training_data$income,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = lasso_grid
)

# Train Ridge model with selected variables
ridge_grid <- expand.grid(alpha = 0, lambda = seq(10^5, 10^-2, length = 50))
ridge_selected_model <- train(
  x = as.matrix(selected_training_data[, -which(names(selected_training_data) == "income")]),
  y = selected_training_data$income,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = ridge_grid
)

# Compare classification accuracies
lasso_selected_accuracy <- max(lasso_selected_model$results$Accuracy)
ridge_selected_accuracy <- max(ridge_selected_model$results$Accuracy)

cat("Lasso classification accuracy with selected variables:", lasso_selected_accuracy, "\n")
cat("Ridge classification accuracy with selected variables:", ridge_selected_accuracy, "\n")


#################
# Question (vi)
#################

##############
# Part (a)
##############

# answered in notes

##############
# Part (b)
##############

# install + load packages
install.packages("randomForest")  # If not already installed
library(randomForest)
library(caret)

# Define control for cross-validation
rf_control <- trainControl(method = "cv", number = 5)

# Define grid of parameters
rf_grid <- expand.grid(
  mtry = c(2, 5, 9)  # Number of random features to consider at each split
)

# Train Random Forest with 100 trees
rf_100 <- randomForest(
  x = training[, -which(names(training) == "income")],
  y = training$income,
  ntree = 100,
  mtry = 2  # Example: Adjust `mtry` manually
)

# Train Random Forest with 200 trees
rf_200 <- randomForest(
  x = training[, -which(names(training) == "income")],
  y = training$income,
  ntree = 200,
  mtry = 2 
)

# Train Random Forest with 300 trees
rf_300 <- randomForest(
  x = training[, -which(names(training) == "income")],
  y = training$income,
  ntree = 300,
  mtry = 2  
)

##############
# Part (c)
##############

# Predict on training data for RF with 100 trees
rf_100_pred <- predict(rf_100, newdata = training[, -which(names(training) == "income")])
rf_100_accuracy <- mean(rf_100_pred == training$income)

# Predict on training data for RF with 200 trees
rf_200_pred <- predict(rf_200, newdata = training[, -which(names(training) == "income")])
rf_200_accuracy <- mean(rf_200_pred == training$income)

# Predict on training data for RF with 300 trees
rf_300_pred <- predict(rf_300, newdata = training[, -which(names(training) == "income")])
rf_300_accuracy <- mean(rf_300_pred == training$income)

# Print accuracies
cat("Accuracy of RF with 100 trees:", rf_100_accuracy, "\n")
cat("Accuracy of RF with 200 trees:", rf_200_accuracy, "\n")
cat("Accuracy of RF with 300 trees:", rf_300_accuracy, "\n")

# Identify the best model
if (rf_100_accuracy >= rf_200_accuracy & rf_100_accuracy >= rf_300_accuracy) {
  best_rf_model <- "RF with 100 trees"
  best_accuracy <- rf_100_accuracy
} else if (rf_200_accuracy >= rf_100_accuracy & rf_200_accuracy >= rf_300_accuracy) {
  best_rf_model <- "RF with 200 trees"
  best_accuracy <- rf_200_accuracy
} else {
  best_rf_model <- "RF with 300 trees"
  best_accuracy <- rf_300_accuracy
}

# Print the best model and its accuracy
cat("Best Random Forest model:", best_rf_model, "\n")
cat("Classification accuracy of the best model:", best_accuracy, "\n")


##############
# Part (d)
##############

# Compare accuracies
cat("Best RF Accuracy:", max(rf_300$results$Accuracy), "\n")  
cat("Best Lasso Accuracy:", lasso_selected_accuracy, "\n")
cat("Best Ridge Accuracy:", ridge_selected_accuracy, "\n")


##############

##############
# Part (e)
##############

# Predict on the training data using the best Random Forest model
best_rf_pred <- predict(rf_300, newdata = training[, -which(names(training) == "income")])

# Ensure the outcome variable is a factor
training$income <- as.factor(training$income)

# Create confusion matrix
conf_matrix <- confusionMatrix(best_rf_pred, training$income)
print(conf_matrix)


# Extract false positives and false negatives
false_positives <- conf_matrix$table[2, 1]  # Predicted 1 but actual 0
false_negatives <- conf_matrix$table[1, 2]  # Predicted 0 but actual 1

cat("False Positives:", false_positives, "\n")
cat("False Negatives:", false_negatives, "\n")


##############





