# ML-Income-Classifier-Using-UCI-Adult-Data-Set.

## Overview
This project implements various machine learning algorithms to classify whether an individual's income exceeds $50,000 using the UCI Adult dataset. The analysis is part of ECON 418 (Econometrics) coursework at the University of Arizona.

Key machine learning algorithms applied:
- Lasso Regression
- Ridge Regression
- Random Forest

The analysis includes data cleaning, model training, evaluation, and comparison to determine the best-performing model.

---

## Dataset
The dataset, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult), contains demographic and employment-related information. The target variable is **income**, classified as:
- `0`: Income ≤ $50,000
- `1`: Income > $50,000

---

## Files in This Repository
- **`Taylor_Ben_HW3.R`**: The R script containing all code for data preprocessing, model training, and evaluation.
- **`README.md`**: This file, describing the project and instructions for replication.

---

## How to Run the Code
1. **Prerequisites**:
   - R (version >= 4.0.0)
   - RStudio (optional)
   - Required R packages:
     - `caret`
     - `glmnet`
     - `randomForest`

   Install packages using:
   ```R
   install.packages(c("caret", "glmnet", "randomForest"))
