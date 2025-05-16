# APS Failure Prediction Project

## Overview

This project focuses on predicting Air Pressure System (APS) failures in Scania Trucks using the APS Failure dataset provided by Scania via the UCI Machine Learning Repository.

## Dataset

* **Source:** [https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip](https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip)
* **Files Provided:**

  * `aps_failure_training_set.csv`
  * `aps_failure_test_set.csv`

## Objective

The objective is to build a predictive model that minimizes the operational cost defined by the following misclassification cost matrix:

* **False Positive (Type 1 Error):** 10 units
* **False Negative (Type 2 Error):** 500 units

## Summary of Results

Logistic Regression (Class Weighted)

 - Accuracy: 98%
 - Recall (Failure Class): 91%
 - Precision (Failure Class): 51%
 - Total Cost: 19,240
      - Type 1 Cost: 3,240
      - Type 2 Cost: 16,000

Random Forest (Class Weighted)

 - Accuracy: 99%
 - Recall (Failure Class): 58%
 - Precision (Failure Class): 94%
 - Total Cost: 78,640
      - Type 1 Cost: 140
      - Type 2 Cost: 78,500

## Analysis

Logistic Regression demonstrates superior cost performance by achieving higher recall on failure detection. This is critical because Type 2 errors are 500 times more costly than Type 1 errors. While Logistic Regression produces more false alarms, the lower total cost makes it a more effective solution in this cost-sensitive context.

Random Forest achieves higher precision but significantly lower recall. This results in a much higher total cost due to its tendency to miss more true failures, which is heavily penalized in the cost function.

## How to Run

Open `aps_failure_analysis.ipynb` and execute the cells step by step.

## Key Findings

* **Logistic Regression** achieved lower total cost (19,240 units) with higher recall (91%) for failure detection.
* **Random Forest** achieved higher precision but a much higher total cost (78,640 units) due to more missed failures.

## Further Imporvements

* Threshold Tuning: Adjust the decision threshold to further reduce Type 2 errors.
* Ensemble Strategies: Combine Logistic Regression and Random Forest to balance recall and precision.
* Feature Engineering: Explore aggregating or transforming histogram features described in the dataset documentation.
* Cost-Sensitive Algorithms: Investigate models like XGBoost with custom cost-sensitive objectives.

