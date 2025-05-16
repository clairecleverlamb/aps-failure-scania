import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load datasets
train_df = pd.read_csv('aps_failure_training_set.csv', skiprows=20)
test_df = pd.read_csv('aps_failure_test_set.csv', skiprows=20)

# Data cleaning
train_df.replace('na', np.nan, inplace=True)
test_df.replace('na', np.nan, inplace=True)

numeric_cols = train_df.columns[1:]
train_df[numeric_cols] = train_df[numeric_cols].astype(float)
test_df[numeric_cols] = test_df[numeric_cols].astype(float)

# Impute missing values with median
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(train_df.median(numeric_only=True), inplace=True) 

# Encode labels
train_df['class'] = train_df['class'].map({'neg': 0, 'pos': 1})
test_df['class'] = test_df['class'].map({'neg': 0, 'pos': 1})

# Features and labels
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']
X_test = test_df.drop('class', axis=1)
y_test = test_df['class']

print(f"Features and labels defined. Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}\n")

# Model definition - logistic regression with class weights
model = LogisticRegression(class_weight={0: 1, 1: 50}, max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluation
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# custom cost calculation
cost_1 = 10  
cost_2 = 500  
type_1_errors = cm[0, 1]
type_2_errors = cm[1, 0]
total_cost = cost_1 * type_1_errors + cost_2 * type_2_errors
# Print cost analysis
print(f"\nType 1 Errors (False Positives): {type_1_errors} * {cost_1} = {type_1_errors * cost_1}")
print(f"Type 2 Errors (False Negatives): {type_2_errors} * {cost_2} = {type_2_errors * cost_2}")
print(f"Total Cost: {total_cost}")

# Evaluation: random forest
rf_model = RandomForestClassifier(class_weight={0: 1, 1: 50}, n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Confusion Matrix:\n", cm_rf)
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# custom cost calculation for random forest
type_1_errors_rf = cm_rf[0, 1]
type_2_errors_rf = cm_rf[1, 0]
total_cost_rf = cost_1 * type_1_errors_rf + cost_2 * type_2_errors_rf
print(f"\nRandom Forest Total Cost: {total_cost_rf}")

