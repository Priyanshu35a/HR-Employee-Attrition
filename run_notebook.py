import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

print(" Loading HR Attrition data...")

# Load the dataset
df_raw = pd.read_csv('HR-Employee-Attrition.csv')
print(f" Data loaded: {df_raw.shape}")

# Drop unnecessary columns
df = df_raw.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])

# Prepare target variable
y = (df['Attrition'] == 'Yes').astype(int)
X_raw = df.drop(columns=['Attrition'])

# Encode categorical variables
X_encoded = pd.get_dummies(X_raw, drop_first=True)

print(f" Training Random Forest model with {X_encoded.shape[0]} samples and {X_encoded.shape[1]} features...")

# Train Random Forest with adapted parameters for class imbalance
rf_model = RandomForestClassifier(
    max_depth=10,
    n_estimators=100,
    n_jobs=-1,
    random_state=10,
    class_weight='balanced'
)

rf_model.fit(X_encoded, y)
print(f" Model trained successfully!")

# Save the model
joblib.dump(rf_model, 'attrition_model.pkl')
print(f" Model saved to 'attrition_model.pkl'")

# Verify the file was created
import os
if os.path.exists('attrition_model.pkl'):
    file_size = os.path.getsize('attrition_model.pkl')
    print(f" File verified: {file_size} bytes")
