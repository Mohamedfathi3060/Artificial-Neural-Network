import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data from the Excel file
data_file = "concrete_data.xlsx"
df = pd.read_excel(data_file)

# Step 2: Separate features and targets
# Features are the first 4 columns, targets are the last column
features = df.iloc[:, :4].values
targets = df.iloc[:, -1].values

# Step 3 (Optional): Normalize the features using Standard Scaler
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Step 4: Split the data into training and testing sets
# 75% for training, 25% for testing
X_train, X_test, y_train, y_test = train_test_split(
    normalized_features, targets, test_size=0.25, random_state=42
)
