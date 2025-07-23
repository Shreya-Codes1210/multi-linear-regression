
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("S07_datasets_13720_18513_insurance.csv")
print(data)

# Separate features and target
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Encode categorical variables
le = preprocessing.LabelEncoder()
X['sex'] = le.fit_transform(X['sex'])
X['smoker'] = le.fit_transform(X['smoker'])
X['region'] = le.fit_transform(X['region'])

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=62)

# Create and train model
model = LinearRegression()
model.fit(X_test, Y_test)

# Predict
Y_pred = model.predict(X_test)

# R² Score
r2 = r2_score(Y_test, Y_pred)
print("R² Score:", r2)
