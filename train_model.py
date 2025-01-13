from statistics import linear_regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

df = pd.read_csv('Datasets/State_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')

state_column = 'RegionName'
date_columns = df.columns[5:]

df_long = df.melt(
    id_vars=[state_column],
    value_vars=date_columns,
    var_name="Date",
    value_name="HomeValueIndex"
)

df_long["Date"] = pd.to_datetime(df_long["Date"])

df_long[state_column] = df_long[state_column].astype("category").cat.codes

X = df_long[[state_column, "Date"]]
Y = df_long["HomeValueIndex"]

df_long = df_long.copy()
X["Date"] = (df_long["Date"] - df_long["Date"].min()).dt.days
print(X)

X = X[Y.notna()]
Y = Y.dropna()
Y.fillna(0, inplace=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
rmse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(rmse)
print(f"Model RMSE: {rmse}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
print("Model coefficients:", model.coef_)
print("Number of features expected:", model.n_features_in_)
