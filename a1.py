import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



ds = pd.read_csv("jamboree.csv")

if "Serial No." in ds.columns:
    ds = ds.drop(columns=["Serial No."])

X = ds.drop(columns=["Chance of Admit "])
y = ds["Chance of Admit "]


# X_train = input data to train the model
# y_train = target values for training
# X_test = the input data reserved for testing (20%)
# y_test = target values for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# predictions
res = model.predict(X_test)


# Print result
print("LINEAR REGRESSION: ")
print("Mean Squared Error (Linear Regression): ", mean_squared_error(y_test, res))
print("R^2 Score (Linear Regression): ", r2_score(y_test, res))


# Sample prediction (GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research )
sample = {
    "GRE Score": 350,
    "TOEFL Score": 110,
    "University Rating": 5,
    "SOP": 4.0,
    "LOR ": 3.0,
    "CGPA": 8.87,
    "Research": 0
}

sample_df = pd.DataFrame([sample])[X.columns] 
predict_sample = model.predict(sample_df)
print("Predicted Chance of Admit (Linear Regression): ", predict_sample[0])

# Model 2 = Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Train the Random Forest model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test set
rf_pred = rf.predict(X_test)

# Find the Mean Squared Error:average squared difference between actual and predicted values
#  Find the R^2 score: how well the model explains variance (closer to 1 is better)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Make predictsion on the sample input
predict_sample_rf = rf.predict(sample_df)[0]

print()
print("RANDOM FOREST:")
print("Mean Squared Error (Random Forest): ", rf_mse)
print("R^2 Score (Random Forest): ", rf_r2)
print("Predicted Chance of Admit (Random Forest Regression): ", predict_sample_rf)

