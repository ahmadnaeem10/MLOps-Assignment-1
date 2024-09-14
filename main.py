import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('train.csv')

# Fill missing values with the mode
mode_values = data.mode().iloc[0]
data_cleaned = data.fillna(mode_values)

# Drop unnecessary columns
columns_to_drop = ['Id', 'Alley', 'MiscVal', 'Fence', 'MiscFeature']
data_cleaned.drop(columns=columns_to_drop, inplace=True)

# Label encoding for categorical features
encoder = LabelEncoder()
for column in data_cleaned.select_dtypes(include=['object']).columns:
    data_cleaned[column] = encoder.fit_transform(data_cleaned[column])

# Calculate the correlation matrix and find columns with a low correlation with 'SalePrice'
corr_matrix = data_cleaned.corr()
low_corr_columns = corr_matrix.index[abs(corr_matrix["SalePrice"]) < 0.6]

# Drop these columns from the DataFrame
X = data_cleaned.drop(columns=low_corr_columns.union(['SalePrice']))
y = data_cleaned['SalePrice']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the test results
y_pred = model.predict(X_test)

# Displaying regression metrics
print("Accuracy Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Save the trained model and scaler as .pkl files
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved to disk")
