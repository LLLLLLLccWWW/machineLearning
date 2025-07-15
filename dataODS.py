import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Check for duplicate rows
duplicates = data.duplicated().sum()
print("Duplicate rows:", duplicates)

# Remove duplicate rows
data_cleaned = data.drop_duplicates()
print("Shape after removing duplicates:", data_cleaned.shape)

# Statistical summary for numerical columns to check for outliers
numerical_summary = data_cleaned.describe()
print("Numerical summary:\n", numerical_summary)

# Check unique values for categorical columns to identify inconsistencies
categorical_summary = data_cleaned.select_dtypes(include=['object']).nunique()
print("Categorical summary:\n", categorical_summary)

# Encode categorical variables
label_encoders = {}
for column in data_cleaned.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data_cleaned[column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le

# Standardize numerical variables
scaler = StandardScaler()
data_cleaned[data_cleaned.select_dtypes(include=['float64']).columns] = scaler.fit_transform(data_cleaned.select_dtypes(include=['float64']))

print("Data after encoding and scaling:\n", data_cleaned.head())

# Save the cleaned and processed data to a new CSV file
output_file_path = 'ObesityDataSet_cleaned.csv'
data_cleaned.to_csv(output_file_path, index=False)
print(f"Cleaned data has been saved to {output_file_path}")
