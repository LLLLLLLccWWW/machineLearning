import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'ObesityDataSet_preprocessed.csv'
data = pd.read_csv(file_path)

# Descriptive statistics
desc_stats = data.describe()

# Sample size for each category in 'NObeyesdad'
category_counts = data['NObeyesdad'].value_counts()

print("Descriptive Statistics:")
print(desc_stats)
print("\nSample Size for each category in 'NObeyesdad':")
print(category_counts)


# Correlation matrix
corr_matrix = data.corr()

print("Correlation Matrix:")
print(corr_matrix)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Correlation Matrix")
plt.show()

# Save the correlation matrix to a CSV file
corr_matrix_file_path = 'C:/Users/sliuw/Desktop/機器學習HW/correlation_matrix.csv'
corr_matrix.to_csv(corr_matrix_file_path)

print(f"Correlation matrix saved to {corr_matrix_file_path}")