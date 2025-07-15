import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import os
import matplotlib

matplotlib.rc('font', family='Microsoft JhengHei')

file_path = 'pokemon_preprocessed.csv'
data = pd.read_csv(file_path)

# 移除 '#' 列
data = data.drop(columns=['#'])

# 敘述統計
desc_stats = data.describe()

# 各類別樣本數
category_counts = data['Legendary'].value_counts()

data_numeric = data.select_dtypes(exclude=['bool'])

# 散點圖矩陣
plt.figure(figsize=(12, 10))
scatter_matrix(data_numeric, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.suptitle("散點圖矩陣")
plt.show()

# 相關係數矩陣
correlation_matrix = data.corr()

# 熱圖
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("相關係數矩陣")
plt.show()

output_dir = '.'
desc_stats_path = os.path.join(output_dir, 'pokemon_敘述統計.csv')
category_counts_path = os.path.join(output_dir, 'pokemon_各類樣本數.csv')
correlation_matrix_path = os.path.join(output_dir, 'pokemon_相關係數矩陣.csv')

desc_stats.to_csv(desc_stats_path)
category_counts.to_csv(category_counts_path, header=['Count'])
correlation_matrix.to_csv(correlation_matrix_path)

print(f"敘述統計已存成CSV: {desc_stats_path}")
print(f"各類樣本數已存成CSV: {category_counts_path}")
print(f"相關係數矩陣已存成CSV: {correlation_matrix_path}")
