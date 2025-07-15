import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 从csv文件加载数据集
pokemon_data = pd.read_csv('Pokemon.csv')

# 数据预处理
# 处理缺失值
pokemon_data.fillna(pokemon_data.mean(), inplace=True)

# 编码分类变量
label_encoders = {}
categorical_features = ['Type 1', 'Type 2']
for feature in categorical_features:
    le = LabelEncoder()
    pokemon_data[feature] = le.fit_transform(pokemon_data[feature])
    label_encoders[feature] = le

# 分离特征和目标变量
X = pokemon_data.drop(columns=['Name', 'Legendary'])
y = pokemon_data['Legendary']

# 标准化数值特征
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 保存预处理后的数据
pokemon_data_preprocessed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
pokemon_data_preprocessed.to_csv('pokemon_preprocessed.csv', index=False)

print("数据预处理完成并保存到'pokemon_preprocessed.csv'")
