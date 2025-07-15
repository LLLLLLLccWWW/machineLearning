import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 讀取 CSV 檔案
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

print(data.columns)


# 檢查是否有缺失值，若有則以平均值填補
print('1.檢查是否有缺失值(有的話將以平均值填補缺失值)')
print()
numValue = data.isnull().sum()
if numValue.any() == 0:
    print(data.isnull().sum())
else:
    data.fillna(data.mean(), inplace=True)

print('=' * 160)
# 敘述統計
print('2.敘述統計')
print()
label_column = 'NObeyesdad'
description = data.groupby(label_column).describe()
print(description)

print('=' * 160)

# 特徵處理
categorical_features = data.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(), categorical_features.tolist())],
    remainder='passthrough')
transformed_data = preprocessor.fit_transform(data)

# 特徵縮放
print('3.特徵縮放')
print()
scaler = StandardScaler()
scaler_features = scaler.fit_transform(transformed_data)
print(scaler_features)

print('=' * 160)

# 類別特徵轉換
print('4.類別特徵轉換')
print()
label_encoder = LabelEncoder()
data[label_column] = label_encoder.fit_transform((data[label_column]))
print(data[label_column])

print('=' * 160)
# 訓練與測試 split data
print('5.訓練與測試 split data')
# 特徵選擇
X=data[['Height','Weight']].values
y=data['NObeyesdad']
y_encoded=label_encoder.fit_transform(y)

# X=scaler_features
# y=data[label_column]
# 標準化特徵
sc = StandardScaler()
X_std=scaler.fit_transform(X)

X_train_std,X_test_std,y_train,y_test=train_test_split(X_std,y_encoded,test_size=0.5,random_state=80)

# 使用 Logistic Regression 模型進行分類
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

# def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#     markers = ('s', 'x', 'o', '^', 'v', 'D', 'P')  
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])

#     x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
#     x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
    
#     if X.shape[1] > 2:
#         Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()] + [np.zeros(xx1.ravel().shape[0]) for _ in range(X.shape[1] - 2)]).T)
#     else:
#         Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        
#     Z = Z.reshape(xx1.shape)

#     plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())

#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0],
#                     y=X[y == cl, 1],
#                     alpha=0.6,
#                     c=[cmap(idx)],
#                     edgecolor='black',
#                     marker=markers[idx % len(markers)],  # 使用取模運算確保 markers 不會越界
#                     label=cl)

def plot_decision_regions_fixed(X, y, classifier, resolution=0.02):
    if isinstance(X, pd.DataFrame):
        X = X.values

    markers = ('s', 'x', 'o', '^', 'v', '*', '+')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'yellow', 'black')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)],
                    edgecolor='black',
                    marker=markers[idx], label=f'Class {cl}')

# 繪製決策邊界

# plot_decision_regions(X_train_std, y_train, classifier=lr)
# plt.xlabel('Height [standardized]')
# plt.ylabel('Weight [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# SVM Model

# label_encoders={}
# for column in data.select_dtypes(include=['object']).columns:
#     le=LabelEncoder()
#     data[column]=le.fit_transform(data[column])
#     label_encoders[column]=le

X_train_std = sc.fit_transform(X_train_std)
X_test_std = sc.transform(X_test_std)

# svm=SVC(kernel='rbf',random_state=1,gamma=100.0,C=1.0)
# svm.fit(X_train_std,y_train)
# plot_decision_regions(X_train_std, y_train, classifier=svm)
# plt.xlabel('Height [standardized]')
# plt.ylabel('Weight [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.title('SVM Model')
# plt.show()

# y_pred=svm.predict(X_test_std)

# print('='*160)
# print('6.SVM Model')
# print(classification_report(y_test, y_pred))

# DT Model

# from sklearn.tree import DecisionTreeClassifier

# dt=DecisionTreeClassifier(random_state=1,criterion='gini',max_depth=4)
# dt.fit(X_train_std,y_train)

# X_combined=np.vstack((X_train_std,X_test_std))
# y_combined=np.hstack((y_train,y_test))

# y_pred_dt=dt.predict(X_test_std)

# plot_decision_regions(X_combined, y_combined, classifier=dt)
# plt.xlabel('Height [standardized]')
# plt.ylabel('Weight [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.title('DT Model')
# plt.show()

# print('='*160)
# print('7.DT Model')
# print(classification_report(y_test, y_pred_dt))

# RF_Feature importance

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_std, y_train)
features = rf.feature_importances_
feature_names = preprocessor.get_feature_names_out()

plt.figure(figsize=(10, 8))
sorted_idx = np.argsort(features)[::-1]
plt.barh([feature_names[i] for i in sorted_idx], features[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in RandomForest Model')
plt.gca().invert_yaxis()  
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_std, y_train)

# 繪製隨機森林決策邊界
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_decision_regions_fixed(X_train_std, y_train, classifier=rf)
plt.title('Random Forest Decision Boundary')


# 繪製KNN決策邊界
plt.subplot(1, 2, 2)
plot_decision_regions_fixed(X_train_std, y_train, classifier=knn_model)
plt.title('KNN Decision Boundary')
plt.show()
