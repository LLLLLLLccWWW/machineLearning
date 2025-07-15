import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score

# 讀取數據
data_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'  
data = pd.read_csv(data_path)

# 編碼類別型變量
categorical_columns = data.select_dtypes(include=['object']).drop(columns=['NObeyesdad']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'
)

# 預處理特徵和目標變量
X = preprocessor.fit_transform(data.drop(columns=['NObeyesdad']))
y = LabelEncoder().fit_transform(data['NObeyesdad'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=80)

# 初始化並訓練 Logistic Regression 模型
lg_model = LogisticRegression(max_iter=1000, random_state=80)
lg_model.fit(X_train, y_train)
lg_predictions = lg_model.predict(X_test)

# 計算 Logistic Regression 的性能指標
lg_confusion_matrix = confusion_matrix(y_test, lg_predictions)
lg_precision = precision_score(y_test, lg_predictions, average='weighted')
lg_accuracy = accuracy_score(y_test, lg_predictions)
lg_recall = recall_score(y_test, lg_predictions, average='weighted')
lg_f1 = f1_score(y_test, lg_predictions, average='weighted')

# 初始化並訓練 SVM 模型
svm_model = SVC(kernel='linear', random_state=80)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# 計算 SVM 的性能指標
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# 輸出結果
print("Logistic Regression Metrics:")
print("Confusion Matrix:\n", lg_confusion_matrix)
print("Precision: {:.3f}".format(lg_precision))
print("Accuracy: {:.3f}".format(lg_accuracy))
print("Recall: {:.3f}".format(lg_recall))
print("F1 Score: {:.3f}".format(lg_f1))

print("\nSVM Metrics:")
print("Confusion Matrix:\n", svm_confusion_matrix)
print("Precision: {:.3f}".format(svm_precision))
print("Accuracy: {:.3f}".format(svm_accuracy))
print("Recall: {:.3f}".format(svm_recall))
print("F1 Score: {:.3f}".format(svm_f1))
