import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

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
X = preprocessor.fit_transform(data.drop(columns=['NObeyesdad','Height','Weight']))
y = LabelEncoder().fit_transform(data['NObeyesdad'])
y_bin = label_binarize(y, classes=np.unique(y))  # Binarize labels in a one-vs-all fashion

# 使用隨機森林評估特徵重要性
rf = RandomForestClassifier(random_state=80)
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
cumulative_importances = np.cumsum(importances[indices])
important_features = indices[cumulative_importances <= 0.5]
X_important = X[:, important_features]

# 獲得經過預處理的特徵名稱
feature_names = preprocessor.get_feature_names_out()

# 使用 important_features 索引獲取特徵名稱
important_feature_names = feature_names[important_features]

# 顯示重要特徵名稱
print("Important Features (Top 50%):")
print(important_feature_names)

plt.figure(figsize=(10,6))
plt.title('RF')
plt.bar(range(len(feature_names)),importances,color='b',align='center')
plt.xticks(range(len(feature_names)),feature_names,rotation=90)
plt.tight_layout()
plt.show()

# 定義交叉驗證方法
cv = StratifiedKFold(n_splits=5)

# 定義模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=80),
    'SVM': SVC(kernel='linear', probability=True, random_state=80)
}

# 訓練模型並計算績效
results = {}
for name, model in models.items():
    results[name] = {}
    for feature_set, X_subset in [('All Features', X), ('Important Features', X_important)]:
        # 初始化存儲指標的列表
        metrics = {
            'Confusion Matrix': np.zeros((len(np.unique(y)), len(np.unique(y)))),
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': [],
            'ROC AUC': []
        }

        # For ROC Curve plotting
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for train_idx, test_idx in cv.split(X_subset, y):
            X_train, X_test = X_subset[train_idx], X_subset[test_idx]
            y_train, y_test = y_bin[train_idx], y_bin[test_idx]

            model.fit(X_train, y_train.argmax(axis=1))
            predictions = model.predict(X_test)
            probas = model.predict_proba(X_test)

            # Update confusion matrix
            metrics['Confusion Matrix'] += confusion_matrix(y_test.argmax(axis=1), predictions)

            # Metrics
            metrics['Accuracy'].append(accuracy_score(y_test.argmax(axis=1), predictions))
            metrics['Precision'].append(precision_score(y_test.argmax(axis=1), predictions, average='weighted'))
            metrics['Recall'].append(recall_score(y_test.argmax(axis=1), predictions, average='weighted'))
            metrics['F1 Score'].append(f1_score(y_test.argmax(axis=1), predictions, average='weighted'))

            # Compute ROC curve and area the curve
            for i in range(y_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_test[:, i], probas[:, i])
                interp_tpr = interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Plot ROC Curve for the mean of the folds
        plt.figure()
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f ± %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {name} using {feature_set}')
        plt.legend(loc="lower right")
        plt.show()
        
        # Save averaged results
        results[name][feature_set] = {
            'Confusion Matrix': metrics['Confusion Matrix'] / cv.n_splits,
            'Accuracy': np.mean(metrics['Accuracy']),
            'Precision': np.mean(metrics['Precision']),
            'Recall': np.mean(metrics['Recall']),
            'F1 Score': np.mean(metrics['F1 Score']),
            'ROC AUC': mean_auc
        }

# 輸出結果
for model, features in results.items():
    for feature_set, metrics in features.items():
        print(f"\n{model} Metrics with {feature_set}:")
        print("Confusion Matrix:\n", metrics['Confusion Matrix'])
        print(f"Precision: {metrics['Precision']:.3f}")
        print(f"Accuracy: {metrics['Accuracy']:.3f}")
        print(f"Recall: {metrics['Recall']:.3f}")
        print(f"F1 Score: {metrics['F1 Score']:.3f}")
        print(f"ROC AUC: {metrics['ROC AUC']:.3f}")
        
# 將結果存儲為CSV
data_for_csv = []
for model, feature_sets in results.items():
    for feature_set, metrics in feature_sets.items():
        data_for_csv.append({
            'Model': model,
            'Feature Set': feature_set,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1 Score': metrics['F1 Score'],
            'ROC AUC': metrics['ROC AUC'],
            'Confusion Matrix': str(metrics['Confusion Matrix']),
            'Important Features':', '.join(important_feature_names)
        })

df = pd.DataFrame(data_for_csv)
df.to_csv('model_performance.csv', index=False)
