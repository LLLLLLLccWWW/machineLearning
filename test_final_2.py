import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """訓練模型並評估其性能。"""
    clf = model.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_metrics = {
        'Accuracy': accuracy_score(y_train, y_pred_train),
        'Precision': precision_score(y_train, y_pred_train, average='macro'),
        'Recall': recall_score(y_train, y_pred_train, average='macro'),
        'F1 Score': f1_score(y_train, y_pred_train, average='macro'),
        'AUC': roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr') if hasattr(clf, 'predict_proba') else 'N/A'
    }

    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'Precision': precision_score(y_test, y_pred_test, average='macro'),
        'Recall': recall_score(y_test, y_pred_test, average='macro'),
        'F1 Score': f1_score(y_test, y_pred_test, average='macro'),
        'AUC': roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr') if hasattr(clf, 'predict_proba') else 'N/A'
    }

    return train_metrics, test_metrics, y_pred_train, y_pred_test, clf

def save_results_to_excel(results, y_train, y_test, filename):
    """將結果保存到Excel文件中。"""
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for name, metrics in results.items():
            train_metrics = metrics['Train']
            test_metrics = metrics['Test']
            y_pred_train = metrics['y_pred_train']
            y_pred_test = metrics['y_pred_test']
            
            # 保存訓練集指標
            df_train_metrics = pd.DataFrame.from_dict(train_metrics, orient='index', columns=[name])
            df_train_metrics.to_excel(writer, sheet_name=safe_sheet_name(name, '_Train_Metrics'))
            
            # 保存測試集指標
            df_test_metrics = pd.DataFrame.from_dict(test_metrics, orient='index', columns=[name])
            df_test_metrics.to_excel(writer, sheet_name=safe_sheet_name(name, '_Test_Metrics'))
            
            # 保存訓練集混淆矩陣
            conf_matrix_train = confusion_matrix(y_train, y_pred_train)
            index_labels_train = [f'Actual:{i}' for i in range(conf_matrix_train.shape[0])]
            column_labels_train = [f'Predicted:{i}' for i in range(conf_matrix_train.shape[1])]
            df_conf_matrix_train = pd.DataFrame(conf_matrix_train, index=index_labels_train, columns=column_labels_train)
            df_conf_matrix_train.to_excel(writer, sheet_name=safe_sheet_name(name, '_Train_ConfMatrix'))
            
            # 保存測試集混淆矩陣
            conf_matrix_test = confusion_matrix(y_test, y_pred_test)
            index_labels_test = [f'Actual:{i}' for i in range(conf_matrix_test.shape[0])]
            column_labels_test = [f'Predicted:{i}' for i in range(conf_matrix_test.shape[1])]
            df_conf_matrix_test = pd.DataFrame(conf_matrix_test, index=index_labels_test, columns=column_labels_test)
            df_conf_matrix_test.to_excel(writer, sheet_name=safe_sheet_name(name, '_Test_ConfMatrix'))

def safe_sheet_name(name, postfix):
    """確保工作表名稱在Excel的字符限制之內。"""
    max_length = 31 - len(postfix)
    return f"{name[:max_length]}{postfix}"

data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# 確認數據的結構和內容
print(data.head())
print(data.info())

categorical_columns = data.select_dtypes(include=['object']).drop(columns=['NObeyesdad']).columns
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_columns)], remainder='passthrough')

X = preprocessor.fit_transform(data.drop(columns=['NObeyesdad','Height', 'Weight']))
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['NObeyesdad'])
label_names = label_encoder.classes_  # 儲存原始標籤名稱

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LR': LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000),
    'DT': DecisionTreeClassifier(max_depth=100, min_samples_split=10, min_samples_leaf=5),
    'SVM': SVC(C=1.0, kernel='rbf', probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RF': RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_split=10, min_samples_leaf=5),
    'XGB': XGBClassifier(eval_metric='mlogloss',n_estimators=100, max_depth=10, learning_rate=0.1, subsample=0.8),
}

results = {}

for name, model in models.items():
    train_metrics, test_metrics, y_pred_train, y_pred_test, clf = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    results[name] = {'Train': train_metrics, 'Test': test_metrics, 'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test}
    
    print(f"\n{name} Model Evaluation on Training Set:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value}")
    print("Confusion Matrix on Training Set:")
    print(confusion_matrix(y_train, y_pred_train))

    print(f"\n{name} Model Evaluation on Test Set:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")
    print("Confusion Matrix on Test Set:")
    print(confusion_matrix(y_test, y_pred_test))

    # Plot ROC curve
    if hasattr(clf, 'predict_proba'):
        y_probas = clf.predict_proba(X_test_scaled)
        fpr, tpr, _ = roc_curve(y_test, y_probas[:, 1], pos_label=1)
        plt.plot(fpr, tpr, lw=2, alpha=0.8, label=f'{name} (AUC = {test_metrics["AUC"]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# 保存結果到Excel
save_results_to_excel(results, y_train, y_test, 'final_2_test.xlsx')


# 新增交叉驗證結果
cross_val_results = {}

# 定義交叉驗證策略
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    accuracies = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precisions = cross_val_score(model, X, y, cv=cv, scoring='precision_macro')
    recalls = cross_val_score(model, X, y, cv=cv, scoring='recall_macro')
    f1s = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc_ovr')

    cross_val_results[name] = {
        'Accuracy': accuracies.mean(),
        'Precision': precisions.mean(),
        'Recall': recalls.mean(),
        'F1 Score': f1s.mean(),
        'AUC': aucs.mean()
    }

# 保存交叉驗證結果到Excel
cross_val_results_df = pd.DataFrame(cross_val_results).T
cross_val_results_df.to_excel('cross_validation_results.xlsx')
