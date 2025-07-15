import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.stats import randint

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """訓練模型並評估其性能。"""
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro'),
        'AUC': roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr') if hasattr(clf, 'predict_proba') else 'N/A'
    }
    return metrics, y_pred, clf

data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

categorical_columns = data.select_dtypes(include=['object']).drop(columns=['NObeyesdad']).columns
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_columns)], remainder='passthrough')
X = preprocessor.fit_transform(data.drop(columns=['NObeyesdad','Height', 'Weight']))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['NObeyesdad'])
label_names = label_encoder.classes_

# 調整訓練和測試集的比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練和評估原始隨機森林模型
rf_model = RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_split=10, min_samples_leaf=5)
metrics_rf_before, y_pred_rf_before, clf_rf_before = train_and_evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

# 使用 SMOTE 進行過採樣
smote = SMOTE(random_state=100)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 超參數調整
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(10, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True, False]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=50, cv=cv, n_jobs=-1, 
                                   verbose=2, scoring='roc_auc', random_state=42)
random_search.fit(X_train_smote, y_train_smote)

best_rf_model = random_search.best_estimator_

# 使用XGBoost作為第二個模型
xgb_model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=100)
xgb_model.fit(X_train_smote, y_train_smote)

# 訓練和評估混合模型
voting_model = VotingClassifier(estimators=[('rf', best_rf_model), ('xgb', xgb_model)], voting='soft')
metrics_voting, y_pred_voting, clf_voting = train_and_evaluate_model(voting_model, X_train_smote, y_train_smote, X_test_scaled, y_test)

# 打印比較結果
print("\nRandom Forest Model Evaluation Before SMOTE:")
for metric, value in metrics_rf_before.items():
    print(f"{metric}: {value}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf_before))

print("\nRandom Forest Model Evaluation After SMOTE and Hyperparameter Tuning:")
for metric, value in metrics_voting.items():
    print(f"{metric}: {value}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_voting))

# 繪製原始和使用 SMOTE 並調參後的隨機森林模型的 ROC 曲線
plt.figure()
if hasattr(clf_rf_before, 'predict_proba'):
    y_probas_rf_before = clf_rf_before.predict_proba(X_test_scaled)
    fpr_rf_before, tpr_rf_before, _ = roc_curve(y_test, y_probas_rf_before[:, 1], pos_label=1)
    plt.plot(fpr_rf_before, tpr_rf_before, lw=2, alpha=0.8, label=f'RF Before SMOTE (AUC = {metrics_rf_before["AUC"]:.2f})')

if hasattr(clf_voting, 'predict_proba'):
    y_probas_voting = clf_voting.predict_proba(X_test_scaled)
    fpr_voting, tpr_voting, _ = roc_curve(y_test, y_probas_voting[:, 1], pos_label=1)
    plt.plot(fpr_voting, tpr_voting, lw=2, alpha=0.8, label=f'RF After SMOTE and Tuning (AUC = {metrics_voting["AUC"]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for Random Forest Before and After SMOTE and Tuning')
plt.legend(loc='lower right')
plt.show()

# 保存結果到 Excel
with pd.ExcelWriter('final_3_2.xlsx', engine='xlsxwriter') as writer:
    # 保存隨機森林模型在使用 SMOTE 前的結果
    df_rf_before_metrics = pd.DataFrame.from_dict(metrics_rf_before, orient='index', columns=['RF_Before_SMOTE'])
    df_rf_before_metrics.to_excel(writer, sheet_name='RF_Before_SMOTE_Metrics')

    conf_matrix_rf_before = confusion_matrix(y_test, y_pred_rf_before)
    index_labels_rf_before = [f'Actual:{i}' for i in range(conf_matrix_rf_before.shape[0])]
    column_labels_rf_before = [f'Predicted:{i}' for i in range(conf_matrix_rf_before.shape[1])]
    df_conf_matrix_rf_before = pd.DataFrame(conf_matrix_rf_before, index=index_labels_rf_before, columns=column_labels_rf_before)
    df_conf_matrix_rf_before.to_excel(writer, sheet_name='RF_Before_SMOTE_ConfMatrix')

    # 保存混合模型在使用 SMOTE 和調參後的結果
    df_voting_metrics = pd.DataFrame.from_dict(metrics_voting, orient='index', columns=['RF_After_SMOTE_Tuning'])
    df_voting_metrics.to_excel(writer, sheet_name='RF_After_SMOTE_Metrics')

    conf_matrix_voting = confusion_matrix(y_test, y_pred_voting)
    index_labels_voting = [f'Actual:{i}' for i in range(conf_matrix_voting.shape[0])]
    column_labels_voting = [f'Predicted:{i}' for i in range(conf_matrix_voting.shape[1])]
    df_conf_matrix_voting = pd.DataFrame(conf_matrix_voting, index=index_labels_voting, columns=column_labels_voting)
    df_conf_matrix_voting.to_excel(writer, sheet_name='RF_After_SMOTE_ConfMatrix')