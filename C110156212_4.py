import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import fontManager
import matplotlib

plt.rcParams['font.family'] = 'MingLiU'
plt.rcParams['axes.unicode_minus'] = False  

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
    if hasattr(clf, 'predict_proba'):
        y_probas = clf.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_probas[:, 1], pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC曲線 (面積 = %0.2f)' % metrics['AUC'])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model.__class__.__name__}')
        plt.legend(loc="lower right")
        plt.show()
    else:
        metrics['AUC'] = 'N/A'
    conf_matrix = confusion_matrix(y_test, y_pred)
    return metrics, conf_matrix, y_pred

def plot_feature_importance(model, feature_names):
    """繪製模型的特徵重要性。"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'estimators_'):
        # 如果是AdaBoost模型，計算基分類器的重要性平均值
        importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    else:
        raise ValueError("error")

    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(10,6))
    plt.title(f'Feature Importance - {model.__class__.__name__}')
    plt.bar(range(len(feature_names)), importance[indices], align='center')
    plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# 載入資料集
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# 處理遺失值
data = data.dropna()

# 處理類別型欄位
categorical_columns = data.select_dtypes(include=['object']).drop(columns=['NObeyesdad']).columns
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_columns)], remainder='passthrough')

# 準備特徵矩陣X和標籤y
X = preprocessor.fit_transform(data.drop(columns=['NObeyesdad', 'Height', 'Weight']))

# 標籤編碼
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['NObeyesdad'])
label_names = label_encoder.classes_

# 確保X和y的樣本數一致
print(f"X shape: {X.shape}, y shape: {len(y)}")

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定義模型
models = {
    'RF': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# 取得特徵名稱
feature_names = preprocessor.get_feature_names_out()

results={}
# 訓練和評估模型
for name, model in models.items():
    print(f"\n訓練 {name}...")
    metrics, conf_matrix, y_pred = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"{name} 指標:\n", metrics)
    print(f"{name} 混淆矩陣:\n", conf_matrix)
    plot_feature_importance(model, feature_names)

    results[name]={
        'Metrics':metrics,
        'Confusion Matrix':conf_matrix
    }

    with pd.ExcelWriter('model_results.xlsx') as writer:
        for name,result in results.items():
            metrics_df=pd.DataFrame([result['Metrics']])
            metrics_df.to_excel(writer,sheet_name=f'{name}_Metrics',index=False)

            conf_matrix_df=pd.DataFrame(result['Confusion Matrix'])
            conf_matrix_df.to_excel(writer,sheet_name=f'{name}_Confusion Matrix',index=False)


