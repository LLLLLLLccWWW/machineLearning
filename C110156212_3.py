import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve
import matplotlib.pyplot as plt

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
        metrics['AUC'] = roc_auc_score(y_test, y_probas, multi_class='ovr')
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics['AUC'])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - {}'.format(model.__class__.__name__))
        plt.legend(loc="lower right")
        plt.show()
    else:
        metrics['AUC'] = 'N/A'
    conf_matrix = confusion_matrix(y_test, y_pred)
    return metrics, conf_matrix, y_pred

def safe_sheet_name(name, postfix):
    """確保工作表名稱在Excel的字符限制之內。"""
    max_length = 31 - len(postfix)
    return f"{name[:max_length]}{postfix}"

data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
categorical_columns = data.select_dtypes(include=['object']).drop(columns=['NObeyesdad']).columns
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_columns)], remainder='passthrough')
X = preprocessor.fit_transform(data.drop(columns=['NObeyesdad','Height','Weight']))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['NObeyesdad'])
label_names = label_encoder.classes_  # 儲存原始標籤名稱

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LR': LogisticRegression(max_iter=1000),
    'DT': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(),
    'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=80)
}

results = {}
with pd.ExcelWriter('Model_Evaluation_Results.xlsx', engine='xlsxwriter') as writer:
    for name, model in models.items():
        metrics, conf_matrix, y_pred = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        results[name] = metrics
        print(f"\n{name} Model Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        print("Confusion Matrix:")
        print(conf_matrix)

        sheet_base_name = safe_sheet_name(name, "")
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=[name])
        df_metrics.to_excel(writer, sheet_name=f'{sheet_base_name}_Metrics')

        index_labels = [f'Actual:{i}' for i in range(conf_matrix.shape[0])]
        column_labels = [f'Predicted:{i}' for i in range(conf_matrix.shape[1])]
        df_conf_matrix = pd.DataFrame(conf_matrix, index=index_labels, columns=column_labels)
        df_conf_matrix.to_excel(writer, sheet_name=safe_sheet_name(name, "_ConfMatrix"))

        # 保存預測的label及其對應的名稱
        predicted_labels_with_names = [(pred, label_names[pred]) for pred in y_pred]
        df_labels_names = pd.DataFrame(predicted_labels_with_names, columns=['Label', 'Name'])
        df_labels_names.to_excel(writer, sheet_name=safe_sheet_name(name, "_PredLabels"))

        



