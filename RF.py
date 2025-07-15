import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加載數據集
pokemon_data = pd.read_csv('pokemon_preprocessed.csv')

# 資料預處理
features = ['HP', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
X = pokemon_data[features]
print(f"features:{features}")

# 使用實際的 'Attack' 值進行回歸
y = pokemon_data['Attack']

# 切分數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)

# 標準化數據
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 繪製殘差圖的函數
def plot_residuals(y_true_train, y_pred_train, y_true_test, y_pred_test, title):
    residuals_train = y_true_train - y_pred_train
    residuals_test = y_true_test - y_pred_test
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred_train, residuals_train, color='blue', label='Training data', alpha=0.5)
    plt.scatter(y_pred_test, residuals_test, color='green', label='Test data', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.legend()
    plt.show()

# Print回歸指標的函數
def print_metrics_regression(y_true_train, y_pred_train, y_true_test, y_pred_test):
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    r2_train = r2_score(y_true_train, y_pred_train)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    r2_test = r2_score(y_true_test, y_pred_test)
    print(f'MSE train: {mse_train:.3f}, test: {mse_test:.3f}')
    print(f'R^2 train: {r2_train:.3f}, test: {r2_test:.3f}')

# 隨機森林回歸 1(非線性)
rf_model = RandomForestRegressor(n_estimators=100, random_state=6)
rf_model.fit(X_train, y_train)
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

print("Random Forest Regression 1:")
print("Training and Test Set Metrics:")
plot_residuals(y_train, y_train_pred_rf, y_test, y_test_pred_rf, 'Residuals - Random Forest Regression ')
print_metrics_regression(y_train, y_train_pred_rf, y_test, y_test_pred_rf)

# 隨機森林回歸 2(非線性)
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_leaf=4, random_state=6)
rf_model.fit(X_train, y_train)
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

print("Random Forest Regression 2:")
print("Training and Test Set Metrics:")
plot_residuals(y_train, y_train_pred_rf, y_test, y_test_pred_rf, 'Residuals - Random Forest Regression ')
print_metrics_regression(y_train, y_train_pred_rf, y_test, y_test_pred_rf)

# 隨機森林回歸 3(非線性)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=6)
rf_model.fit(X_train, y_train)
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

print("Random Forest Regression 3:")
print("Training and Test Set Metrics:")
plot_residuals(y_train, y_train_pred_rf, y_test, y_test_pred_rf, 'Residuals - Random Forest Regression ')
print_metrics_regression(y_train, y_train_pred_rf, y_test, y_test_pred_rf)





