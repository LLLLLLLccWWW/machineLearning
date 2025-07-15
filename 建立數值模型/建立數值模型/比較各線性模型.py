import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RANSACRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加載數據集
pokemon_data = pd.read_csv('pokemon_preprocessed.csv')

# 資料預處理
features = ['HP', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
X = pokemon_data[features]

# 使用實際的 'Attack' 值進行回歸
y = pokemon_data['Attack']

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分數據集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=1)

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

# 回歸指標的函數
def print_metrics_regression(y_true_train, y_pred_train, y_true_test, y_pred_test):
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    r2_train = r2_score(y_true_train, y_pred_train)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    r2_test = r2_score(y_true_test, y_pred_test)
    return mse_train, r2_train, mse_test, r2_test

# 儲存評估結果的列表
results = []

# 線性回歸
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_train_pred_linear = linear_model.predict(X_train)
y_test_pred_linear = linear_model.predict(X_test)
plot_residuals(y_train, y_train_pred_linear, y_test, y_test_pred_linear, 'Residuals - Linear Regression')
mse_train, r2_train, mse_test, r2_test = print_metrics_regression(y_train, y_train_pred_linear, y_test, y_test_pred_linear)
results.append(['Linear Regression', mse_train, r2_train, mse_test, r2_test])

# RANSAC回歸
ransac_model = RANSACRegressor(estimator=LinearRegression(), random_state=42)
ransac_model.fit(X_train, y_train)
y_train_pred_ransac = ransac_model.predict(X_train)
y_test_pred_ransac = ransac_model.predict(X_test)
plot_residuals(y_train, y_train_pred_ransac, y_test, y_test_pred_ransac, 'Residuals - RANSAC Regression')
mse_train, r2_train, mse_test, r2_test = print_metrics_regression(y_train, y_train_pred_ransac, y_test, y_test_pred_ransac)
results.append(['RANSAC Regression', mse_train, r2_train, mse_test, r2_test])

# Lasso回歸
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_train_pred_lasso = lasso_model.predict(X_train)
y_test_pred_lasso = lasso_model.predict(X_test)
plot_residuals(y_train, y_train_pred_lasso, y_test, y_test_pred_lasso, 'Residuals - Lasso Regression')
mse_train, r2_train, mse_test, r2_test = print_metrics_regression(y_train, y_train_pred_lasso, y_test, y_test_pred_lasso)
results.append(['Lasso Regression', mse_train, r2_train, mse_test, r2_test])

# Ridge回歸
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
y_train_pred_ridge = ridge_model.predict(X_train)
y_test_pred_ridge = ridge_model.predict(X_test)
plot_residuals(y_train, y_train_pred_ridge, y_test, y_test_pred_ridge, 'Residuals - Ridge Regression')
mse_train, r2_train, mse_test, r2_test = print_metrics_regression(y_train, y_train_pred_ridge, y_test, y_test_pred_ridge)
results.append(['Ridge Regression', mse_train, r2_train, mse_test, r2_test])

# ElasticNet回歸
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_model.fit(X_train, y_train)
y_train_pred_elastic = elastic_model.predict(X_train)
y_test_pred_elastic = elastic_model.predict(X_test)
plot_residuals(y_train, y_train_pred_elastic, y_test, y_test_pred_elastic, 'Residuals - ElasticNet Regression')
mse_train, r2_train, mse_test, r2_test = print_metrics_regression(y_train, y_train_pred_elastic, y_test, y_test_pred_elastic)
results.append(['ElasticNet Regression', mse_train, r2_train, mse_test, r2_test])

# 將結果存到CSV
results_df = pd.DataFrame(results, columns=['Model', 'MSE (Train)', 'R^2 (Train)', 'MSE (Test)', 'R^2 (Test)'])
results_df.to_csv('regression_results (Model2).csv', index=False)

print("Regression results have been saved to 'regression_results (Model2).csv'")

