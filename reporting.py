from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_evaluation_metrics(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Test MAE: {mae}")
    print(f"Test MSE: {mse}")
    print(f"Test R²: {r2}")
