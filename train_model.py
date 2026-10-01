import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os
# pyrefly: ignore [missing-import]
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = 'electricity_demand_forecasting_1500.csv'

def generate_mock_data(filename=DATA_PATH, n_rows=1500):
    """Generates mock dataset if not already present."""
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return pd.read_csv(filename)
    
    print(f"Generating mock data: {filename}")
    np.random.seed(42)
    data = {
        'Temperature': np.random.uniform(-5, 35, n_rows),
        'Humidity': np.random.uniform(20, 100, n_rows),
        'Wind Speed': np.random.uniform(0, 25, n_rows),
        'Rainfall': np.random.uniform(0, 50, n_rows),
        'Day Of Week': np.random.randint(0, 7, n_rows),
        'Month': np.random.randint(1, 13, n_rows),
        'Holiday': np.random.choice([0, 1], p=[0.95, 0.05], size=n_rows)
    }
    df = pd.DataFrame(data)
    df['Weekend'] = df['Day Of Week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Target formulation
    df['Electricity_Demand_MWh'] = (
        1000 
        - 15 * df['Temperature']
        + 5 * df['Humidity']
        + 2 * df['Wind Speed']
        - 10 * df['Rainfall']
        - 100 * df['Weekend']
        - 200 * df['Holiday']
        + 30 * df['Month']
        + np.random.normal(0, 100, n_rows)
    )
    
    df.to_csv(filename, index=False)
    return df

def train_and_evaluate():
    """Trains multiple models and saves the best one."""
    df = generate_mock_data()
    
    X = df.drop(columns=['Electricity_Demand_MWh'])
    y = df['Electricity_Demand_MWh']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_model_name = ""
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        print(f"{name} - R2: {r2:.4f}, MAE: {mae:.2f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with R2: {best_r2:.4f}")
    
    # Model persistence using joblib
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved model.pkl and scaler.pkl successfully.")
    
    return results

if __name__ == "__main__":
    train_and_evaluate()
