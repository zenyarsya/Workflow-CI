import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Inisialisasi DagsHub Tracking
if os.getenv('MLFLOW_TRACKING_USERNAME'):
    import dagshub
    dagshub.init(
        repo_owner='zenyfor28', 
        repo_name='Workflow-CI', 
        setup_mlflow=True
    )

def train_model():
    # Load Dataset
    df = pd.read_csv('OnlineRetail_preprocessing.csv')
    
    # SOLUSI ERROR STRING: Hanya ambil kolom angka
    # Ini akan membuang kode produk teks seperti '15056BL'
    df_numeric = df.select_dtypes(include=[np.number])
    
    if 'TotalPrice' not in df_numeric.columns:
        print("Error: Target TotalPrice tidak ditemukan dalam data numerik!")
        return

    X = df_numeric.drop('TotalPrice', axis=1)
    y = df_numeric['TotalPrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        mlflow.log_metric("rmse", rmse)
        
        mlflow.sklearn.log_model(model, "model")
        
        joblib.dump(model, "model_manual.pkl")
        
        print(f"Training selesai. RMSE: {rmse}")

if __name__ == "__main__":
    train_model()
