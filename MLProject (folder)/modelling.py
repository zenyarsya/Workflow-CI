import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Inisialisasi DagsHub untuk MLflow Tracking
# Menggunakan username zenyfor28 dan nama repo GitHub kamu
if os.getenv('MLFLOW_TRACKING_USERNAME'):
    import dagshub
    dagshub.init(
        repo_owner='zenyfor28', 
        repo_name='Workflow-CI',
        setup_mlflow=True
    )

def train_model():
    # Load Dataset
    # Pastikan file OnlineRetail_preprocessed.csv ada di folder 'MLProject (folder)'
    df = pd.read_csv('OnlineRetail_preprocessed.csv')
    
    X = df.drop('TotalPrice', axis=1)
    y = df['TotalPrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow Run
    with mlflow.start_run():
        # Parameter sederhana
        n_estimators = 100
        max_depth = 5
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        # Evaluasi
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Log ke DagsHub via MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        
        # PENTING: Membuat folder 'model' yang berisi file MLmodel untuk Docker
        mlflow.sklearn.log_model(model, "model")
        
        # Simpan file pkl untuk artifact GitHub
        joblib.dump(model, "model_manual.pkl")
        
        print(f"Training selesai. RMSE: {rmse}")

if __name__ == "__main__":
    train_model()
