import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Gunakan TOKEN dari Secrets GitHub agar aman
USER_NAME = "zenyfor28"
REPO_NAME = "Workflow-CI"
TOKEN = os.getenv('MLFLOW_TRACKING_PASSWORD') 

os.environ["MLFLOW_TRACKING_USERNAME"] = USER_NAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN

mlflow.set_tracking_uri(f"https://dagshub.com/{USER_NAME}/{REPO_NAME}.mlflow")
mlflow.set_experiment("Retail_Project_Final")

# KUNCI SUKSES: Autolog otomatis membuat folder 'model'
mlflow.sklearn.autolog(log_models=True)

def train_model():
    # Load dataset (pastikan file ini ada di folder yang sama)
    df = pd.read_csv('OnlineRetail_preprocessed.csv')
    
    # Filter hanya angka (mencegah error string lagi)
    df_numeric = df.select_dtypes(include=[np.number])
    
    X = df_numeric.drop("TotalPrice", axis=1)
    y = df_numeric["TotalPrice"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Run_Build_Docker"):
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan pkl manual untuk artifact GitHub
        joblib.dump(model, "model_manual.pkl")
        
        print("Training Selesai! Folder 'model' otomatis dibuat oleh autolog.")

if __name__ == "__main__":
    train_model()
