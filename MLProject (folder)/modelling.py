import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- KONFIGURASI DAGSHUB ---
USER_NAME = "zenyfor28"
REPO_NAME = "Workflow-CI"
TOKEN = os.getenv('MLFLOW_TRACKING_PASSWORD') 

# Set up tracking URI secara eksplisit
mlflow.set_tracking_uri(f"https://dagshub.com/{USER_NAME}/{REPO_NAME}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = USER_NAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN

# Set experiment (Pastikan nama ini sudah ada di DagsHub atau gunakan default)
mlflow.set_experiment("Retail_Final_Project")

# Gunakan autolog agar semua artefak tersimpan otomatis
mlflow.sklearn.autolog(log_models=True)

def train_model():
    # Load dataset
    df = pd.read_csv('OnlineRetail_preprocessed.csv')
    
    # Filter hanya numerik (Fix error ValueError)
    df_numeric = df.select_dtypes(include=[np.number])
    
    X = df_numeric.drop("TotalPrice", axis=1)
    y = df_numeric["TotalPrice"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Build_Docker_Run"):
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan pkl manual untuk artifact GitHub
        joblib.dump(model, "model_manual.pkl")
        print("Training Berhasil!")

if __name__ == "__main__":
    train_model()
