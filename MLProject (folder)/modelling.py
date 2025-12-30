import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- KONFIGURASI DAGSHUB ---
# Diambil dari GitHub Secrets agar aman
USER_NAME = "zenyfor28" 
REPO_NAME = "Eksperimen_SML_Zeny-Arsya-Fortilla" 
TOKEN = os.getenv('MLFLOW_TRACKING_PASSWORD') 

# Set environment variables sebelum tracking dimulai
os.environ["MLFLOW_TRACKING_USERNAME"] = USER_NAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN

# Hubungkan ke MLflow DagsHub
mlflow.set_tracking_uri(f"https://dagshub.com/{USER_NAME}/{REPO_NAME}.mlflow")
mlflow.set_experiment("Retail_Final_Project")

# Mengaktifkan autolog agar folder 'model' otomatis dibuat
mlflow.sklearn.autolog(log_models=True)

def train_model():
    # Load dataset (Pastikan file CSV ada di folder yang sama)
    try:
        df = pd.read_csv('OnlineRetail_preprocessed.csv')
    except FileNotFoundError:
        df = pd.read_csv('MLProject (folder)/OnlineRetail_preprocessing.csv')
    
    # Filter hanya kolom angka untuk menghindari ValueError
    df_numeric = df.select_dtypes(include=[np.number])
    
    X = df_numeric.drop("TotalPrice", axis=1)
    y = df_numeric["TotalPrice"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Docker_Build_Run"):
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan pkl manual sebagai cadangan artefak
        joblib.dump(model, "model_manual.pkl")
        print("Training Berhasil! Folder 'model' telah dibuat otomatis.")

if __name__ == "__main__":
    train_model()
