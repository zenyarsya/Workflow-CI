import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import json
from sklearn.ensemble import RandomForestRegressor # Pakai Regressor untuk TotalPrice
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Gunakan path relatif agar aman di GitHub Runner
csv_path = 'OnlineRetail_preprocessing.csv' 
if not os.path.exists(csv_path):
    # Jika tidak ada di root, cari di folder MLProject
    csv_path = 'MLProject (folder)/OnlineRetail_preprocessing.csv'

# Load Data
df = pd.read_csv(csv_path)
# Pilih hanya kolom numerik seperti kode awalmu
df_numeric = df.select_dtypes(include=['number'])
X = df_numeric.drop('TotalPrice', axis=1)
y = df_numeric['TotalPrice']

# Split Data 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)

# Logging (Tanpa start_run karena dijalankan lewat 'mlflow run')
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 5)
mlflow.log_metric("mse", mse)

# Simpan Model sebagai Artefak MLflow
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="Retail_RF_Model" 
)

# Simpan pkl manual (untuk build Docker nantinya)
joblib.dump(model, "model.pkl")
mlflow.log_artifact("model.pkl")

print("Training Selesai dan Log berhasil dikirim ke DagsHub!")
