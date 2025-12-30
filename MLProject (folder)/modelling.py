import pandas as pd
import mlflow
import joblib
import argparse
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=5)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv('dataset_preprocessing.csv')
    X = df.select_dtypes(include=['number']).dropna().iloc[:, :-1].head(5000)
    y = df.select_dtypes(include=['number']).dropna().iloc[:, -1].head(5000)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
        model.fit(X, y)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "model_manual.pkl")
