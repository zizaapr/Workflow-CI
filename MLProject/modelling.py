# modelling.py
# ====================================================
# Training Model untuk Prediksi Harga Rumah + Tracking MLflow
# ====================================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ====================================================
# 1️. Load & Persiapan Data
# ====================================================
def load_and_prepare_data(base_path: str):

    # File dataset tunggal
    data_path = os.path.join(base_path, "boston_data_preprocessed.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {data_path}")

    data = pd.read_csv(data_path)

    # Tentukan nama kolom target (ganti sesuai dataset lu)
    target_column = "MEDV"

    if target_column not in data.columns:
        raise KeyError(f"❌ Kolom target '{target_column}' tidak ditemukan dalam dataset.")

    # Drop baris tanpa target
    data = data.dropna(subset=[target_column])

    # Pisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode kolom kategorikal jika ada
    for col in X.columns:
        if X[col].dtype == "object":
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))

    # Split 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"✅ Data siap digunakan. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ====================================================
# 2️. Training Model + MLflow Tracking
# ====================================================
def train_and_log_model(X_train, X_test, y_train, y_test):

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\n📊 HASIL EVALUASI MODEL")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Log tracking MLflow
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print("\n✅ Model berhasil dicatat di MLflow!")


# ====================================================
# 3️. Entry Point (untuk MLflow CLI / GitHub Actions)
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for house price prediction")
    parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="Folder tempat file boston_data_preprocessed.csv berada",
    )
    args = parser.parse_args()

    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data(args.data_path)
        train_and_log_model(X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"\n❌ Terjadi kesalahan: {e}")
        exit(1)
