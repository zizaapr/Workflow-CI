import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================
# 1. Load data hasil preprocessing
# ==========================
df = pd.read_csv('boston_data_preprocessed.csv')

# Pisahkan fitur (X) dan target (y)
target_column = 'MEDV'
X = df.drop(columns=[target_column])
y = df[target_column]

# Bagi data menjadi train dan test (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Jumlah data latih:", X_train.shape[0])
print("Jumlah data uji  :", X_test.shape[0])

# ==========================
# 2. Inisialisasi MLflow
# ==========================
mlflow.set_experiment("Boston_Housing_Price_Tracking")

# Autolog semua parameter, metrik, model, artifact secara otomatis
mlflow.sklearn.autolog()

# ==========================
# 3. Jalankan experiment MLflow
# ==========================
with mlflow.start_run(run_name="RandomForest_Boston_Autolog"):
    
    # Inisialisasi dan latih model Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Prediksi pada test set
    preds = model.predict(X_test)
    
    # Hitung metrik evaluasi
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Tampilkan hasil evaluasi
    print("\n=== Evaluation Metrics ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

mlflow.end_run()