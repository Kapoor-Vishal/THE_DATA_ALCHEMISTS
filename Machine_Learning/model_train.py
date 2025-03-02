import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import time

# ✅ Load Data
df = pd.read_csv("../data/processed/cleaned_data.csv")

# ✅ Selecting Optimized Features
features = ["Revenue (USD)", "Operating Cost (USD)", "Load Factor (%)"]
X = df[features]
y = df["Profit (USD)"]

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ✅ Define Optimized Models
models = {
    "Linear Regression": LinearRegression(n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostRegressor(iterations=50, learning_rate=0.1, depth=5, random_state=42, verbose=0)
}

# ✅ Train and Evaluate Models
model_results = []
for name, model in models.items():
    start_time = time.time()
    
    model.fit(X_train, y_train)  # Train model

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    train_time = time.time() - start_time

    model_results.append((name, model, r2, mse, mae, train_time))

    print(f"🔹 {name}:")
    print(f"   - R² Score: {r2:.4f}")
    print(f"   - MSE: {mse:.4f}")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - Training Time: {train_time:.2f} sec\n")

# ✅ Select Best Model (Highest R² Score)
best_model_name, best_model, best_r2, _, _, best_time = max(model_results, key=lambda x: x[2])

# ✅ Save Best Model
joblib.dump(best_model, "best_model.pkl")

print(f"✅ Best Model: {best_model_name} with R²={best_r2:.4f} (Trained in {best_time:.2f} sec)")
