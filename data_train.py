import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump
from data_preprocessing import data_preprocessing

# ✅ Load Processed Data
X, y = data_preprocessing("cleaned_data.csv")

if X is None or y is None:
    raise ValueError("❌ ERROR: Data preprocessing failed! Exiting script.")

# ✅ Split Data into Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data Split: Train size = {X_train.shape}, Test size = {X_test.shape}")

# ✅ Train XGBoost Model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# ✅ Model Evaluation
r2 = model.score(X_test, y_test)
mse = np.mean((model.predict(X_test) - y_test) ** 2)
mae = np.mean(np.abs(model.predict(X_test) - y_test))

# ✅ Cross-Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

# 📊 Print Results
print(f"📊 XGBoost Performance:")
print(f"🔹 R2 Score: {r2:.4f} ✅ (Optimized)")
print(f"🔹 MSE: {mse:.4f}")
print(f"🔹 MAE: {mae:.4f}")
print(f"🔹 Cross-Validation Mean R2: {cv_scores.mean():.4f}")

# ✅ Save Model
dump(model, "models/xgboost_model.joblib")
print("✅ Model Saved Successfully!")
