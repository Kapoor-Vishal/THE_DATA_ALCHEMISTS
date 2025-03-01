import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix

def data_preprocessing(file_path):
    try:
        # ✅ Load Data
        df = pd.read_csv(file_path)
        print(f"✅ Data Loaded Successfully! Shape: {df.shape}")

        # ✅ Drop Missing Values
        df.dropna(inplace=True)

        # ✅ Ensure Target Column Exists
        if 'Profit (USD)' not in df.columns:
            raise ValueError("❌ ERROR: 'Profit (USD)' column is missing in the dataset!")

        # ✅ Separate Features (X) & Target (y)
        X = df.drop(columns=['Profit (USD)'])  # Features
        y = df['Profit (USD)']  # Target Variable

        # ✅ Reduce One-Hot Encoding Memory Usage
        high_cardinality_cols = [col for col in X.columns if X[col].nunique() > 100]  # Identify high-cardinality features
        print(f"⚠️ Dropping High-Cardinality Columns: {high_cardinality_cols}")
        X.drop(columns=high_cardinality_cols, inplace=True)

        # ✅ Apply Label Encoding (Instead of One-Hot for High Cardinality)
        label_enc_cols = ['Airline', 'Route']  # Change based on dataset
        for col in label_enc_cols:
            if col in X.columns:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])

        # ✅ Convert Categorical Variables → One-Hot Encoding (Only for Low Cardinality)
        X = pd.get_dummies(X, drop_first=True)

        # ✅ Convert to Sparse Matrix (Fixes Memory Issue)
        X_sparse = csr_matrix(X.values)

        # ✅ Feature Scaling
        scaler = StandardScaler(with_mean=False)  # `with_mean=False` required for sparse matrix
        X_scaled = scaler.fit_transform(X_sparse)

        # ✅ Feature Reduction (PCA)
        n_features = X_scaled.shape[1]
        n_components = min(50, n_features)  # Reduce to 50 components (changeable)

        if n_features > 50:
            pca = PCA(n_components=n_components)
            X_scaled = pca.fit_transform(X_scaled.toarray())  # Convert back to dense before PCA
            print(f"✅ PCA Applied: Reduced to {n_components} components.")

        print(f"✅ Processed Data: X shape = {X_scaled.shape}, y shape = {y.shape}")
        return X_scaled, y

    except Exception as e:
        print(f"❌ ERROR in data_preprocessing: {e}")
        return None, None  # Return None to prevent script failure
