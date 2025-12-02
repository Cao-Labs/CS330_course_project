from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET = "response_rt" 


def prep_xy(df: pd.DataFrame, target_col: str):
    """
    Drop rows without target, remove ID-ish columns, one-hot encode categoricals.
    Returns X (DataFrame), y (Series).
    """
    print(f"\n[prep_xy] Starting with {len(df)} rows")
    if target_col not in df.columns:
        print(f"[ERROR] Target column '{target_col}' not found. Available columns:")
        print(df.columns.tolist())
        sys.exit(1)

    df = df.copy()

    df = df.dropna(subset=[target_col])
    print(f"[prep_xy] After dropna on {target_col}: {len(df)} rows")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    print(f"[prep_xy] After forcing numeric {target_col}: {len(df)} rows")

    if len(df) == 0:
        print("[ERROR] After cleaning, there are 0 rows left. "
              "Check that your CSV actually has non-missing numeric values for the target.")
        sys.exit(1)

    y = df[target_col]

    drop_cols = ["participant_id", "Trial", "expt_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    X = df.drop(columns=[target_col], errors="ignore")

    X = pd.get_dummies(X, drop_first=True)

    print(f"[prep_xy] Final X shape: {X.shape}, y length: {len(y)}")
    return X, y


def align_cols(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Make sure test has the same columns as train (for one-hot encoding).
    """
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def main():
    base = Path(__file__).resolve().parent
    train_path = base / "train.csv"
    val_path   = base / "val.csv"
    test_path  = base / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"[ERROR] Expected file not found: {p}")
            sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    print(f"[INFO] Loaded train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")

    train_full = pd.concat([train_df, val_df], ignore_index=True)
    print(f"[INFO] Combined train+val: {train_full.shape}")

    X_train, y_train = prep_xy(train_full, TARGET)
    X_test,  y_test  = prep_xy(test_df, TARGET)

    X_train, X_test = align_cols(X_train, X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\n[INFO] Training Deep MLPRegressor (3 hidden layers)...")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), 
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
    )
    model.fit(X_train_sc, y_train)

    preds = model.predict(X_test_sc)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print("\n=== Deep MLP Results ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    out_path = base / "preds_deep_mlp.csv"
    pd.DataFrame({"y_true": y_test.values, "y_pred": preds}).to_csv(out_path, index=False)
    print(f"\n[INFO] Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
