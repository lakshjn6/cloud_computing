"""
Phase 2 — Model Training & Evaluation
=======================================
Trains three forecasting models on the preprocessed Bitbrains workload data:
  1. ARIMA  — statistical baseline (statsmodels)
  2. XGBoost — gradient-boosted trees on lag/rolling features (xgboost)
  3. LSTM   — stacked recurrent neural network (TensorFlow / Keras)

Evaluates all three on the held-out test set using RMSE, MAE, and MAPE,
selects the best model, persists artifacts, and saves a comparison chart.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TensorFlow C++ info logs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_prep import WorkloadDataPreprocessor


# -----------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------
def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Returns a dict with RMSE, MAE, and MAPE for a single model's predictions.

    MAPE uses a small epsilon guard to avoid zero-division when actual values
    are very close to zero (can happen with normalised memory metrics).
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    epsilon = 1e-6

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0)

    return {
        "Model"   : model_name,
        "RMSE"    : round(rmse, 4),
        "MAE"     : round(mae,  4),
        "MAPE (%)": round(mape, 4),
    }


# -----------------------------------------------------------------------
# Model 1 — ARIMA baseline
# -----------------------------------------------------------------------
def train_arima_model(train_series, test_series, order=(2, 1, 1)):
    """
    Fits an ARIMA(p,d,q) model on the training series and generates a
    multi-step forecast equal in length to test_series.

    Falls back to a 12-step rolling average if ARIMA convergence fails.
    """
    print("\n[ARIMA] Training baseline ARIMA model ...")
    t0 = time.time()
    try:
        model      = ARIMA(train_series, order=order)
        fitted     = model.fit()
        forecast   = fitted.forecast(steps=len(test_series))
        elapsed    = time.time() - t0
        print(f"[ARIMA] Done in {elapsed:.1f}s. AIC={fitted.aic:.2f}")
        return fitted, np.array(forecast)
    except Exception as exc:
        print(f"[ARIMA] Fitting failed ({exc}). Using rolling-average fallback.")
        history  = list(train_series)
        forecast = []
        for _ in range(len(test_series)):
            pred = float(np.mean(history[-12:]))
            forecast.append(pred)
            history.append(pred)
        return None, np.array(forecast)


# -----------------------------------------------------------------------
# Model 2 — XGBoost
# -----------------------------------------------------------------------
def train_xgboost_model(X_train, y_train, X_test):
    """
    Trains an XGBoost regression model on the lag / rolling-statistics
    feature matrix produced by WorkloadDataPreprocessor.
    """
    print("\n[XGBoost] Training XGBoost Regressor ...")
    t0 = time.time()

    model = xgb.XGBRegressor(
        n_estimators    = 200,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        min_child_weight= 3,
        random_state    = 42,
        n_jobs          = -1,
    )
    model.fit(X_train, y_train, verbose=False)
    preds   = model.predict(X_test)
    elapsed = time.time() - t0
    print(f"[XGBoost] Done in {elapsed:.1f}s.")
    return model, preds


# -----------------------------------------------------------------------
# Model 3 — LSTM
# -----------------------------------------------------------------------
def train_lstm_model(
    X_train, y_train, X_test,
    seq_length, num_features,
    epochs=30, batch_size=32
):
    """
    Builds and trains a two-layer stacked LSTM using TensorFlow/Keras.
    EarlyStopping monitors val_loss to prevent overfitting.
    Returns the trained model and raw (scaled) predictions on X_test.
    """
    print("\n[LSTM] Training stacked LSTM neural network ...")
    t0 = time.time()
    tf.random.set_seed(42)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, num_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    preds_scaled = model.predict(X_test, verbose=0)
    elapsed = time.time() - t0
    print(f"[LSTM] Done in {elapsed:.1f}s.")
    return model, preds_scaled


# -----------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------
def plot_comparison(
    y_actual, predictions_dict,
    output_path="model_performance_comparison.png",
    display_steps=200
):
    """
    Saves a matplotlib figure comparing actual vs all three model forecasts.
    Only the last `display_steps` time steps are plotted for clarity.
    """
    n = min(display_steps, len(y_actual))
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(y_actual[-n:], label="Actual", color="black", lw=2.0, ls="--")

    colors = {"ARIMA": "#ff7f0e", "XGBoost": "#2ca02c", "LSTM": "#1f77b4"}
    for name, preds in predictions_dict.items():
        ax.plot(
            preds[-n:],
            label=f"{name} Forecast",
            color=colors.get(name, "purple"),
            alpha=0.85, lw=1.8
        )

    ax.set_title(
        "VM Resource Forecasting — Actual vs Model Predictions",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Time Steps (5-min intervals)")
    ax.set_ylabel("CPU Utilisation (%)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Comparison chart saved to '{output_path}'.")


# -----------------------------------------------------------------------
# Main training pipeline
# -----------------------------------------------------------------------
def run_training_pipeline(data_file="processed_data.joblib"):
    """
    Orchestrates all three models:
      1. Loads (or generates) the preprocessed data bundle
      2. Trains ARIMA, XGBoost, LSTM
      3. Aligns prediction lengths (LSTM produces seq_length fewer points)
      4. Computes and prints the evaluation leaderboard
      5. Saves the best model + all artifacts for cloud_sim.py / dashboard.py
    """

    # --- Load preprocessed data ---
    if not os.path.exists(data_file):
        print("[INFO] Preprocessed data not found. Running data_prep.py ...")
        prep = WorkloadDataPreprocessor(filepath="1.csv")
        data = prep.process_and_save(data_file)
    else:
        data = joblib.load(data_file)

    target_scaler    = data["target_scaler"]
    target_col       = data["target_col"]
    seq_length       = data["seq_length"]

    # ---- ARIMA --------------------------------------------------------
    # ARIMA is trained on the raw (unscaled) series.
    # The test forecast covers the full test set, but we must later trim
    # it to match the shorter LSTM test window (which loses seq_length rows).
    arima_train = data["arima_train"]
    arima_test  = data["arima_test"]          # full test set (raw)
    _, arima_preds_full = train_arima_model(arima_train, arima_test)

    # ---- XGBoost ------------------------------------------------------
    X_train_xgb = data["X_train_xgb"]
    y_train_xgb = data["y_train_xgb"]
    X_test_xgb  = data["X_test_xgb"]
    y_test_xgb  = data["y_test_xgb"]         # full test set (raw)
    xgb_model, xgb_preds_full = train_xgboost_model(
        X_train_xgb, y_train_xgb, X_test_xgb
    )

    # ---- LSTM ---------------------------------------------------------
    X_train_lstm = data["X_train_lstm"]
    y_train_lstm = data["y_train_lstm"]
    X_test_lstm  = data["X_test_lstm"]        # already seq_length shorter
    y_test_lstm  = data["y_test_lstm"]
    num_features = X_train_lstm.shape[2]

    lstm_model, lstm_preds_scaled = train_lstm_model(
        X_train_lstm, y_train_lstm, X_test_lstm, seq_length, num_features
    )

    # Inverse-transform LSTM predictions back to the original % scale
    lstm_preds_full = target_scaler.inverse_transform(
        lstm_preds_scaled
    ).flatten()

    # ---- Align all predictions to the LSTM test window ---------------
    # LSTM test window starts seq_length steps into the test set.
    # We align ARIMA and XGBoost to the same window for a fair comparison.
    n_lstm = len(lstm_preds_full)

    arima_preds = arima_preds_full[-n_lstm:]
    xgb_preds   = xgb_preds_full[-n_lstm:]
    lstm_preds  = lstm_preds_full
    y_actual    = y_test_xgb[-n_lstm:]       # ground truth on same window

    # Safety clip: ensure all arrays are exactly the same length
    min_len     = min(len(y_actual), len(arima_preds), len(xgb_preds), len(lstm_preds))
    y_actual    = y_actual[:min_len]
    arima_preds = arima_preds[:min_len]
    xgb_preds   = xgb_preds[:min_len]
    lstm_preds  = lstm_preds[:min_len]

    # ---- Metrics leaderboard -----------------------------------------
    rows = [
        compute_metrics(y_actual, arima_preds, "ARIMA Baseline"),
        compute_metrics(y_actual, xgb_preds,   "XGBoost Regressor"),
        compute_metrics(y_actual, lstm_preds,   "LSTM Neural Net"),
    ]
    metrics_df = pd.DataFrame(rows)

    print("\n" + "=" * 56)
    print("  MODEL EVALUATION LEADERBOARD")
    print("=" * 56)
    print(metrics_df.to_string(index=False))
    print("=" * 56)

    # ---- Select best model by lowest RMSE ----------------------------
    best_row  = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    best_name = best_row["Model"]
    print(f"\n[INFO] Best model: {best_name}  (RMSE={best_row['RMSE']})\n")

    # ---- Save comparison chart ---------------------------------------
    preds_dict = {
        "ARIMA"  : arima_preds,
        "XGBoost": xgb_preds,
        "LSTM"   : lstm_preds,
    }
    plot_comparison(y_actual, preds_dict)

    # ---- Persist artifacts -------------------------------------------
    artifact_bundle = {
        "best_model_name" : best_name,
        "xgb_model"       : xgb_model,
        "target_scaler"   : target_scaler,
        "feature_scaler"  : data["feature_scaler"],
        "xgb_feature_cols": data["xgb_feature_cols"],
        "metrics"         : metrics_df,
        "y_actual"        : y_actual,
        "predictions"     : preds_dict,
        "seq_length"      : seq_length,
        "target_col"      : target_col,
    }
    joblib.dump(artifact_bundle, "model_artifacts.joblib")
    print("[INFO] Model artifacts saved to 'model_artifacts.joblib'.")

    # Save the LSTM model separately (Keras format)
    lstm_model.save("lstm_model.keras")
    print("[INFO] LSTM model saved to 'lstm_model.keras'.")

    # Also save the XGBoost model separately
    joblib.dump(xgb_model, "best_model.pkl")
    print("[INFO] XGBoost model saved to 'best_model.pkl'.")

    return metrics_df, preds_dict


# -----------------------------------------------------------------------
if __name__ == "__main__":
    run_training_pipeline()
