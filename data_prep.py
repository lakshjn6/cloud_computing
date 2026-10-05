"""
Phase 1 — Data Pipeline & Preprocessing
========================================
Loads the Bitbrains cloud workload trace (1.csv or any compatible CSV),
cleans and engineers features, scales data, and produces sliding-window
tensors for ARIMA, XGBoost, and LSTM training.

Dataset columns expected (Bitbrains format):
  Timestamp [ms], CPU cores, CPU capacity provisioned [MHZ],
  CPU usage [MHZ], CPU usage [%], Memory capacity provisioned [KB],
  Memory usage [KB], Disk read throughput [KB/s],
  Disk write throughput [KB/s], Network received throughput [KB/s],
  Network transmitted throughput [KB/s]
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class WorkloadDataPreprocessor:
    """
    Full data pipeline for cloud workload preprocessing and feature engineering.

    Steps performed:
      1. Load CSV (Bitbrains format) and parse Unix timestamps
      2. Forward-fill / backward-fill any missing values
      3. Engineer cyclical time features (hour_sin, hour_cos) and lag features
      4. Build rolling statistics (mean, std, max)
      5. Fit scalers on training data only (no data leakage)
      6. Produce XGBoost tabular arrays, LSTM 3-D tensors, and raw ARIMA series
      7. Persist everything as a single joblib bundle for downstream scripts
    """

    def __init__(
        self,
        filepath="1.csv",          # <-- points to your real Bitbrains dataset
        target_col="CPU usage [%]",
        seq_length=12,              # 12 steps x 5 min = 1-hour look-back window
        train_ratio=0.80,
        scaler_type="minmax"
    ):
        """
        Parameters
        ----------
        filepath    : Path to the raw workload CSV file
        target_col  : Column to forecast (CPU or Memory)
        seq_length  : Sliding-window width for LSTM / lag features
        train_ratio : Fraction of data used for training (rest = test)
        scaler_type : 'minmax' (0-1) or 'standard' (z-score)
        """
        self.filepath = filepath
        self.target_col = target_col
        self.seq_length = seq_length
        self.train_ratio = train_ratio
        self.scaler_type = scaler_type

        # Scalers are instantiated here and fitted only on training data
        self.target_scaler = (
            MinMaxScaler(feature_range=(0, 1))
            if scaler_type == "minmax"
            else StandardScaler()
        )
        self.feature_scaler = (
            MinMaxScaler(feature_range=(0, 1))
            if scaler_type == "minmax"
            else StandardScaler()
        )

    # ------------------------------------------------------------------
    def load_and_clean_data(self):
        """
        Loads CSV, converts the Unix-second timestamp column to datetime,
        sorts chronologically, and imputes missing values.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset not found at '{self.filepath}'.\n"
                "Place your 1.csv file in the project folder, or run "
                "generate_synthetic_data.py to create a synthetic version."
            )

        print(f"[INFO] Loading dataset from '{self.filepath}' ...")
        df = pd.read_csv(self.filepath)

        # --- Timestamp handling ---
        if "Timestamp [ms]" in df.columns:
            # Bitbrains stores Unix seconds (despite the [ms] label in some files)
            # We detect whether values look like seconds or milliseconds
            sample_val = df["Timestamp [ms]"].iloc[0]
            if sample_val > 1e12:          # genuine milliseconds
                df["datetime"] = pd.to_datetime(df["Timestamp [ms]"], unit="ms")
            else:                          # stored as seconds
                df["datetime"] = pd.to_datetime(df["Timestamp [ms]"], unit="s")
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
        else:
            # Fallback: synthesize a 5-minute index
            df["datetime"] = pd.date_range(
                start="2013-08-12", periods=len(df), freq="5min"
            )

        df = df.sort_values("datetime").reset_index(drop=True)

        # --- Impute missing values ---
        df = df.ffill().bfill()

        # --- Validate target column ---
        if self.target_col not in df.columns:
            cpu_cols = [c for c in df.columns if "CPU" in c or "cpu" in c]
            if cpu_cols:
                self.target_col = cpu_cols[0]
                print(f"[WARN] Target column not found; using '{self.target_col}'.")
            else:
                raise KeyError(
                    f"Target column '{self.target_col}' not in dataset. "
                    f"Available: {list(df.columns)}"
                )

        print(
            f"[INFO] Loaded {len(df):,} records. "
            f"Date range: {df['datetime'].min()} -> {df['datetime'].max()}. "
            f"Target: '{self.target_col}'"
        )
        return df

    # ------------------------------------------------------------------
    def engineer_features(self, df):
        """
        Creates all features needed by XGBoost and LSTM:
          - Cyclical hour encoding (sin / cos) to capture diurnal patterns
          - Day-of-week + weekend flag
          - Lag features (t-1 … t-seq_length) for autoregressive modelling
          - Rolling mean, std, and max over 3- and 6-step windows
        """
        df = df.copy()

        # Cyclical time encoding
        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

        # Lag features  (shift by 1..seq_length steps; all use the TARGET column)
        for lag in range(1, self.seq_length + 1):
            df[f"lag_{lag}"] = df[self.target_col].shift(lag)

        # Rolling statistics (shift-1 so we never leak the current value)
        shifted = df[self.target_col].shift(1)
        df["rolling_mean_3"] = shifted.rolling(window=3).mean()
        df["rolling_std_3"]  = shifted.rolling(window=3).std()
        df["rolling_mean_6"] = shifted.rolling(window=6).mean()
        df["rolling_max_6"]  = shifted.rolling(window=6).max()

        # Drop the NaN rows created by shifting / rolling
        df_clean = df.dropna().reset_index(drop=True)
        print(f"[INFO] After feature engineering: {len(df_clean):,} usable rows.")
        return df_clean

    # ------------------------------------------------------------------
    def create_lstm_sequences(self, data_scaled, target_scaled):
        """
        Builds 3-D sliding-window tensors for LSTM training.

        Returns
        -------
        X : np.ndarray of shape (N, seq_length, n_features)
        y : np.ndarray of shape (N,)
        """
        X, y = [], []
        for i in range(len(data_scaled) - self.seq_length):
            X.append(data_scaled[i : i + self.seq_length])
            y.append(target_scaled[i + self.seq_length])
        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    def process_and_save(self, output_filepath="processed_data.joblib"):
        """
        Executes the full pipeline and saves a joblib bundle containing:
          - XGBoost train/test arrays
          - LSTM 3-D tensors
          - ARIMA raw series
          - Fitted scalers (for inverse-transforming predictions later)
          - Full feature DataFrame for dashboard visualisation
        """
        raw_df  = self.load_and_clean_data()
        feat_df = self.engineer_features(raw_df)

        # Feature column lists
        lag_cols  = [c for c in feat_df.columns if c.startswith("lag_") or c.startswith("rolling_")]
        time_cols = ["hour_sin", "hour_cos", "is_weekend"]
        xgb_feature_cols  = lag_cols + time_cols
        lstm_feature_cols = [self.target_col, "hour_sin", "hour_cos"]

        # ---- Sequential train / test split (no shuffle — time-series!) ----
        split_idx = int(len(feat_df) * self.train_ratio)
        train_df = feat_df.iloc[:split_idx].copy()
        test_df  = feat_df.iloc[split_idx:].copy()

        print(
            f"[INFO] Train rows: {len(train_df):,} | "
            f"Test rows: {len(test_df):,}"
        )

        # ---- Fit + transform scalers on TRAIN only ----
        train_target_scaled = self.target_scaler.fit_transform(
            train_df[[self.target_col]].values
        ).flatten()
        test_target_scaled = self.target_scaler.transform(
            test_df[[self.target_col]].values
        ).flatten()

        train_feats_scaled = self.feature_scaler.fit_transform(
            train_df[lstm_feature_cols].values
        )
        test_feats_scaled = self.feature_scaler.transform(
            test_df[lstm_feature_cols].values
        )

        # ---- LSTM 3-D tensors ----
        X_train_lstm, y_train_lstm = self.create_lstm_sequences(
            train_feats_scaled, train_target_scaled
        )
        X_test_lstm, y_test_lstm = self.create_lstm_sequences(
            test_feats_scaled, test_target_scaled
        )

        # ---- XGBoost tabular arrays ----
        X_train_xgb = train_df[xgb_feature_cols].values
        y_train_xgb = train_df[self.target_col].values
        X_test_xgb  = test_df[xgb_feature_cols].values
        y_test_xgb  = test_df[self.target_col].values

        # ---- Raw ARIMA series ----
        arima_train = train_df[self.target_col].values
        arima_test  = test_df[self.target_col].values

        bundle = {
            "target_col"      : self.target_col,
            "seq_length"      : self.seq_length,
            "target_scaler"   : self.target_scaler,
            "feature_scaler"  : self.feature_scaler,
            "xgb_feature_cols": xgb_feature_cols,
            "lstm_feature_cols": lstm_feature_cols,
            "df_full"         : feat_df,
            "train_df"        : train_df,
            "test_df"         : test_df,
            # XGBoost
            "X_train_xgb"     : X_train_xgb,
            "y_train_xgb"     : y_train_xgb,
            "X_test_xgb"      : X_test_xgb,
            "y_test_xgb"      : y_test_xgb,
            # LSTM
            "X_train_lstm"    : X_train_lstm,
            "y_train_lstm"    : y_train_lstm,
            "X_test_lstm"     : X_test_lstm,
            "y_test_lstm"     : y_test_lstm,
            # ARIMA
            "arima_train"     : arima_train,
            "arima_test"      : arima_test,
            # Timestamps aligned to the LSTM test window
            "timestamps_test" : test_df["datetime"].values[self.seq_length:],
        }

        joblib.dump(bundle, output_filepath)
        print(f"[INFO] Preprocessed bundle saved to '{output_filepath}'.")
        return bundle


# -----------------------------------------------------------------------
if __name__ == "__main__":
    prep = WorkloadDataPreprocessor(filepath="1.csv")
    prep.process_and_save()
