"""
train_autoencoder_per_source.py
------------------------------------------------
Entrenamiento de Autoencoder por source
(QTI / TOSHIBA-2024-W25 / TOSHIBA-2025-W27)
"""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.constants.constants import (
    INPUT_FEATURED_PATH,
    MODEL_OUTPUT_DIR
)

# ================================================================
# CONFIG
# ================================================================

SOURCES = [
    "QTI",
    "TOSHIBA-2024-W25",
    "TOSHIBA-2025-W27"
]

LATENT_DIM = 16
EPOCHS = 25
BATCH_SIZE = 128
LR = 1e-3
NAN_THRESHOLD = 0.4


# ================================================================
# UTILS
# ================================================================

def normalize_source(df):
    df["source"] = df["source"].astype(str).str.strip().str.upper()
    return df


def prepare_numeric_features(df):
    X = df.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    nan_ratio = X.isna().mean()
    cols = nan_ratio[nan_ratio <= NAN_THRESHOLD].index

    if len(cols) == 0:
        return None, None

    X = X[cols]
    valid_idx = X.dropna().index

    if len(valid_idx) == 0:
        return None, None

    return X.loc[valid_idx], df.loc[valid_idx]


# ================================================================
# MODEL
# ================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ================================================================
# TRAIN / SCORE
# ================================================================

def train_autoencoder(X_scaled):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(X_scaled.shape[1], LATENT_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"[EPOCH {epoch+1}/{EPOCHS}] Loss = {np.mean(losses):.6f}")

    return model


def reconstruction_error(model, X_scaled):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        recon = model(X_tensor).cpu().numpy()

    return np.mean((X_scaled - recon) ** 2, axis=1)


# ================================================================
# MAIN
# ================================================================

def main():

    df = pd.read_csv(INPUT_FEATURED_PATH)
    df = normalize_source(df)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:

        print(f"\n========== AUTOENCODER SOURCE: {source} ==========")
        df_src = df[df["source"] == source]

        if df_src.empty:
            print("[WARN] No data")
            continue

        X, df_clean = prepare_numeric_features(df_src)
        if X is None:
            print("[WARN] No valid features")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = train_autoencoder(X_scaled)
        scores = reconstruction_error(model, X_scaled)

        df_out = df_clean.copy()
        df_out["anomaly_score_autoencoder"] = scores

        df_out.to_csv(
            MODEL_OUTPUT_DIR / f"AUTOENCODER_RESULTS_{source}.csv",
            index=False
        )

        torch.save(
            model.state_dict(),
            MODEL_OUTPUT_DIR / f"AUTOENCODER_MODEL_{source}.pt"
        )

        joblib.dump(
            scaler,
            MODEL_OUTPUT_DIR / f"AUTOENCODER_SCALER_{source}.joblib"
        )

        print(f"[OK] Autoencoder guardado para {source}")


if __name__ == "__main__":
    main()
