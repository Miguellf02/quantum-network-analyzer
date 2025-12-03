"""
train_autoencoder.py
------------------------------------
Entrenamiento de un Autoencoder (PyTorch) para detección de anomalías
en el sistema QKD, usando el dataset QKD_FEATURES.csv.

Pipeline:
- Carga del dataset desde constants.py
- Limpieza de NaN/inf
- Selección de columnas numéricas
- Estandarización con StandardScaler
- Construcción de Autoencoder MLP
- Entrenamiento
- Cálculo de reconstruction error
- Guardado de modelo, scaler y CSV con anomaly_score
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler

# Rutas desde constants.py
from src.constants.constants import (
    INPUT_FEATURED_PATH,
    MODEL_OUTPUT_DIR
)


# ==================================================
#               UTILS
# ==================================================

def load_featured_dataset():
    if not INPUT_FEATURED_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de características:\n{INPUT_FEATURED_PATH}"
        )

    df = pd.read_csv(INPUT_FEATURED_PATH)
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def select_numeric_clean(df):
    """Selecciona columnas numéricas, limpia INF/NaN y devuelve df limpio + matriz."""
    X = df.select_dtypes(include=[np.number])

    # Reemplazar INF -> NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop NaN rows
    valid_idx = X.dropna().index
    X_clean = X.loc[valid_idx]
    df_clean = df.loc[valid_idx]

    print(f"[INFO] Columnas numéricas: {X_clean.shape[1]}")
    print(f"[INFO] Filas válidas tras limpieza: {X_clean.shape[0]} / {df.shape[0]}")

    return X_clean, df_clean


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ==================================================
#              AUTOENCODER MODEL
# ==================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
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
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ==================================================
#               TRAINING LOOP
# ==================================================

def train_autoencoder(X_scaled, latent_dim=16, epochs=25, batch_size=128, lr=1e-3):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = X_scaled.shape[1]

    model = Autoencoder(input_dim, latent_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoader
    tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_X), batch_size=batch_size, shuffle=True)

    print(f"[INFO] Entrenando Autoencoder en {device}...")
    print(f"[INFO] Input dim = {input_dim}, Latent dim = {latent_dim}")

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            xb = batch[0].to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, xb)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"[EPOCH {epoch+1}/{epochs}] Loss = {np.mean(losses):.6f}")

    return model


# ==================================================
#          SCORE BY RECONSTRUCTION ERROR
# ==================================================

def compute_reconstruction_error(model, X_scaled):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_tensor).cpu().numpy()

    mse = np.mean((X_scaled - recon)**2, axis=1)
    return mse  # anomaly_score


# ==================================================
#                SAVE RESULTS
# ==================================================

def save_outputs(df, scores, model, scaler):
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) CSV
    df_out = df.copy()
    df_out["anomaly_score_autoencoder"] = scores

    output_csv = MODEL_OUTPUT_DIR / "AUTOENCODER_RESULTS.csv"
    df_out.to_csv(output_csv, index=False)
    print(f"[OK] Resultados guardados en: {output_csv}")

    # 2) Modelo .pt
    model_path = MODEL_OUTPUT_DIR / "AUTOENCODER_MODEL.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Modelo guardado en: {model_path}")

    # 3) Scaler
    scaler_path = MODEL_OUTPUT_DIR / "AUTOENCODER_SCALER.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Scaler guardado en: {scaler_path}")


# ==================================================
#                   MAIN
# ==================================================

def main():
    print("\n========== AUTOENCODER TRAINING ==========\n")

    # 1. Cargar dataset
    df = load_featured_dataset()

    # 2. Selección + limpieza
    X_clean, df_clean = select_numeric_clean(df)

    # 3. Estandarización
    print("[INFO] Estandarizando características...")
    X_scaled, scaler = scale_features(X_clean)

    # 4. Entrenar Autoencoder
    model = train_autoencoder(
        X_scaled,
        latent_dim=16,
        epochs=25,
        batch_size=128,
        lr=1e-3
    )

    # 5. Cálculo de reconstruction error
    print("[INFO] Calculando reconstruction error...")
    scores = compute_reconstruction_error(model, X_scaled)

    # 6. Guardar resultados
    print("[INFO] Guardando resultados y modelo...")
    save_outputs(df_clean, scores, model, scaler)

    print("\n[OK] Autoencoder COMPLETADO.\n")


if __name__ == "__main__":
    main()
