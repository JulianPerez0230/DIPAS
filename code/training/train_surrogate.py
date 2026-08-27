# -*- coding: utf-8 -*-
"""
Script de Entrenamiento de los Dos Modelos Sustitutos (Surrogates):
1. Surrogate 1 (Numérico): XFOIL -> Fluent CFD.
2. Surrogate 2 (Híbrido): XFOIL -> UIUC Tunnel Data -> Fluent CFD.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

from surrogate_model import AerodynamicSurrogate

class TabularAeroDataset(Dataset):
    def __init__(self, df, input_cols, output_cols, scaler=None):
        self.inputs = df[input_cols].values.astype(np.float32)
        self.outputs = df[output_cols].values.astype(np.float32)
        
        # Guardar o calcular scalers para normalización
        if scaler is None:
            self.input_mean = self.inputs.mean(axis=0)
            self.input_std = self.inputs.std(axis=0)
            self.input_std[self.input_std == 0] = 1.0
            
            self.output_mean = self.outputs.mean(axis=0)
            self.output_std = self.outputs.std(axis=0)
            self.output_std[self.output_std == 0] = 1.0
        else:
            self.input_mean = scaler["input_mean"]
            self.input_std = scaler["input_std"]
            self.output_mean = scaler["output_mean"]
            self.output_std = scaler["output_std"]
            
        # Normalizar datos
        self.inputs_norm = (self.inputs - self.input_mean) / self.input_std
        self.outputs_norm = (self.outputs - self.output_mean) / self.output_std
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return torch.tensor(self.inputs_norm[idx], dtype=torch.float32), torch.tensor(self.outputs_norm[idx], dtype=torch.float32)
        
    def get_scaler_dict(self):
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "output_mean": self.output_mean.tolist(),
            "output_std": self.output_std.tolist()
        }

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(x)
    return epoch_loss / len(loader.dataset)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * len(x)
    return val_loss / len(loader.dataset)

def main():
    # Rutas de datos
    xfoil_path = SCRIPT_DIR.parent / "data" / "dataset.csv"
    cfd_path = SCRIPT_DIR.parent / "data" / "dataset_cfd.csv"
    uiuc_path = SCRIPT_DIR.parent / "data" / "uiuc_polar_dataset.csv"
    output_dir = SCRIPT_DIR.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scalers_save_path = output_dir / "surrogate_scalers.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Iniciando entrenamiento de los Surrogates en: {device}")
    
    # Columnas de entrada y salida
    cst_cols = [f"au{i}" for i in range(6)] + [f"al{i}" for i in range(6)]
    input_cols = cst_cols + ["alpha", "reynolds"]
    output_cols = ["cl", "cd"]
    
    # -------------------------------------------------------------
    # 1. Cargar y normalizar datasets
    # -------------------------------------------------------------
    print(">> Cargando datasets...")
    df_xfoil = pd.read_csv(xfoil_path)
    df_cfd = pd.read_csv(cfd_path)
    df_uiuc = pd.read_csv(uiuc_path)
    
    # Usar las estadísticas de XFOIL como scaler base para garantizar un espacio unificado
    train_xfoil_dataset = TabularAeroDataset(df_xfoil, input_cols, output_cols)
    base_scaler = train_xfoil_dataset.get_scaler_dict()
    
    # Guardar scalers a JSON
    with open(scalers_save_path, "w", encoding="utf-8") as f:
        json.dump(base_scaler, f, indent=4)
    print(f">> Scaler aerodinámico unificado guardado en: {scalers_save_path.resolve()}")
    
    # Cargar datasets con el scaler unificado
    cfd_full_dataset = TabularAeroDataset(df_cfd, input_cols, output_cols, scaler=base_scaler)
    uiuc_full_dataset = TabularAeroDataset(df_uiuc, input_cols, output_cols, scaler=base_scaler)
    
    # Separación entrenamiento/validación para CFD (90/10)
    train_size = int(0.9 * len(cfd_full_dataset))
    val_size = len(cfd_full_dataset) - train_size
    cfd_train, cfd_val = random_split(
        cfd_full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Dataloaders
    batch_size = 128
    xfoil_loader = DataLoader(train_xfoil_dataset, batch_size=batch_size, shuffle=True)
    uiuc_loader = DataLoader(uiuc_full_dataset, batch_size=batch_size, shuffle=True)
    cfd_train_loader = DataLoader(cfd_train, batch_size=32, shuffle=True)
    cfd_val_loader = DataLoader(cfd_val, batch_size=32, shuffle=False)
    
    criterion = nn.MSELoss()
    
    # =========================================================================
    # ENTRENAMIENTO SURROGATE 1: NUMÉRICO (XFOIL -> CFD)
    # =========================================================================
    print("\n==================================================")
    print("  ENTRENANDO SURROGATE 1: NUMÉRICO (XFOIL -> CFD) ")
    print("==================================================")
    model1 = AerodynamicSurrogate().to(device)
    
    # Fase A: Pre-entrenamiento XFOIL (30 épocas)
    optimizer = optim.Adam(model1.parameters(), lr=0.001)
    print(">> Fase A: Pre-entrenamiento en XFOIL...")
    for epoch in range(1, 31):
        loss = train_epoch(model1, xfoil_loader, criterion, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/30 | Loss XFOIL: {loss:.5f}")
            
    # Fase B: Fine-tuning CFD (40 épocas)
    optimizer = optim.Adam(model1.parameters(), lr=0.0001)
    print(">> Fase B: Fine-tuning en Fluent CFD...")
    best_val_loss = float('inf')
    for epoch in range(1, 41):
        train_loss = train_epoch(model1, cfd_train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model1, cfd_val_loader, criterion, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/40 | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
    torch.save(model1.state_dict(), output_dir / "surrogate_numeric.pth")
    print(f">> Modelo Surrogate Numérico guardado.")
    
    # =========================================================================
    # ENTRENAMIENTO SURROGATE 2: HÍBRIDO (XFOIL -> UIUC -> CFD)
    # =========================================================================
    print("\n==================================================")
    print("  ENTRENANDO SURROGATE 2: HÍBRIDO (XFOIL -> UIUC -> CFD) ")
    print("==================================================")
    model2 = AerodynamicSurrogate().to(device)
    
    # Fase A: Pre-entrenamiento XFOIL (30 épocas)
    optimizer = optim.Adam(model2.parameters(), lr=0.001)
    print(">> Fase A: Pre-entrenamiento en XFOIL...")
    for epoch in range(1, 31):
        loss = train_epoch(model2, xfoil_loader, criterion, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/30 | Loss XFOIL: {loss:.5f}")
            
    # Fase B: Sintonización UIUC Experimental (30 épocas)
    optimizer = optim.Adam(model2.parameters(), lr=0.0005)
    print(">> Fase B: Sintonización en Datos de Túnel UIUC...")
    for epoch in range(1, 31):
        loss = train_epoch(model2, uiuc_loader, criterion, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/30 | Loss UIUC: {loss:.5f}")
            
    # Fase C: Fine-tuning CFD (40 épocas)
    optimizer = optim.Adam(model2.parameters(), lr=0.0001)
    print(">> Fase C: Fine-tuning final en Fluent CFD...")
    for epoch in range(1, 41):
        train_loss = train_epoch(model2, cfd_train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model2, cfd_val_loader, criterion, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/40 | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
    torch.save(model2.state_dict(), output_dir / "surrogate_hybrid.pth")
    print(f">> Modelo Surrogate Híbrido guardado.")
    print("\n>> Entrenamiento de ambos Surrogates finalizado con éxito.")

if __name__ == "__main__":
    main()
