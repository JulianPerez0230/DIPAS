# -*- coding: utf-8 -*-
"""
Script para realizar un barrido (sweep) de hiperparámetros sobre la dimensión latente (z).
Entrena el Autoencoder con z = [2, 4, 6, 8] y compara los errores geométricos
y de espesor sobre el conjunto de validación.
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import random_split

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

from cvae_model import CVAE, cvae_loss_function
from cvae_dataset import AirfoilCSTDataset, DataLoader
from cst_generator import CSTParametrization
from evaluate_reconstruction import calculate_thickness_properties

def train_and_evaluate(latent_dim, dataset_path, epochs=100, batch_size=32, lr=0.001, beta=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar y dividir dataset
    full_dataset = AirfoilCSTDataset(str(dataset_path), use_conditions=False)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Inicializar modelo
    model = CVAE(cst_dim=12, cond_dim=0, latent_dim=latent_dim, hidden_dims=[64, 32]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Bucle de entrenamiento rápido
    for epoch in range(epochs):
        model.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, _, _ = cvae_loss_function(recon_x, batch_x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            
    # Evaluación
    model.eval()
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    
    rmse_list = []
    mae_list = []
    delta_tc_list = []
    delta_xtc_list = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            x_cst_real_tensor, _ = val_dataset[i]
            x_cst_real = x_cst_real_tensor.numpy()
            
            x_cst_recon_tensor, _, _ = model(x_cst_real_tensor.unsqueeze(0).to(device))
            x_cst_recon = x_cst_recon_tensor.squeeze(0).cpu().numpy()
            
            tc_real, xtc_real, x_pts, y_up_real, y_low_real = calculate_thickness_properties(cst, x_cst_real)
            tc_recon, xtc_recon, _, y_up_recon, y_low_recon = calculate_thickness_properties(cst, x_cst_recon)
            
            y_real = np.concatenate([y_up_real, y_low_real])
            y_recon = np.concatenate([y_up_recon, y_low_recon])
            
            rmse = np.sqrt(np.mean((y_real - y_recon) ** 2))
            mae = np.mean(np.abs(y_real - y_recon))
            
            rmse_list.append(rmse)
            mae_list.append(mae)
            delta_tc_list.append(abs(tc_real - tc_recon))
            delta_xtc_list.append(abs(xtc_real - xtc_recon))
            
    return {
        "latent_dim": latent_dim,
        "rmse_mean": np.mean(rmse_list),
        "rmse_std": np.std(rmse_list),
        "mae_mean": np.mean(mae_list),
        "mae_std": np.std(mae_list),
        "dtc_mean": np.mean(delta_tc_list),
        "dtc_std": np.std(delta_tc_list),
        "dxtc_mean": np.mean(delta_xtc_list),
        "dxtc_std": np.std(delta_xtc_list)
    }

def main():
    dataset_path = SCRIPT_DIR.parent / "data" / "uiuc_cst_dataset.csv"
    if not dataset_path.exists():
        print(f"❌ Dataset no encontrado: {dataset_path}")
        sys.exit(1)
        
    latent_dims = [2, 4, 6, 8]
    results = []
    
    print(">> Iniciando barrido de dimensiones latentes z = [2, 4, 6, 8]...")
    print("   Cada configuración entrenará por 100 épocas en CPU/GPU.")
    
    for l_dim in latent_dims:
        print(f"\nEntrenando con z = {l_dim}...")
        res = train_and_evaluate(l_dim, dataset_path, epochs=100)
        results.append(res)
        print(f"  -> RMSE: {res['rmse_mean']:.6f} | Delta(t/c): {res['dtc_mean']:.6f}")
        
    # Imprimir tabla comparativa
    print("\n==========================================================================================")
    print("                      TABLA COMPARATIVA DE DIMENSIÓN LATENTE (z)                         ")
    print("==========================================================================================")
    print(f" {'z':^5} | {'RMSE Global':^18} | {'MAE Global':^18} | {'Delta(t/c)':^18} | {'Delta(xtmax/c)':^18}")
    print("------------------------------------------------------------------------------------------")
    for r in results:
        print(f" {r['latent_dim']:^5d} | "
              f"{r['rmse_mean']:.6f} +/- {r['rmse_std']:.5f} | "
              f"{r['mae_mean']:.6f} +/- {r['mae_std']:.5f} | "
              f"{r['dtc_mean']:.6f} +/- {r['dtc_std']:.5f} | "
              f"{r['dxtc_mean']:.6f} +/- {r['dxtc_std']:.5f}")
    print("==========================================================================================\n")

if __name__ == "__main__":
    main()
