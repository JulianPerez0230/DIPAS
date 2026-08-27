# -*- coding: utf-8 -*-
"""
Experimento 1: DIPAS Base.
Entrenamiento condicional (CVAE) desde cero utilizando únicamente datos propios:
Fase A: Pre-entrenamiento con 10.000 muestras de XFOIL (data/dataset.csv).
Fase B: Ajuste fino de alta fidelidad con ~3.000 muestras de Fluent (data/dataset_cfd.csv).
"""

import os
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import random_split

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

from cvae_model import CVAE, cvae_loss_function
from cvae_dataset import AirfoilCSTDataset, DataLoader
from cst_generator import CSTParametrization
from evaluate_reconstruction import calculate_thickness_properties

def main():
    # Rutas
    xfoil_path = SCRIPT_DIR.parent / "data" / "dataset.csv"
    cfd_path = SCRIPT_DIR.parent / "data" / "dataset_cfd.csv"
    output_dir = SCRIPT_DIR.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_path = output_dir / "dipas_base_model.pth"
    loss_plot_path = output_dir / "dipas_base_losses.png"
    recon_plot_path = output_dir / "dipas_base_evaluation_extremes.png"
    
    if not xfoil_path.exists():
        print(f"❌ Dataset de XFOIL no encontrado: {xfoil_path}")
        sys.exit(1)
    if not cfd_path.exists():
        print(f"❌ Dataset de CFD no encontrado: {cfd_path}")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Iniciando Experimento 1 (DIPAS Base) en {device}...")
    
    # -------------------------------------------------------------
    # FASE A: Entrenamiento en 10.000 muestras de XFOIL
    # -------------------------------------------------------------
    print("\n--- Fase A: Entrenamiento con 10.000 muestras de XFOIL ---")
    batch_size = 64
    epochs_xfoil = 100
    lr_xfoil = 0.001
    
    # Cargar submuestra de 10k de XFOIL
    xfoil_dataset = AirfoilCSTDataset(str(xfoil_path), use_conditions=True, sample_size=10000)
    xfoil_loader = DataLoader(xfoil_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Inicializar CVAE (dim_entrada=12, dim_condición=4 [cl, cd, re, tc], latente=6)
    model = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_xfoil)
    
    for epoch in range(1, epochs_xfoil + 1):
        model.train()
        epoch_loss = 0
        for batch_x, batch_c in xfoil_loader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_c)
            loss, _, _, _ = cvae_loss_function(
                recon_x, batch_x, mu, logvar, model.cst_to_coord, beta=0.01, geom_weight=20.0
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
            
        epoch_loss /= len(xfoil_loader.dataset)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs_xfoil:03d} | XFOIL Loss: {epoch_loss:.5f}")
            
    print(">> Fase A completada.")
    
    # -------------------------------------------------------------
    # FASE B: Ajuste Fino con Dataset CFD de Fluent (~3.000 corridas)
    # -------------------------------------------------------------
    print("\n--- Fase B: Ajuste Fino con CFD (Fluent) ---")
    epochs_cfd = 60
    lr_cfd = 0.0001 # Tasa de aprendizaje más baja para preservar conocimiento previo
    
    # Cargar CFD y separar en train/val (90/10)
    cfd_dataset = AirfoilCSTDataset(str(cfd_path), use_conditions=True)
    train_size = int(0.9 * len(cfd_dataset))
    val_size = len(cfd_dataset) - train_size
    
    cfd_train, cfd_val = random_split(
        cfd_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(cfd_train, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Ajustamos la tasa de aprendizaje en el optimizador
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_cfd
        
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs_cfd + 1):
        model.train()
        epoch_loss = 0
        for batch_x, batch_c in train_loader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_c)
            loss, _, _, _ = cvae_loss_function(
                recon_x, batch_x, mu, logvar, model.cst_to_coord, beta=0.01, geom_weight=20.0
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validación
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch_x, batch_c in DataLoader(cfd_val, batch_size=batch_size, shuffle=False):
                batch_x = batch_x.to(device)
                batch_c = batch_c.to(device)
                recon_x, mu, logvar = model(batch_x, batch_c)
                loss, _, _, _ = cvae_loss_function(
                    recon_x, batch_x, mu, logvar, model.cst_to_coord, beta=0.01, geom_weight=20.0
                )
                v_loss += loss.item() * len(batch_x)
        v_loss /= len(cfd_val)
        val_losses.append(v_loss)
        
        if epoch % 15 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs_cfd:02d} | Train CFD Loss: {epoch_loss:.5f} | Val CFD Loss: {v_loss:.5f}")
            
    # Guardar modelo
    torch.save(model.state_dict(), model_save_path)
    print(f"\n>> Modelo final DIPAS Base guardado en: {model_save_path}")
    
    # Graficar curva de pérdidas
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss (CFD)")
    plt.plot(val_losses, '--', label="Val Loss (CFD)")
    plt.title("Curva de Pérdidas de Ajuste Fino CFD - DIPAS Base")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------
    # EVALUACIÓN DE MÉTRICAS (CFD Validación)
    # -------------------------------------------------------------
    print("\n--- Evaluación de Métricas en el Conjunto de Validación (CFD) ---")
    model.eval()
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    
    rmse_list = []
    mae_list = []
    delta_tc_list = []
    delta_xtc_list = []
    
    with torch.no_grad():
        for i in range(len(cfd_val)):
            x_cst_real_tensor, c_tensor = cfd_val[i]
            x_cst_real = x_cst_real_tensor.numpy()
            
            x_cst_recon_tensor, _, _ = model(x_cst_real_tensor.unsqueeze(0).to(device), c_tensor.unsqueeze(0).to(device))
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
            
    print("==========================================================================")
    print("                      DIPAS Base: Metricas de Validacion                  ")
    print("==========================================================================")
    print(f" RMSE Geometrico Global   : {np.mean(rmse_list):.6f} +/- {np.std(rmse_list):.6f}")
    print(f" MAE Geometrico Global    : {np.mean(mae_list):.6f} +/- {np.std(mae_list):.6f}")
    print(f" Delta(t/c) [Espesor Max.]: {np.mean(delta_tc_list):.6f} +/- {np.std(delta_tc_list):.6f}")
    print(f" Delta(xtmax/c) [Posicion]: {np.mean(delta_xtc_list):.6f} +/- {np.std(delta_xtc_list):.6f}")
    print("==========================================================================\n")
    
    # Graficar extremos específicos del dataset CFD
    targets = {
        "Fácil/Estándar": "clarky",
        "Muy Curvado (Camber)": "s3021",
        "Grueso": "goe428",
        "Fino": "ag03",
        "No Convencional": "fx60126"
    }
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    cst_cols = ["au0", "au1", "au2", "au3", "au4", "au5", "al0", "al1", "al2", "al3", "al4", "al5"]
    
    for idx, (label, name) in enumerate(targets.items()):
        row = cfd_dataset.df[cfd_dataset.df["seed"] == name]
        if row.empty:
            row_idx = idx * 50
            row = cfd_dataset.df.iloc[[row_idx]]
            name = row["seed"].values[0]
        else:
            row_idx = row.index[0]
            
        real_cst_vals = row[cst_cols].values[0].astype(np.float32)
        real_tensor = torch.tensor(real_cst_vals).unsqueeze(0).to(device)
        
        # Obtener las condiciones normalizadas correspondientes
        c_vals = cfd_dataset.cond_data[row_idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            recon_tensor, _, _ = model(real_tensor, c_vals)
            recon_cst_vals = recon_tensor.squeeze(0).cpu().numpy()
            
        tc_r, xtc_r, x_pts, y_up_r, y_low_r = calculate_thickness_properties(cst, real_cst_vals)
        tc_rec, xtc_rec, _, y_up_rec, y_low_rec = calculate_thickness_properties(cst, recon_cst_vals)
        
        ax = axes[idx]
        ax.plot(x_pts, y_up_r, 'k-', label=f"Real ({name})", linewidth=2.0)
        ax.plot(x_pts, y_low_r, 'k-', linewidth=2.0)
        ax.plot(x_pts, y_up_rec, 'r--', label="DIPAS Base (IA)", linewidth=1.5)
        ax.plot(x_pts, y_low_rec, 'r--', linewidth=1.5)
        ax.axvline(x=xtc_r, color='gray', linestyle=':', alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Categoría: {label} (seed: {name})\n"
                     f"Real: t/c={tc_r:.3f} @ x={xtc_r:.3f} | IA: t/c={tc_rec:.3f} @ x={xtc_rec:.3f}", fontsize=10)
        ax.legend(loc="upper right")
        
    axes[-1].set_xlabel("x/c")
    plt.tight_layout()
    plt.savefig(recon_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f">> Gráfico de perfiles extremos guardado en: {recon_plot_path.resolve()}")
    print("\n>> Experimento 1 finalizado con éxito.")

if __name__ == "__main__":
    main()
