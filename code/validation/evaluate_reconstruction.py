# -*- coding: utf-8 -*-
"""
Script de Evaluación de Reconstrucción Geométrica y Aerodinámica.
Calcula RMSE, MAE, Delta(t/c) y Delta(xtmax/c) sobre el conjunto de validación.
Grafica 5 perfiles representativos (fácil, curvado, grueso, fino y no convencional).
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import random_split

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

from cvae_model import CVAE
from cvae_dataset import AirfoilCSTDataset
from cst_generator import CSTParametrization

def calculate_thickness_properties(cst, coefs):
    """
    Calcula el espesor máximo (t/c) y su posición (xtmax/c) a partir de los coeficientes CST.
    """
    # Generar coordenadas con alta resolución para precisión en el pico
    x, y_up, y_low = cst.generate_coordinates(coefs[:6], coefs[6:], n_points=300)
    thickness = y_up - y_low
    
    max_idx = np.argmax(thickness)
    t_c = thickness[max_idx]
    x_t_c = x[max_idx]
    
    return t_c, x_t_c, x, y_up, y_low

def main():
    dataset_path = SCRIPT_DIR.parent / "data" / "uiuc_cst_dataset.csv"
    model_path = SCRIPT_DIR.parent / "outputs" / "base_autoencoder.pth"
    output_dir = SCRIPT_DIR.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not dataset_path.exists():
        print(f"❌ Dataset no encontrado: {dataset_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"❌ Modelo entrenado no encontrado: {model_path}")
        sys.exit(1)
        
    # 1. Cargar Dataset y dividir idénticamente (90/10)
    full_dataset = AirfoilCSTDataset(str(dataset_path), use_conditions=False)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    _, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 2. Cargar modelo
    device = torch.device("cpu")
    model = CVAE(cst_dim=12, cond_dim=0, latent_dim=6, hidden_dims=[256, 128, 64])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    
    # 3. Evaluar métricas en el conjunto de validación
    rmse_list = []
    mae_list = []
    delta_tc_list = []
    delta_xtc_list = []
    
    print(">> Calculando métricas de reconstrucción sobre el conjunto de validación...")
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            # Obtener datos reales
            x_cst_real_tensor, _ = val_dataset[i]
            x_cst_real = x_cst_real_tensor.numpy()
            
            # Reconstruir con el modelo
            x_cst_recon_tensor, _, _ = model(x_cst_real_tensor.unsqueeze(0))
            x_cst_recon = x_cst_recon_tensor.squeeze(0).numpy()
            
            # Propiedades reales
            tc_real, xtc_real, x_pts, y_up_real, y_low_real = calculate_thickness_properties(cst, x_cst_real)
            # Propiedades reconstruidas
            tc_recon, xtc_recon, _, y_up_recon, y_low_recon = calculate_thickness_properties(cst, x_cst_recon)
            
            # Unir coordenadas y para calcular RMSE y MAE global
            y_real = np.concatenate([y_up_real, y_low_real])
            y_recon = np.concatenate([y_up_recon, y_low_recon])
            
            rmse = np.sqrt(np.mean((y_real - y_recon) ** 2))
            mae = np.mean(np.abs(y_real - y_recon))
            
            rmse_list.append(rmse)
            mae_list.append(mae)
            delta_tc_list.append(abs(tc_real - tc_recon))
            delta_xtc_list.append(abs(xtc_real - xtc_recon))
            
    # Mostrar tabla resumen de métricas
    print("\n==========================================================================")
    print("                      Metricas de Validacion (Promedios)                  ")
    print("==========================================================================")
    print(f" RMSE Geometrico Global   : {np.mean(rmse_list):.6f} +/- {np.std(rmse_list):.6f}")
    print(f" MAE Geometrico Global    : {np.mean(mae_list):.6f} +/- {np.std(mae_list):.6f}")
    print(f" Delta(t/c) [Espesor Max.]: {np.mean(delta_tc_list):.6f} +/- {np.std(delta_tc_list):.6f}")
    print(f" Delta(xtmax/c) [Posicion]: {np.mean(delta_xtc_list):.6f} +/- {np.std(delta_xtc_list):.6f}")
    print("==========================================================================\n")
    
    # 4. Probar perfiles extremos específicos en el dataset
    # Mapearemos perfiles de interés representativos por nombre
    targets = {
        "Fácil/Estándar": "clarky",
        "Muy Curvado (Camber)": "s3021",
        "Grueso (t/c > 18%)": "goe428",
        "Fino (t/c < 6%)": "ag03",
        "No Convencional / Reflex": "fx60126"
    }
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    
    df_cst = pd.read_csv(dataset_path)
    
    cst_cols = [
        "au0", "au1", "au2", "au3", "au4", "au5",
        "al0", "al1", "al2", "al3", "al4", "al5"
    ]
    
    for idx, (label, name) in enumerate(targets.items()):
        # Buscar en el CSV
        row = df_cst[df_cst["airfoil_name"] == name]
        
        if row.empty:
            # Fallback en caso de no encontrarse con el nombre exacto
            print(f"⚠️ Perfil {name} no encontrado en el dataset, usando fila fallback.")
            row = df_cst.iloc[idx * 150]
            name = row["airfoil_name"]
            
        real_cst_vals = row[cst_cols].values[0].astype(np.float32)
        real_tensor = torch.tensor(real_cst_vals).unsqueeze(0)
        
        with torch.no_grad():
            recon_tensor, _, _ = model(real_tensor)
            recon_cst_vals = recon_tensor.squeeze(0).numpy()
            
        tc_r, xtc_r, x_pts, y_up_r, y_low_r = calculate_thickness_properties(cst, real_cst_vals)
        tc_rec, xtc_rec, _, y_up_rec, y_low_rec = calculate_thickness_properties(cst, recon_cst_vals)
        
        ax = axes[idx]
        ax.plot(x_pts, y_up_r, 'k-', label=f"Real ({name})", linewidth=2.0)
        ax.plot(x_pts, y_low_r, 'k-', linewidth=2.0)
        
        ax.plot(x_pts, y_up_rec, 'r--', label=f"Reconstruido (IA)", linewidth=1.5)
        ax.plot(x_pts, y_low_rec, 'r--', linewidth=1.5)
        
        # Marcar la posición del espesor máximo real
        ax.axvline(x=xtc_r, color='gray', linestyle=':', alpha=0.7)
        
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Categoría: {label} (Perfil: {name})\n"
                     f"Real: t/c={tc_r:.3f} @ x={xtc_r:.3f} | IA: t/c={tc_rec:.3f} @ x={xtc_rec:.3f}", fontsize=10)
        ax.legend(loc="upper right")
        
    axes[-1].set_xlabel("x/c")
    plt.tight_layout()
    plot_path = output_dir / "evaluation_extremes.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f">> Gráfico de perfiles extremos guardado en: {plot_path.resolve()}")

if __name__ == "__main__":
    main()
