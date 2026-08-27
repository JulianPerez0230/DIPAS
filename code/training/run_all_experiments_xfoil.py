# -*- coding: utf-8 -*-
"""
Ejecución Integral de los 3 Experimentos CVAE y los 2 Modelos Surrogates (Sin CFD).
Diseñado para correr de inmediato mientras el barrido de Fluent de 3 días se completa.

Flujo:
1. CVAE Exp 1 (DIPAS Base): XFOIL (10k)
2. CVAE Exp 2 (Transfer Learning): UniFoil (10k) -> XFOIL (10k)
3. CVAE Exp 3 (TL + UIUC): UniFoil (10k) -> UIUC (1.6k) -> XFOIL (10k)
4. Surrogate 1 (Numérico): XFOIL
5. Surrogate 2 (Híbrido): XFOIL -> UIUC Experimental
6. Evaluación Comparativa Unificada (Geométrica + Aerodinámica)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

from cvae_model import CVAE, cvae_loss_function
from cvae_dataset import AirfoilCSTDataset
from surrogate_model import AerodynamicSurrogate
from cst_generator import CSTParametrization
from evaluate_reconstruction import calculate_thickness_properties

# =========================================================================
# UTILIDADES DE DATOS PARA SURROGATES
# =========================================================================
class TabularAeroDataset(torch.utils.data.Dataset):
    def __init__(self, df, input_cols, output_cols, scaler=None):
        self.inputs = df[input_cols].values.astype(np.float32)
        self.outputs = df[output_cols].values.astype(np.float32)
        
        if scaler is None:
            self.input_mean = self.inputs.mean(axis=0)
            self.input_std = self.inputs.std(axis=0)
            self.input_std[self.input_std == 0] = 1.0
            
            self.output_mean = self.outputs.mean(axis=0)
            self.output_std = self.outputs.std(axis=0)
            self.output_std[self.output_std == 0] = 1.0
        else:
            self.input_mean = np.array(scaler["input_mean"], dtype=np.float32)
            self.input_std = np.array(scaler["input_std"], dtype=np.float32)
            self.output_mean = np.array(scaler["output_mean"], dtype=np.float32)
            self.output_std = np.array(scaler["output_std"], dtype=np.float32)
            
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

def train_surrogate_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)

def evaluate_surrogate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * len(x)
    return val_loss / len(loader.dataset)

# =========================================================================
# MAIN EXECUTION PIPELINE
# =========================================================================
def main():
    root_dir = SCRIPT_DIR.parent
    data_dir = root_dir / "data"
    output_dir = root_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xfoil_path = data_dir / "dataset.csv"
    unifoil_path = data_dir / "unifoil_cst_dataset.csv"
    uiuc_cst_path = data_dir / "uiuc_cst_dataset.csv"
    uiuc_polar_path = data_dir / "uiuc_polar_dataset.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========================================================================")
    print(f"  DIPAS: EJECUCIÓN INTEGRAL DE EXPERIMENTOS Y SURROGATES (FASE XFOIL)      ")
    print(f"  Dispositivo: {device}")
    print(f"==========================================================================\n")
    
    # -------------------------------------------------------------
    # 1. Carga y partición de datasets
    # -------------------------------------------------------------
    print(">> [1/6] Preparando datasets y particiones...")
    batch_size = 64
    
    # Dataset XFOIL para CVAE (muestra fija de 10.000 para consistencia)
    full_xfoil_cvae = AirfoilCSTDataset(str(xfoil_path), use_conditions=True, sample_size=10000)
    train_size = int(0.9 * len(full_xfoil_cvae))
    val_size = len(full_xfoil_cvae) - train_size
    
    cvae_train_xfoil, cvae_val_xfoil = random_split(
        full_xfoil_cvae, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    cvae_train_loader = DataLoader(cvae_train_xfoil, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Datasets incondicionales para Pre-entrenamiento
    unifoil_dataset = AirfoilCSTDataset(str(unifoil_path), use_conditions=False)
    unifoil_loader = DataLoader(unifoil_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    uiuc_dataset = AirfoilCSTDataset(str(uiuc_cst_path), use_conditions=False)
    uiuc_loader = DataLoader(uiuc_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"   XFOIL CVAE: {train_size} train / {val_size} val")
    print(f"   UniFoil Shapes: {len(unifoil_dataset)} perfiles")
    print(f"   UIUC Shapes: {len(uiuc_dataset)} perfiles")
    
    # -------------------------------------------------------------
    # 2. Experimento 1: DIPAS Base (Entrenamiento desde Cero)
    # -------------------------------------------------------------
    print("\n==========================================================================")
    print(">> [2/6] Entrenando CVAE Experimento 1: DIPAS Base (Desde Cero)...")
    print("==========================================================================")
    model1 = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    opt1 = optim.Adam(model1.parameters(), lr=0.001)
    
    for epoch in range(1, 101):
        model1.train()
        loss_epoch = 0
        for bx, bc in cvae_train_loader:
            bx, bc = bx.to(device), bc.to(device)
            opt1.zero_grad()
            rx, mu, logvar = model1(bx, bc)
            loss, _, _, _ = cvae_loss_function(rx, bx, mu, logvar, model1.cst_to_coord, beta=0.01, geom_weight=20.0)
            loss.backward()
            opt1.step()
            loss_epoch += loss.item() * len(bx)
        loss_epoch /= len(cvae_train_loader.dataset)
        if epoch % 25 == 0 or epoch == 1:
            print(f"   Epoch {epoch:03d}/100 | Loss: {loss_epoch:.5f}")
            
    torch.save(model1.state_dict(), output_dir / "dipas_base_model.pth")
    print(f"   [OK] Modelo Exp 1 guardado en: outputs/dipas_base_model.pth")
    
    # -------------------------------------------------------------
    # 3. Experimento 2: Transfer Learning (UniFoil -> XFOIL)
    # -------------------------------------------------------------
    print("\n==========================================================================")
    print(">> [3/6] Entrenando CVAE Experimento 2: Transfer Learning (UniFoil -> XFOIL)...")
    print("==========================================================================")
    model2 = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    opt2 = optim.Adam(model2.parameters(), lr=0.001)
    
    # Fase A: UniFoil Pretrain (80 épocas incondicional)
    print("   [Fase A] Pre-entrenamiento geométrico con 10k UniFoil...")
    for epoch in range(1, 81):
        model2.train()
        loss_epoch = 0
        for bx, _ in unifoil_loader:
            bx = bx.to(device)
            bc = torch.zeros((bx.size(0), 4), dtype=torch.float32).to(device)
            opt2.zero_grad()
            rx, mu, logvar = model2(bx, bc)
            loss, _, _, _ = cvae_loss_function(rx, bx, mu, logvar, model2.cst_to_coord, beta=0.01, geom_weight=20.0)
            loss.backward()
            opt2.step()
            loss_epoch += loss.item() * len(bx)
        loss_epoch /= len(unifoil_loader.dataset)
        if epoch % 40 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/80 | Loss UniFoil: {loss_epoch:.5f}")
            
    # Fase B: Fine-tuning condicional XFOIL (60 épocas)
    print("   [Fase B] Fine-tuning condicional con XFOIL...")
    for param_group in opt2.param_groups:
        param_group['lr'] = 0.0005
        
    for epoch in range(1, 61):
        model2.train()
        loss_epoch = 0
        for bx, bc in cvae_train_loader:
            bx, bc = bx.to(device), bc.to(device)
            opt2.zero_grad()
            rx, mu, logvar = model2(bx, bc)
            loss, _, _, _ = cvae_loss_function(rx, bx, mu, logvar, model2.cst_to_coord, beta=0.01, geom_weight=20.0)
            loss.backward()
            opt2.step()
            loss_epoch += loss.item() * len(bx)
        loss_epoch /= len(cvae_train_loader.dataset)
        if epoch % 30 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/60 | Loss XFOIL: {loss_epoch:.5f}")
            
    torch.save(model2.state_dict(), output_dir / "dipas_tl_model.pth")
    print(f"   [OK] Modelo Exp 2 guardado en: outputs/dipas_tl_model.pth")
    
    # -------------------------------------------------------------
    # 4. Experimento 3: TL + UIUC (UniFoil -> UIUC -> XFOIL)
    # -------------------------------------------------------------
    print("\n==========================================================================")
    print(">> [4/6] Entrenando CVAE Experimento 3: TL + UIUC (UniFoil -> UIUC -> XFOIL)...")
    print("==========================================================================")
    model3 = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    opt3 = optim.Adam(model3.parameters(), lr=0.001)
    
    # Fase A: UniFoil Pretrain (80 épocas)
    print("   [Fase A] Pre-entrenamiento con UniFoil...")
    for epoch in range(1, 81):
        model3.train()
        loss_epoch = 0
        for bx, _ in unifoil_loader:
            bx = bx.to(device)
            bc = torch.zeros((bx.size(0), 4), dtype=torch.float32).to(device)
            opt3.zero_grad()
            rx, mu, logvar = model3(bx, bc)
            loss, _, _, _ = cvae_loss_function(rx, bx, mu, logvar, model3.cst_to_coord, beta=0.01, geom_weight=20.0)
            loss.backward()
            opt3.step()
            loss_epoch += loss.item() * len(bx)
        loss_epoch /= len(unifoil_loader.dataset)
        
    # Fase B: Adaptación UIUC (50 épocas)
    print("   [Fase B] Adaptación con UIUC experimental...")
    for param_group in opt3.param_groups:
        param_group['lr'] = 0.0005
    for epoch in range(1, 51):
        model3.train()
        loss_epoch = 0
        for bx, _ in uiuc_loader:
            bx = bx.to(device)
            bc = torch.zeros((bx.size(0), 4), dtype=torch.float32).to(device)
            opt3.zero_grad()
            rx, mu, logvar = model3(bx, bc)
            loss, _, _, _ = cvae_loss_function(rx, bx, mu, logvar, model3.cst_to_coord, beta=0.01, geom_weight=20.0)
            loss.backward()
            opt3.step()
            loss_epoch += loss.item() * len(bx)
        loss_epoch /= len(uiuc_loader.dataset)
        if epoch % 25 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/50 | Loss UIUC: {loss_epoch:.5f}")
            
    # Fase C: Fine-tuning XFOIL (60 épocas)
    print("   [Fase C] Fine-tuning condicional con XFOIL...")
    for epoch in range(1, 61):
        model3.train()
        loss_epoch = 0
        for bx, bc in cvae_train_loader:
            bx, bc = bx.to(device), bc.to(device)
            opt3.zero_grad()
            rx, mu, logvar = model3(bx, bc)
            loss, _, _, _ = cvae_loss_function(rx, bx, mu, logvar, model3.cst_to_coord, beta=0.01, geom_weight=20.0)
            loss.backward()
            opt3.step()
            loss_epoch += loss.item() * len(bx)
        loss_epoch /= len(cvae_train_loader.dataset)
        if epoch % 30 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/60 | Loss XFOIL: {loss_epoch:.5f}")
            
    torch.save(model3.state_dict(), output_dir / "dipas_tl_exp_model.pth")
    print(f"   [OK] Modelo Exp 3 guardado en: outputs/dipas_tl_exp_model.pth")
    
    # -------------------------------------------------------------
    # 5. Entrenamiento de los Surrogates (Numérico e Híbrido)
    # -------------------------------------------------------------
    print("\n==========================================================================")
    print(">> [5/6] Entrenando Modelos Surrogates (Numérico e Híbrido)...")
    print("==========================================================================")
    df_xfoil_full = pd.read_csv(xfoil_path)
    df_uiuc_polar = pd.read_csv(uiuc_polar_path)
    
    cst_cols = [f"au{i}" for i in range(6)] + [f"al{i}" for i in range(6)]
    input_cols = cst_cols + ["alpha", "reynolds"]
    output_cols = ["cl", "cd"]
    
    surr_xfoil_dataset = TabularAeroDataset(df_xfoil_full, input_cols, output_cols)
    base_scaler = surr_xfoil_dataset.get_scaler_dict()
    
    with open(output_dir / "surrogate_scalers.json", "w", encoding="utf-8") as f:
        json.dump(base_scaler, f, indent=4)
        
    surr_uiuc_dataset = TabularAeroDataset(df_uiuc_polar, input_cols, output_cols, scaler=base_scaler)
    
    surr_xfoil_loader = DataLoader(surr_xfoil_dataset, batch_size=128, shuffle=True)
    surr_uiuc_loader = DataLoader(surr_uiuc_dataset, batch_size=128, shuffle=True)
    
    crit = nn.MSELoss()
    
    # Surrogate 1 (Numérico: XFOIL)
    print("\n   [Surrogate 1: Numérico] Entrenando sobre 146.916 muestras de XFOIL...")
    surr_num = AerodynamicSurrogate().to(device)
    opt_s1 = optim.Adam(surr_num.parameters(), lr=0.001)
    for ep in range(1, 31):
        loss_ep = train_surrogate_epoch(surr_num, surr_xfoil_loader, crit, opt_s1, device)
        if ep % 10 == 0 or ep == 1:
            print(f"   Epoch {ep:02d}/30 | Loss: {loss_ep:.5f}")
    torch.save(surr_num.state_dict(), output_dir / "surrogate_numeric.pth")
    print("   [OK] Surrogate Numerico guardado.")
    
    # Surrogate 2 (Híbrido: XFOIL -> UIUC)
    print("\n   [Surrogate 2: Hibrido] XFOIL -> UIUC Tunel Experimental...")
    surr_hyb = AerodynamicSurrogate().to(device)
    opt_s2 = optim.Adam(surr_hyb.parameters(), lr=0.001)
    for ep in range(1, 31):
        train_surrogate_epoch(surr_hyb, surr_xfoil_loader, crit, opt_s2, device)
        
    print("   Sintonizando con datos experimentales UIUC...")
    for param_group in opt_s2.param_groups:
        param_group['lr'] = 0.0005
    for ep in range(1, 31):
        loss_ep = train_surrogate_epoch(surr_hyb, surr_uiuc_loader, crit, opt_s2, device)
        if ep % 10 == 0 or ep == 1:
            print(f"   Epoch {ep:02d}/30 | Loss UIUC: {loss_ep:.5f}")
    torch.save(surr_hyb.state_dict(), output_dir / "surrogate_hybrid.pth")
    print("   [OK] Surrogate Hibrido guardado.")
    
    # -------------------------------------------------------------
    # 6. Evaluación Geométrica y Aerodinámica Comparativa
    # -------------------------------------------------------------
    print("\n==========================================================================")
    print(">> [6/6] Ejecutando Evaluación y Comparativa Unificada...")
    print("==========================================================================")
    
    models = {
        "Exp 1 (DIPAS Base)": model1,
        "Exp 2 (Transfer Learning)": model2,
        "Exp 3 (TL + UIUC)": model3
    }
    
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    results_summary = {}
    
    for name, m in models.items():
        m.eval()
        rmse_l, mae_l, dtc_l, dxtc_l = [], [], [], []
        with torch.no_grad():
            for i in range(len(cvae_val_xfoil)):
                xr_t, cr_t = cvae_val_xfoil[i]
                xr = xr_t.numpy()
                rx_t, _, _ = m(xr_t.unsqueeze(0).to(device), cr_t.unsqueeze(0).to(device))
                rx = rx_t.squeeze(0).cpu().numpy()
                
                tcr, xtcr, xpts, yu_r, yl_r = calculate_thickness_properties(cst, xr)
                tcrec, xtcrec, _, yu_rec, yl_rec = calculate_thickness_properties(cst, rx)
                
                yr = np.concatenate([yu_r, yl_r])
                yrec = np.concatenate([yu_rec, yl_rec])
                
                rmse_l.append(np.sqrt(np.mean((yr - yrec)**2)))
                mae_l.append(np.mean(np.abs(yr - yrec)))
                dtc_l.append(abs(tcr - tcrec))
                dxtc_l.append(abs(xtcr - xtcrec))
                
        results_summary[name] = {
            "RMSE": np.mean(rmse_l),
            "MAE": np.mean(mae_l),
            "Delta_tc": np.mean(dtc_l),
            "Delta_xtc": np.mean(dxtc_l)
        }
        
    print("\n==========================================================================")
    print("          TABLA COMPARATIVA DE RECONSTRUCCIÓN GEOMÉTRICA (XFOIL TEST)     ")
    print("==========================================================================")
    for k, v in results_summary.items():
        print(f" {k:<26} | RMSE: {v['RMSE']:.6f} | MAE: {v['MAE']:.6f} | Delta(t/c): {v['Delta_tc']:.6f} | Delta(xt/c): {v['Delta_xtc']:.6f}")
    print("==========================================================================\n")
    
    # -------------------------------------------------------------
    # 7. Graficar Comparativa de Perfiles Extremos (Superposición de los 3)
    # -------------------------------------------------------------
    targets = {
        "Fácil/Estándar": "clarky",
        "Muy Curvado (Camber)": "s3021",
        "Grueso": "goe428",
        "Fino": "ag03",
        "No Convencional": "fx60126"
    }
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    df_raw = full_xfoil_cvae.df
    
    for idx, (label, seed_name) in enumerate(targets.items()):
        row = df_raw[df_raw["semilla"] == seed_name]
        if row.empty:
            row_idx = idx * 100
            row = df_raw.iloc[[row_idx]]
            seed_name = row["semilla"].values[0]
        else:
            row_idx = row.index[0]
            row = df_raw.loc[[row_idx]]
            
        real_cst = row[cst_cols].values[0].astype(np.float32)
        real_tensor = torch.tensor(real_cst).unsqueeze(0).to(device)
        cond_tensor = full_xfoil_cvae.cond_data[row_idx].unsqueeze(0).to(device)
        
        tc_r, xtc_r, xpts, yu_r, yl_r = calculate_thickness_properties(cst, real_cst)
        
        ax = axes[idx]
        ax.plot(xpts, yu_r, 'k-', label=f"Real ({seed_name})", linewidth=2.5)
        ax.plot(xpts, yl_r, 'k-', linewidth=2.5)
        
        colors = {"Exp 1 (DIPAS Base)": "blue", "Exp 2 (Transfer Learning)": "red", "Exp 3 (TL + UIUC)": "green"}
        styles = {"Exp 1 (DIPAS Base)": ":", "Exp 2 (Transfer Learning)": "--", "Exp 3 (TL + UIUC)": "-."}
        
        for m_name, m_model in models.items():
            with torch.no_grad():
                rec_t, _, _ = m_model(real_tensor, cond_tensor)
                rec_cst = rec_t.squeeze(0).cpu().numpy()
            _, _, _, yu_rec, yl_rec = calculate_thickness_properties(cst, rec_cst)
            ax.plot(xpts, yu_rec, color=colors[m_name], linestyle=styles[m_name], label=m_name, linewidth=1.5)
            ax.plot(xpts, yl_rec, color=colors[m_name], linestyle=styles[m_name], linewidth=1.5)
            
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Categoría: {label} (Perfil semilla: {seed_name})", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        
    axes[-1].set_xlabel("x/c")
    plt.tight_layout()
    comparison_plot_path = output_dir / "all_experiments_comparison_extremes.png"
    plt.savefig(comparison_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f">> Gráfico comparativo de perfiles extremos guardado en: {comparison_plot_path.resolve()}")
    
    print("\n>> Flujo integral completado con exito!")

if __name__ == "__main__":
    main()
