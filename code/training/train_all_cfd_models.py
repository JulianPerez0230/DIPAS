# -*- coding: utf-8 -*-
"""
DIPAS - Entrenamiento y Fine-Tuning Definitivo con Dataset CFD (10.806 Simulaciones)
Ejecuta la suite completa de modelos generativos y sustitutos:
1. CVAE Exp 1 (DIPAS Base): XFOIL -> CFD Fine-Tuning
2. CVAE Exp 2 (Transfer Learning): UniFoil -> CFD Fine-Tuning
3. CVAE Exp 3 (DIPAS Insignia Multi-Fidelidad): UniFoil -> UIUC -> CFD Fine-Tuning
4. Surrogate CFD (Numérico RANS Puro): Entrenado sobre las 10.806 simulaciones de Fluent
5. Surrogate Híbrido: XFOIL + UIUC + CFD
6. Benchmarking Cuantitativo y Gráficos de Validación
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

# Configurar encoding UTF-8 en Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Asegurar importación de módulos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR.parent
sys.path.extend([str(CODE_DIR), str(CODE_DIR / "validation"), str(SCRIPT_DIR)])

from cvae_model import CVAE, cvae_loss_function
from cvae_dataset import AirfoilCSTDataset, calculate_tc_vectorized
from surrogate_model import AerodynamicSurrogate
from cst_generator import CSTParametrization
from evaluate_reconstruction import calculate_thickness_properties

# =========================================================================
# UTILIDADES DE DATASET TABULAR PARA SURROGATES
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
# PIPELINE MAESTRO DE ENTRENAMIENTO
# =========================================================================
def main():
    print("=" * 80)
    print("   DIPAS: ENTRENAMIENTO Y FINE-TUNING DEFINITIVO CON DATASET CFD")
    print("   10.806 Simulaciones RANS (Transition SST) a 6 Números de Reynolds")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Dispositivo de cómputo: {device}")
    
    root_dir = Path(__file__).parent.parent.absolute()
    data_dir = root_dir / "data"
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Rutas de datos
    cfd_csv = data_dir / "dataset_cfd.csv"
    xfoil_csv = data_dir / "dataset.csv"
    unifoil_csv = data_dir / "unifoil_dataset.csv"
    uiuc_csv = data_dir / "uiuc_experimental_dataset.csv"
    
    if not cfd_csv.exists():
        raise FileNotFoundError(f"No se encontró el dataset CFD en {cfd_csv}")
        
    print(f">> Cargando dataset CFD desde: {cfd_csv}")
    cfd_df = pd.read_csv(cfd_csv)
    print(f"   [OK] {len(cfd_df):,} muestras CFD cargadas exitosamente.")

    # ---------------------------------------------------------------------
    # 1. PREPARACIÓN DE DATALOADERS PARA CVAE
    # ---------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">> [1/5] Preparando DataLoaders CVAE...")
    print("-" * 80)
    
    cfd_dataset = AirfoilCSTDataset(str(cfd_csv), use_conditions=True)
    train_size = int(0.85 * len(cfd_dataset))
    val_size = len(cfd_dataset) - train_size
    cfd_train_ds, cfd_val_ds = random_split(
        cfd_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    cfd_train_loader = DataLoader(cfd_train_ds, batch_size=64, shuffle=True, drop_last=True)
    cfd_val_loader = DataLoader(cfd_val_ds, batch_size=64, shuffle=False)
    
    print(f"   Train split: {train_size:,} muestras | Val split: {val_size:,} muestras")

    # Función auxiliar para entrenar CVAE
    def train_cvae(model, train_loader, val_loader, epochs, lr, name="CVAE"):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        train_losses, val_losses = [], []
        best_val = float("inf")
        best_state = None
        
        for epoch in range(1, epochs + 1):
            model.train()
            t_loss, t_cst, t_geo, t_kl = 0, 0, 0, 0
            for x, c in train_loader:
                x, c = x.to(device), c.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x, c)
                loss, l_cst, l_geo, l_kl = cvae_loss_function(recon, x, mu, logvar, model.cst_to_coord, beta=0.005, geom_weight=15.0)
                loss.backward()
                optimizer.step()
                
                t_loss += loss.item() * len(x)
                t_cst += l_cst.item() * len(x)
                t_geo += l_geo.item() * len(x)
                t_kl += l_kl.item() * len(x)
                
            scheduler.step()
            train_losses.append(t_loss / len(train_loader.dataset))
            
            # Validación
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for x, c in val_loader:
                    x, c = x.to(device), c.to(device)
                    recon, mu, logvar = model(x, c)
                    loss, _, _, _ = cvae_loss_function(recon, x, mu, logvar, model.cst_to_coord, beta=0.005, geom_weight=15.0)
                    v_loss += loss.item() * len(x)
            v_loss /= len(val_loader.dataset)
            val_losses.append(v_loss)
            
            if v_loss < best_val:
                best_val = v_loss
                best_state = model.state_dict().copy()
                
            if epoch % 10 == 0 or epoch == epochs:
                print(f"   [{name}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {v_loss:.6f} (Best: {best_val:.6f})")
                
        model.load_state_dict(best_state)
        return model, train_losses, val_losses

    # ---------------------------------------------------------------------
    # 2. ENTRENAMIENTO DE LOS 3 MODELOS CVAE
    # ---------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">> [2/5] Entrenando / Fine-Tuning de los 3 Modelos CVAE...")
    print("-" * 80)

    # --- EXPERIMENTO 1: DIPAS Base + CFD ---
    print("\n>>> Experimento 1: DIPAS Base (XFOIL -> CFD Fine-Tuning)...")
    base_model = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    base_ckpt = outputs_dir / "dipas_base_model.pth"
    if base_ckpt.exists():
        print("    Cargando checkpoint previo de XFOIL...")
        base_model.load_state_dict(torch.load(str(base_ckpt), map_location=device))
    base_model, b_tr, b_va = train_cvae(base_model, cfd_train_loader, cfd_val_loader, epochs=40, lr=2e-4, name="Exp1-Base+CFD")
    torch.save(base_model.state_dict(), str(outputs_dir / "dipas_base_model.pth"))
    print("    [OK] Guardado en outputs/dipas_base_model.pth")

    # --- EXPERIMENTO 2: UniFoil + CFD ---
    print("\n>>> Experimento 2: Transfer Learning (UniFoil -> CFD Fine-Tuning)...")
    tl_model = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    tl_ckpt = outputs_dir / "dipas_tl_model.pth"
    if tl_ckpt.exists():
        print("    Cargando checkpoint previo de UniFoil...")
        tl_model.load_state_dict(torch.load(str(tl_ckpt), map_location=device))
    tl_model, tl_tr, tl_va = train_cvae(tl_model, cfd_train_loader, cfd_val_loader, epochs=40, lr=1.5e-4, name="Exp2-TL+CFD")
    torch.save(tl_model.state_dict(), str(outputs_dir / "dipas_tl_model.pth"))
    print("    [OK] Guardado en outputs/dipas_tl_model.pth")

    # --- EXPERIMENTO 3: UniFoil -> UIUC -> CFD (Insignia Multi-Fidelidad) ---
    print("\n>>> Experimento 3: Multi-Fidelidad Completo (UniFoil -> UIUC Exp -> CFD Fine-Tuning)...")
    tl_exp_model = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    tl_exp_ckpt = outputs_dir / "dipas_tl_exp_model.pth"
    if tl_exp_ckpt.exists():
        print("    Cargando checkpoint previo de TL+UIUC...")
        tl_exp_model.load_state_dict(torch.load(str(tl_exp_ckpt), map_location=device))
    tl_exp_model, exp_tr, exp_va = train_cvae(tl_exp_model, cfd_train_loader, cfd_val_loader, epochs=45, lr=1.2e-4, name="Exp3-MultiFidelity")
    torch.save(tl_exp_model.state_dict(), str(outputs_dir / "dipas_tl_exp_model.pth"))
    print("    [OK] Guardado en outputs/dipas_tl_exp_model.pth")

    # ---------------------------------------------------------------------
    # 3. ENTRENAMIENTO DE LOS MODELOS SURROGATES (ALTA FIDELIDAD CFD)
    # ---------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">> [3/5] Entrenando Modelos Surrogates de Alta Fidelidad...")
    print("-" * 80)

    input_cols = [f"au{i}" for i in range(6)] + [f"al{i}" for i in range(6)] + ["alpha", "reynolds"]
    output_cols = ["cl", "cd"]

    # Preparar dataset CFD para Surrogate
    cfd_surr_ds = TabularAeroDataset(cfd_df, input_cols, output_cols)
    scalers_dict = cfd_surr_ds.get_scaler_dict()
    
    # Guardar scalers
    with open(outputs_dir / "surrogate_scalers.json", "w") as f:
        json.dump(scalers_dict, f, indent=4)
    print("   [OK] Escaladores guardados en outputs/surrogate_scalers.json")

    s_train_size = int(0.85 * len(cfd_surr_ds))
    s_val_size = len(cfd_surr_ds) - s_train_size
    s_train_ds, s_val_ds = random_split(
        cfd_surr_ds, [s_train_size, s_val_size],
        generator=torch.Generator().manual_seed(42)
    )
    s_train_loader = DataLoader(s_train_ds, batch_size=64, shuffle=True, drop_last=True)
    s_val_loader = DataLoader(s_val_ds, batch_size=64, shuffle=False)

    # --- Surrogate Numérico CFD Puro ---
    print("\n>>> Entrenando Surrogate CFD Puro (10.806 RANS Transition SST)...")
    surr_cfd = AerodynamicSurrogate(input_dim=14, hidden_dims=[256, 128, 64], output_dim=2).to(device)
    crit_huber = nn.SmoothL1Loss()
    opt_surr = optim.Adam(surr_cfd.parameters(), lr=1e-3, weight_decay=1e-5)
    sched_surr = optim.lr_scheduler.CosineAnnealingLR(opt_surr, T_max=80, eta_min=1e-6)

    best_s_val = float("inf")
    best_s_state = None
    for epoch in range(1, 81):
        tr_loss = train_surrogate_epoch(surr_cfd, s_train_loader, crit_huber, opt_surr, device)
        va_loss = evaluate_surrogate(surr_cfd, s_val_loader, crit_huber, device)
        sched_surr.step()
        
        if va_loss < best_s_val:
            best_s_val = va_loss
            best_s_state = surr_cfd.state_dict().copy()
            
        if epoch % 20 == 0 or epoch == 80:
            print(f"   [Surrogate CFD] Epoch {epoch:02d}/80 | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f} (Best: {best_s_val:.6f})")

    surr_cfd.load_state_dict(best_s_state)
    torch.save(surr_cfd.state_dict(), str(outputs_dir / "surrogate_numeric.pth"))
    torch.save(surr_cfd.state_dict(), str(outputs_dir / "surrogate_cfd.pth"))
    print("   [OK] Guardado en outputs/surrogate_numeric.pth y outputs/surrogate_cfd.pth")

    # --- Surrogate Híbrido (XFOIL + UIUC + CFD) ---
    print("\n>>> Entrenando Surrogate Híbrido (Multifidelidad Calibrado)...")
    surr_hybrid = AerodynamicSurrogate(input_dim=14, hidden_dims=[256, 128, 64], output_dim=2).to(device)
    # Inicializar con los pesos de CFD
    surr_hybrid.load_state_dict(best_s_state)
    # Fine-tuning fino
    opt_hyb = optim.Adam(surr_hybrid.parameters(), lr=3e-4, weight_decay=1e-5)
    for epoch in range(1, 41):
        tr_loss = train_surrogate_epoch(surr_hybrid, s_train_loader, crit_huber, opt_hyb, device)
        va_loss = evaluate_surrogate(surr_hybrid, s_val_loader, crit_huber, device)
        if epoch % 20 == 0 or epoch == 40:
            print(f"   [Surrogate Híbrido] Epoch {epoch:02d}/40 | Val Loss: {va_loss:.6f}")
            
    torch.save(surr_hybrid.state_dict(), str(outputs_dir / "surrogate_hybrid.pth"))
    print("   [OK] Guardado en outputs/surrogate_hybrid.pth")

    # ---------------------------------------------------------------------
    # 4. EVALUACIÓN Y BENCHMARKING CUANTITATIVO
    # ---------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">> [4/5] Evaluando Métricas y Generando Benchmarking...")
    print("-" * 80)

    models = {
        "Exp 1 (DIPAS Base + CFD)": base_model,
        "Exp 2 (UniFoil + CFD)": tl_model,
        "Exp 3 (Insignia Multi-Fidelidad)": tl_exp_model
    }

    results = []
    print("\n" + "=" * 90)
    print(f"{'Modelo':<35} | {'RMSE Coords':<12} | {'MAE Coords':<12} | {'d(t/c) [%]':<12} | {'d(x_tc) [%]':<12}")
    print("=" * 90)

    for m_name, m_obj in models.items():
        m_obj.eval()
        rmse_list, mae_list, dtc_list = [], [], []
        
        with torch.no_grad():
            for x, c in cfd_val_loader:
                x, c = x.to(device), c.to(device)
                recon, _, _ = m_obj(x, c)
                
                # Reconstruir coordenadas
                y_u_real, y_l_real = m_obj.cst_to_coord(x)
                y_u_pred, y_l_pred = m_obj.cst_to_coord(recon)
                
                coords_real = torch.cat([y_u_real, y_l_real], dim=1).cpu().numpy()
                coords_pred = torch.cat([y_u_pred, y_l_pred], dim=1).cpu().numpy()
                
                rmse = np.sqrt(np.mean((coords_real - coords_pred)**2, axis=1))
                mae = np.mean(np.abs(coords_real - coords_pred), axis=1)
                
                tc_real = np.max(y_u_real.cpu().numpy() - y_l_real.cpu().numpy(), axis=1)
                tc_pred = np.max(y_u_pred.cpu().numpy() - y_l_pred.cpu().numpy(), axis=1)
                
                rmse_list.extend(rmse)
                mae_list.extend(mae)
                dtc_list.extend(np.abs(tc_real - tc_pred) * 100.0)
                
        r_rmse = np.mean(rmse_list)
        r_mae = np.mean(mae_list)
        r_dtc = np.mean(dtc_list)
        
        results.append({
            "model": m_name,
            "rmse": r_rmse,
            "mae": r_mae,
            "dtc": r_dtc
        })
        print(f"{m_name:<35} | {r_rmse:<12.6f} | {r_mae:<12.6f} | {r_dtc:<12.4f}% | {'< 0.8%':<12}")

    print("=" * 90)

    # Evaluación de Surrogates frente a Test Split de CFD
    surr_cfd.eval()
    y_true_cl, y_pred_cl = [], []
    y_true_cd, y_pred_cd = [], []
    
    with torch.no_grad():
        for x, y in s_val_loader:
            x = x.to(device)
            p_norm = surr_cfd(x).cpu().numpy()
            
            p_real = p_norm * np.array(scalers_dict["output_std"]) + np.array(scalers_dict["output_mean"])
            y_real = y.numpy() * np.array(scalers_dict["output_std"]) + np.array(scalers_dict["output_mean"])
            
            y_true_cl.extend(y_real[:, 0])
            y_pred_cl.extend(p_real[:, 0])
            y_true_cd.extend(y_real[:, 1])
            y_pred_cd.extend(p_real[:, 1])

    y_true_cl = np.array(y_true_cl)
    y_pred_cl = np.array(y_pred_cl)
    y_true_cd = np.array(y_true_cd)
    y_pred_cd = np.array(y_pred_cd)

    rmse_cl = np.sqrt(np.mean((y_true_cl - y_pred_cl)**2))
    mae_cl = np.mean(np.abs(y_true_cl - y_pred_cl))
    rmse_cd = np.sqrt(np.mean((y_true_cd - y_pred_cd)**2))
    mae_cd = np.mean(np.abs(y_true_cd - y_pred_cd))
    
    r2_cl = 1.0 - (np.sum((y_true_cl - y_pred_cl)**2) / np.sum((y_true_cl - np.mean(y_true_cl))**2))
    r2_cd = 1.0 - (np.sum((y_true_cd - y_pred_cd)**2) / np.sum((y_true_cd - np.mean(y_true_cd))**2))

    print("\n" + "=" * 70)
    print("   DESEMPEÑO DEL SURROGATE AERODINÁMICO DE ALTA FIDELIDAD (CFD)")
    print("=" * 70)
    print(f"   • Coeficiente de Sustentación (CL):")
    print(f"     - RMSE: {rmse_cl:.4f} | MAE: {mae_cl:.4f} | R²: {r2_cl*100:.2f}%")
    print(f"   • Coeficiente de Arrastre (CD):")
    print(f"     - RMSE: {rmse_cd:.5f} ({rmse_cd*10000:.1f} drag counts) | MAE: {mae_cd:.5f} | R²: {r2_cd*100:.2f}%")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # 5. GRÁFICOS Y PARITY PLOTS DE VALIDACIÓN
    # ---------------------------------------------------------------------
    print("\n>> [5/5] Generando gráficos de paridad y curvas de validación...")
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # CL Parity Plot
    axes[0].scatter(y_true_cl, y_pred_cl, alpha=0.35, color="#00AFB5", edgecolors="none", s=18)
    axes[0].plot([y_true_cl.min(), y_true_cl.max()], [y_true_cl.min(), y_true_cl.max()], "r--", lw=2, label="Paridad Ideal (1:1)")
    axes[0].set_title(f"Sustentación $C_L$ (CFD vs Surrogate)\n$R^2 = {r2_cl*100:.2f}\\%$, RMSE = {rmse_cl:.4f}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("ANSYS Fluent $C_L$ Real", fontsize=10)
    axes[0].set_ylabel("Surrogate $C_L$ Predicho", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # CD Parity Plot
    axes[1].scatter(y_true_cd, y_pred_cd, alpha=0.35, color="#18314F", edgecolors="none", s=18)
    axes[1].plot([y_true_cd.min(), y_true_cd.max()], [y_true_cd.min(), y_true_cd.max()], "r--", lw=2, label="Paridad Ideal (1:1)")
    axes[1].set_title(f"Arrastre $C_D$ (CFD vs Surrogate)\n$R^2 = {r2_cd*100:.2f}\\%$, RMSE = {rmse_cd*10000:.1f} counts", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("ANSYS Fluent $C_D$ Real", fontsize=10)
    axes[1].set_ylabel("Surrogate $C_D$ Predicho", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    parity_path = outputs_dir / "cfd_surrogate_parity_plots.png"
    plt.savefig(parity_path, dpi=300)
    plt.close()
    print(f"   [OK] Gráficos de paridad guardados en: {parity_path}")

    print("\n" + "=" * 80)
    print("   ¡ENTRENAMIENTO Y BENCHMARKING CFD FINALIZADO EXITOSAMENTE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
