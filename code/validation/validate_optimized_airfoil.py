# -*- coding: utf-8 -*-
"""
Pipeline de Validación Cruzada Automática:
Compara la predicción de la IA (Surrogate) frente a una simulación física real en XFOIL
para el perfil optimizado. Genera las polares completas de sustentación, arrastre y finura.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR.parent
ROOT_DIR = CODE_DIR.parent
sys.path.extend([str(CODE_DIR), str(SCRIPT_DIR)])

from xfoil_wrapper import XFoilWrapper
from surrogate_model import AerodynamicSurrogate
from cst_generator import CSTParametrization

def fit_cst_from_dat(dat_path, n_upper=6, n_lower=6, te_thickness=0.003):
    """
    Ajusta coeficientes CST a partir de un archivo .dat normalizado.
    """
    coords = []
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or any(c.isalpha() for c in line.replace("e", "").replace("E", "").replace("-", "").replace("+", "").replace(".", "")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
                    
    coords = np.array(coords)
    idx_le = np.argmin(coords[:, 0])
    
    upper = coords[:idx_le + 1] # Desde TE hasta LE por arriba
    lower = coords[idx_le:]     # Desde LE hasta TE por abajo
    upper = upper[::-1]
    
    theta = np.linspace(0, np.pi, 100)
    x_grid = 0.5 * (1.0 - np.cos(theta))
    
    y_u_interp = np.interp(x_grid, upper[:, 0], upper[:, 1])
    y_l_interp = np.interp(x_grid, lower[:, 0], lower[:, 1])
    
    cst = CSTParametrization(n_coefs_upper=n_upper, n_coefs_lower=n_lower, te_thickness=te_thickness)
    cst_u, cst_l = cst.fit_airfoil(x_grid, y_u_interp, y_l_interp)
    cst_coefs = np.concatenate([cst_u, cst_l])
    
    return cst_coefs, x_grid, y_u_interp, y_l_interp

def run_cross_validation(dat_path, reynolds=150000.0, alpha_start=0.0, alpha_end=8.0, alpha_step=1.0,
                         name="uav_mission_test"):
    root_dir = SCRIPT_DIR.parent
    output_dir = root_dir / "outputs" / "optimized_airfoils"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n==========================================================================")
    print(f"  VALIDACIÓN CRUZADA AUTOMÁTICA (SURROGATE vs XFOIL)")
    print(f"  Perfil: {dat_path.name} | Re = {reynolds:,.0f} | Alphas: [{alpha_start}°, {alpha_end}°]")
    print(f"==========================================================================")
    
    # 1. Extraer CST y Coordenadas del archivo .dat
    print(">> [1/4] Cargando geometría y ajustando parámetros CST...")
    cst_coefs, x_grid, y_up, y_low = fit_cst_from_dat(dat_path)
    
    # 2. Ejecutar Simulación Física Real en XFOIL
    print(">> [2/4] Ejecutando simulación física real en XFOIL...")
    xfoil = XFoilWrapper()
    raw_results = xfoil.run_simulation(
        x=x_grid,
        y_upper=y_up,
        y_lower=y_low,
        reynolds=reynolds,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_step=alpha_step,
        file_prefix="temp_xfoil_val"
    )
    
    if raw_results is None or len(raw_results["alpha"]) == 0:
        print("Error: XFOIL no pudo converger en la simulación.")
        return None
        
    df_xfoil = pd.DataFrame({
        "alpha": raw_results["alpha"],
        "cl": raw_results["CL"],
        "cd": raw_results["CD"],
        "cm": raw_results["CM"]
    })
    df_xfoil["ld"] = df_xfoil["cl"] / df_xfoil["cd"]
    print(f"   [OK] XFOIL convergió con éxito en {len(df_xfoil)} puntos de la polar.")
    
    # 3. Evaluar con el Modelo Surrogate en los MISMOS ángulos de ataque
    print(">> [3/4] Evaluando predicciones del Modelo Surrogate...")
    scalers_path = root_dir / "outputs" / "surrogate_scalers.json"
    surrogate_path = root_dir / "outputs" / "surrogate_numeric.pth"
    
    with open(scalers_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
        
    in_mean = np.array(scaler["input_mean"], dtype=np.float32)
    in_std = np.array(scaler["input_std"], dtype=np.float32)
    out_mean = np.array(scaler["output_mean"], dtype=np.float32)
    out_std = np.array(scaler["output_std"], dtype=np.float32)
    
    device = torch.device("cpu")
    surrogate = AerodynamicSurrogate()
    surrogate.load_state_dict(torch.load(surrogate_path, map_location=device))
    surrogate.eval()
    
    surr_cl, surr_cd = [], []
    for a in df_xfoil["alpha"].values:
        in_raw = np.concatenate([cst_coefs, [a, reynolds]]).astype(np.float32)
        in_norm = torch.tensor((in_raw - in_mean) / in_std, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_norm = surrogate(in_norm).squeeze(0).numpy()
        p_real = (p_norm * out_std) + out_mean
        surr_cl.append(p_real[0])
        surr_cd.append(p_real[1])
        
    df_xfoil["cl_surrogate"] = surr_cl
    df_xfoil["cd_surrogate"] = surr_cd
    df_xfoil["ld_surrogate"] = df_xfoil["cl_surrogate"] / df_xfoil["cd_surrogate"]
    
    # 4. Cálculo de Métricas de Fidelidad
    rmse_cl = np.sqrt(np.mean((df_xfoil["cl"] - df_xfoil["cl_surrogate"])**2))
    rmse_cd = np.sqrt(np.mean((df_xfoil["cd"] - df_xfoil["cd_surrogate"])**2))
    max_ld_xfoil = df_xfoil["ld"].max()
    alpha_opt_xfoil = df_xfoil.loc[df_xfoil["ld"].idxmax(), "alpha"]
    
    print("\n==========================================================================")
    print("                RESULTADOS DE LA VALIDACIÓN CRUZADA                       ")
    print("==========================================================================")
    print(f" RMSE en Sustentación (Cl): {rmse_cl:.4f}")
    print(f" RMSE en Arrastre (Cd):      {rmse_cd:.5f}")
    print(f" Máxima Finura XFOIL Real:   (L/D)_max = {max_ld_xfoil:.1f} @ Alpha = {alpha_opt_xfoil:.1f}°")
    print("==========================================================================\n")
    
    print(f"{'Alpha':<6} | {'Cl XFOIL':<9} | {'Cl Surr':<8} | {'Cd XFOIL':<9} | {'Cd Surr':<8} | {'L/D XFOIL':<10} | {'L/D Surr'}")
    print("-" * 75)
    for _, r in df_xfoil.iterrows():
        print(f"{r['alpha']:<6.1f} | {r['cl']:<9.4f} | {r['cl_surrogate']:<8.4f} | {r['cd']:<9.5f} | {r['cd_surrogate']:<8.5f} | {r['ld']:<10.1f} | {r['ld_surrogate']:.1f}")
    print("-" * 75)
    
    # 5. Generar Panel Gráfico Comparativo de 4 Cuadrantes
    print("\n>> [4/4] Generando panel de curvas polares comparativas...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    # Cuadrante 1: Cl vs Alpha
    ax1 = axes[0, 0]
    ax1.plot(df_xfoil["alpha"], df_xfoil["cl"], 'b-o', label="XFOIL (Simulación Real)", linewidth=2.0)
    ax1.plot(df_xfoil["alpha"], df_xfoil["cl_surrogate"], 'r--', label="Surrogate (Predicción IA)", linewidth=2.0)
    ax1.set_title("Curva de Sustentación ($C_L$ vs $\\alpha$)", fontsize=11)
    ax1.set_xlabel("Ángulo de ataque $\\alpha$ (grados)")
    ax1.set_ylabel("Coeficiente de Sustentación $C_L$")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cuadrante 2: Cd vs Alpha
    ax2 = axes[0, 1]
    ax2.plot(df_xfoil["alpha"], df_xfoil["cd"], 'b-o', label="XFOIL (Simulación Real)", linewidth=2.0)
    ax2.plot(df_xfoil["alpha"], df_xfoil["cd_surrogate"], 'r--', label="Surrogate (Predicción IA)", linewidth=2.0)
    ax2.set_title("Curva de Arrastre ($C_D$ vs $\\alpha$)", fontsize=11)
    ax2.set_xlabel("Ángulo de ataque $\\alpha$ (grados)")
    ax2.set_ylabel("Coeficiente de Arrastre $C_D$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Cuadrante 3: Eficiencia L/D vs Alpha
    ax3 = axes[1, 0]
    ax3.plot(df_xfoil["alpha"], df_xfoil["ld"], 'b-o', label="XFOIL (Simulación Real)", linewidth=2.0)
    ax3.plot(df_xfoil["alpha"], df_xfoil["ld_surrogate"], 'r--', label="Surrogate (Predicción IA)", linewidth=2.0)
    ax3.axvline(x=alpha_opt_xfoil, color='g', linestyle=':', label=f"Óptimo Real ({max_ld_xfoil:.1f} @ {alpha_opt_xfoil}°)")
    ax3.set_title("Finura Aerodinámica ($L/D$ vs $\\alpha$)", fontsize=11)
    ax3.set_xlabel("Ángulo de ataque $\\alpha$ (grados)")
    ax3.set_ylabel("Eficiencia $L/D = C_L / C_D$")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Cuadrante 4: Polar de Arrastre Cl vs Cd
    ax4 = axes[1, 1]
    ax4.plot(df_xfoil["cd"], df_xfoil["cl"], 'b-o', label="XFOIL (Simulación Real)", linewidth=2.0)
    ax4.plot(df_xfoil["cd_surrogate"], df_xfoil["cl_surrogate"], 'r--', label="Surrogate (Predicción IA)", linewidth=2.0)
    ax4.set_title("Polar de Arrastre ($C_L$ vs $C_D$)", fontsize=11)
    ax4.set_xlabel("Coeficiente de Arrastre $C_D$")
    ax4.set_ylabel("Coeficiente de Sustentación $C_L$")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle(f"Validación Cruzada Aerodinámica: {name} (Re = {reynolds:,.0f})\nRMSE Cl: {rmse_cl:.4f} | RMSE Cd: {rmse_cd:.5f}", fontsize=13, weight="bold")
    plt.tight_layout()
    
    plot_path = output_dir / f"validation_{name}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Guardar tabla de datos
    csv_path = output_dir / f"validation_{name}.csv"
    df_xfoil.to_csv(csv_path, index=False)
    print(f">> Gráfico de validación cruzada guardado en: {plot_path.resolve()}")
    print(f">> Tabla de polares guardada en: {csv_path.resolve()}")
    
    return df_xfoil

def main():
    parser = argparse.ArgumentParser(description="Validación Cruzada XFOIL vs Surrogate para Perfiles Optimizados")
    parser.add_argument("--dat", type=str, default="outputs/optimized_airfoils/uav_mission_test.dat", help="Ruta al archivo .dat del perfil")
    parser.add_argument("--reynolds", type=float, default=150000.0, help="Número de Reynolds")
    parser.add_argument("--alpha_start", type=float, default=0.0, help="Ángulo de ataque inicial (grados)")
    parser.add_argument("--alpha_end", type=float, default=8.0, help="Ángulo de ataque final (grados)")
    parser.add_argument("--alpha_step", type=float, default=1.0, help="Paso de ángulo de ataque")
    parser.add_argument("--name", type=str, default="uav_mission_test", help="Nombre para exportar resultados")
    
    args = parser.parse_args()
    
    root_dir = SCRIPT_DIR.parent
    dat_file = root_dir / args.dat if not Path(args.dat).is_absolute() else Path(args.dat)
    
    run_cross_validation(
        dat_path=dat_file,
        reynolds=args.reynolds,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_step=args.alpha_step,
        name=args.name
    )

if __name__ == "__main__":
    main()
