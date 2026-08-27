# -*- coding: utf-8 -*-
"""
Evaluación de Consistencia Aerodinámica:
Testea los 3 modelos generativos CVAE pasándoles objetivos de diseño específicos
(CL, CD, Re, t/c) y evaluando los perfiles generados a través de los Surrogates.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

from cvae_model import CVAE
from surrogate_model import AerodynamicSurrogate
from cvae_dataset import AirfoilCSTDataset, calculate_tc_vectorized
from cst_generator import CSTParametrization
from evaluate_reconstruction import calculate_thickness_properties

def main():
    root_dir = SCRIPT_DIR.parent
    data_dir = root_dir / "data"
    output_dir = root_dir / "outputs"
    
    xfoil_path = data_dir / "dataset.csv"
    scalers_path = output_dir / "surrogate_scalers.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Evaluando consistencia aerodinámica de los 3 CVAEs...")
    
    # 1. Cargar Scalers y CVAE Dataset (para normalizar condiciones)
    with open(scalers_path, "r", encoding="utf-8") as f:
        surr_scaler = json.load(f)
        
    cvae_ds = AirfoilCSTDataset(str(xfoil_path), use_conditions=True, sample_size=5000)
    cond_min = cvae_ds.cond_min
    cond_max = cvae_ds.cond_max
    
    # 2. Cargar los 3 CVAEs
    m1 = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    m1.load_state_dict(torch.load(output_dir / "dipas_base_model.pth", map_location=device))
    m1.eval()
    
    m2 = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    m2.load_state_dict(torch.load(output_dir / "dipas_tl_model.pth", map_location=device))
    m2.eval()
    
    m3 = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(device)
    m3.load_state_dict(torch.load(output_dir / "dipas_tl_exp_model.pth", map_location=device))
    m3.eval()
    
    cvae_models = {
        "Exp 1 (DIPAS Base)": m1,
        "Exp 2 (Transfer Learning)": m2,
        "Exp 3 (TL + UIUC)": m3
    }
    
    # 3. Cargar los 2 Surrogates
    s_num = AerodynamicSurrogate().to(device)
    s_num.load_state_dict(torch.load(output_dir / "surrogate_numeric.pth", map_location=device))
    s_num.eval()
    
    s_hyb = AerodynamicSurrogate().to(device)
    s_hyb.load_state_dict(torch.load(output_dir / "surrogate_hybrid.pth", map_location=device))
    s_hyb.eval()
    
    surrogates = {
        "Surrogate Numerico": s_num,
        "Surrogate Hibrido (Exp)": s_hyb
    }
    
    # 4. Definir Objetivos de Diseño Aerodinámico de Ensayo (UAV / Bajo Reynolds)
    design_targets = [
        {"desc": "Crucero Eficiente", "cl": 0.50, "cd": 0.012, "reynolds": 150000.0, "t_c": 0.11, "alpha": 2.0},
        {"desc": "Alta Sustentacion / Ascenso", "cl": 0.85, "cd": 0.018, "reynolds": 150000.0, "t_c": 0.12, "alpha": 5.0},
        {"desc": "Bajo Arrastre / Alta Velocidad", "cl": 0.30, "cd": 0.009, "reynolds": 200000.0, "t_c": 0.09, "alpha": 0.0},
        {"desc": "Régimen Critico (Re bajo)", "cl": 0.65, "cd": 0.016, "reynolds": 100000.0, "t_c": 0.10, "alpha": 3.5},
    ]
    
    cst_builder = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    results = []
    
    fig, axes = plt.subplots(len(design_targets), 1, figsize=(11, 14), sharex=True)
    
    in_mean = np.array(surr_scaler["input_mean"], dtype=np.float32)
    in_std = np.array(surr_scaler["input_std"], dtype=np.float32)
    out_mean = np.array(surr_scaler["output_mean"], dtype=np.float32)
    out_std = np.array(surr_scaler["output_std"], dtype=np.float32)
    
    for t_idx, target in enumerate(design_targets):
        # Normalizar condición para el CVAE
        raw_cond = pd.Series({
            "reynolds": target["reynolds"],
            "cl": target["cl"],
            "cd": target["cd"],
            "t_c": target["t_c"]
        })
        denom = (cond_max - cond_min)
        denom[denom == 0] = 1.0
        norm_cond = (raw_cond - cond_min) / denom
        c_tensor = torch.tensor(norm_cond.values, dtype=torch.float32).unsqueeze(0).to(device)
        
        ax = axes[t_idx]
        ax.set_title(f"Caso {t_idx+1}: {target['desc']} | Objetivo: Cl={target['cl']:.2f}, Cd={target['cd']:.3f} (L/D={target['cl']/target['cd']:.1f}) @ Re={int(target['reynolds'])}, t/c={target['t_c']:.2f}", fontsize=10)
        
        colors = {"Exp 1 (DIPAS Base)": "blue", "Exp 2 (Transfer Learning)": "red", "Exp 3 (TL + UIUC)": "green"}
        styles = {"Exp 1 (DIPAS Base)": ":", "Exp 2 (Transfer Learning)": "-", "Exp 3 (TL + UIUC)": "--"}
        
        for cvae_name, cvae_m in cvae_models.items():
            # Generar geometría condicionada usando el centro del espacio latente (z = 0)
            z_zero = torch.zeros((1, 6), dtype=torch.float32).to(device)
            with torch.no_grad():
                gen_cst_t = cvae_m.decode(z_zero, c_tensor)
                gen_cst = gen_cst_t.squeeze(0).cpu().numpy()
                
            # Evaluar espesor real generado
            tc_real, xtc_real, xpts, yu, yl = calculate_thickness_properties(cst_builder, gen_cst)
            
            # Graficar perfil
            ax.plot(xpts, yu, color=colors[cvae_name], linestyle=styles[cvae_name], label=f"{cvae_name} (t/c={tc_real:.3f})", linewidth=1.8)
            ax.plot(xpts, yl, color=colors[cvae_name], linestyle=styles[cvae_name], linewidth=1.8)
            
            # Evaluar a través de los Surrogates
            surr_input_raw = np.concatenate([gen_cst, [target["alpha"], target["reynolds"]]]).astype(np.float32)
            surr_input_norm = (surr_input_raw - in_mean) / in_std
            s_in_tensor = torch.tensor(surr_input_norm, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_norm_num = s_num(s_in_tensor).squeeze(0).cpu().numpy()
                pred_norm_hyb = s_hyb(s_in_tensor).squeeze(0).cpu().numpy()
                
            pred_num = (pred_norm_num * out_std) + out_mean
            pred_hyb = (pred_norm_hyb * out_std) + out_mean
            
            cl_num, cd_num = pred_num[0], pred_num[1]
            cl_hyb, cd_hyb = pred_hyb[0], pred_hyb[1]
            
            results.append({
                "Caso": target["desc"],
                "Modelo CVAE": cvae_name,
                "CL_Obj": target["cl"],
                "CL_SurrNum": cl_num,
                "Delta_CL_Num": abs(target["cl"] - cl_num),
                "CD_Obj": target["cd"],
                "CD_SurrNum": cd_num,
                "Delta_CD_Num": abs(target["cd"] - cd_num),
                "L/D_Num": cl_num / max(cd_num, 1e-4),
                "CL_SurrHyb": cl_hyb,
                "CD_SurrHyb": cd_hyb,
                "L/D_Hyb": cl_hyb / max(cd_hyb, 1e-4),
                "t/c_Real": tc_real
            })
            
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        
    axes[-1].set_xlabel("x/c")
    plt.tight_layout()
    plot_save_path = output_dir / "aerodynamic_consistency_profiles.png"
    plt.savefig(plot_save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f">> Gráfico de consistencia aerodinámica guardado en: {plot_save_path.resolve()}")
    
    # Guardar tabla de resultados a CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "aerodynamic_consistency_results.csv", index=False)
    
    print("\n==========================================================================================================")
    print("                      TABLA DE CONSISTENCIA AERODINÁMICA (EVALUADA VÍA SURROGATES)                        ")
    print("==========================================================================================================")
    print(f"{'Caso':<22} | {'Modelo CVAE':<22} | {'CL Obj':<6} | {'CL Pred':<7} | {'CD Obj':<6} | {'CD Pred':<7} | {'L/D':<5} | {'t/c'}")
    print("----------------------------------------------------------------------------------------------------------")
    for _, r in res_df.iterrows():
        print(f"{r['Caso']:<22} | {r['Modelo CVAE']:<22} | {r['CL_Obj']:<6.2f} | {r['CL_SurrNum']:<7.3f} | {r['CD_Obj']:<6.3f} | {r['CD_SurrNum']:<7.4f} | {r['L/D_Num']:<5.1f} | {r['t/c_Real']:.3f}")
    print("==========================================================================================================\n")
    print(">> Evaluación de consistencia completada exitosamente.")

if __name__ == "__main__":
    main()
