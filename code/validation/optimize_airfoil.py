# -*- coding: utf-8 -*-
"""
Optimizador Aerodinámico en el Espacio Latente (Inverse Design Optimizer) - DIPAS.
Encuentra el vector latente z* óptimo usando Adam guiado por el Surrogate
para satisfacer requerimientos exactos de diseño (CL, CD, t/c) o maximizar la finura (L/D).
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
CODE_DIR = SCRIPT_DIR.parent
ROOT_DIR = CODE_DIR.parent
sys.path.extend([str(CODE_DIR), str(SCRIPT_DIR)])

from cvae_model import CVAE
from surrogate_model import AerodynamicSurrogate
from cvae_dataset import AirfoilCSTDataset
from cst_generator import CSTParametrization
from evaluate_reconstruction import calculate_thickness_properties

class LatentAirfoilOptimizer:
    def __init__(self, cvae_path=None, surrogate_path=None, scalers_path=None, device=None):
        """
        Inicializa el optimizador cargando los modelos y escaladores entrenados.
        """
        self.root_dir = ROOT_DIR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Rutas por defecto si no se especifican
        if cvae_path is None:
            cvae_path = self.root_dir / "outputs" / "dipas_tl_model.pth" # Usamos Exp 2 (Transfer Learning) por defecto
        if surrogate_path is None:
            surrogate_path = self.root_dir / "outputs" / "surrogate_numeric.pth"
        if scalers_path is None:
            scalers_path = self.root_dir / "outputs" / "surrogate_scalers.json"
            
        print(f">> Cargando modelos para optimización en {self.device}...")
        
        # 1. Cargar Escaladores del Surrogate
        with open(scalers_path, "r", encoding="utf-8") as f:
            self.scaler = json.load(f)
        self.in_mean = torch.tensor(self.scaler["input_mean"], dtype=torch.float32).to(self.device)
        self.in_std = torch.tensor(self.scaler["input_std"], dtype=torch.float32).to(self.device)
        self.out_mean = torch.tensor(self.scaler["output_mean"], dtype=torch.float32).to(self.device)
        self.out_std = torch.tensor(self.scaler["output_std"], dtype=torch.float32).to(self.device)
        
        # 2. Cargar Dataset de referencia para cotas de normalización del CVAE
        xfoil_dataset_path = self.root_dir / "data" / "dataset.csv"
        self.cvae_ds = AirfoilCSTDataset(str(xfoil_dataset_path), use_conditions=True, sample_size=2000)
        self.cond_min = torch.tensor(self.cvae_ds.cond_min.values, dtype=torch.float32).to(self.device)
        self.cond_max = torch.tensor(self.cvae_ds.cond_max.values, dtype=torch.float32).to(self.device)
        self.cond_denom = self.cond_max - self.cond_min
        self.cond_denom[self.cond_denom == 0] = 1.0
        
        # 3. Cargar CVAE Decodificador (congelado)
        self.cvae = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(self.device)
        self.cvae.load_state_dict(torch.load(cvae_path, map_location=self.device))
        self.cvae.eval()
        for param in self.cvae.parameters():
            param.requires_grad = False
            
        # 4. Cargar Surrogate Aerodinámico (congelado)
        self.surrogate = AerodynamicSurrogate().to(self.device)
        self.surrogate.load_state_dict(torch.load(surrogate_path, map_location=self.device))
        self.surrogate.eval()
        for param in self.surrogate.parameters():
            param.requires_grad = False
            
        self.cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
        print("   [OK] Modelos cargados y congelados exitosamente.")
        
    def optimize(self, target_cl=0.70, target_cd=None, target_tc=0.12, reynolds=150000.0, alpha=3.0,
                 maximize_ld=False, n_iters=100, lr=0.05, reg_lambda=0.01):
        """
        Ejecuta la optimización sobre el vector latente z.
        """
        print(f"\n==========================================================================")
        print(f"  INICIANDO OPTIMIZACIÓN LATENTE DIPAS")
        print(f"  Objetivo: Cl={target_cl}, t/c={target_tc*100:.1f}%, Re={reynolds:,.0f}, Alpha={alpha}° | Maximize L/D: {maximize_ld}")
        print(f"==========================================================================")
        
        # 1. Normalizar condiciones para el CVAE [reynolds, cl, cd, t_c]
        cd_init = target_cd if target_cd is not None else 0.015
        raw_cond = torch.tensor([reynolds, target_cl, cd_init, target_tc], dtype=torch.float32).to(self.device)
        norm_cond = ((raw_cond - self.cond_min) / self.cond_denom).unsqueeze(0)
        
        # 2. Inicializar el vector latente z como parámetro optimizable en z=0
        z = torch.zeros((1, 6), dtype=torch.float32, device=self.device, requires_grad=True)
        
        # 3. Inicializar el optimizador Adam sobre la variable z
        optimizer = optim.Adam([z], lr=lr)
        
        # Constantes de condición para el surrogate
        alpha_re_tensor = torch.tensor([alpha, reynolds], dtype=torch.float32, device=self.device)
        
        history = {
            "loss": [],
            "cl": [],
            "cd": [],
            "ld": [],
            "tc": []
        }
        
        # 4. Bucle de Optimización Continua
        for it in range(1, n_iters + 1):
            optimizer.zero_grad()
            
            # A. Decodificar perfil CST a partir de z y las condiciones
            cst_params = self.cvae.decode(z, norm_cond) # Shape: (1, 12)
            
            # B. Calcular coordenadas y espesor de forma diferenciable a través de la capa CST
            y_upper, y_lower = self.cvae.cst_to_coord(cst_params) # y_upper: (1, 100), y_lower: (1, 100)
            thickness = y_upper - y_lower
            tc_pred = torch.max(thickness) # Espesor máximo derivado
            
            # C. Pasar por el Surrogate para predecir Cl y Cd
            surr_in_raw = torch.cat([cst_params.squeeze(0), alpha_re_tensor], dim=0) # (14,)
            surr_in_norm = (surr_in_raw - self.in_mean) / self.in_std
            
            pred_norm = self.surrogate(surr_in_norm.unsqueeze(0)) # (1, 2)
            pred_real = (pred_norm * self.out_std) + self.out_mean
            cl_pred = pred_real[0, 0]
            cd_pred = pred_real[0, 1]
            
            # D. Función de Costo (Loss)
            # Penalización por espesor
            loss_tc = 200.0 * (tc_pred - target_tc)**2
            
            # Regularización en z para mantener suavidad gaussiana
            loss_reg = reg_lambda * torch.sum(z**2)
            
            if maximize_ld:
                # Maximizar L/D equivale a minimizar -(Cl / Cd)
                loss_aero = - (cl_pred / torch.clamp(cd_pred, min=1e-4)) + 50.0 * (cl_pred - target_cl)**2
            else:
                loss_aero = 100.0 * (cl_pred - target_cl)**2 + 500.0 * (cd_pred - target_cd)**2
                
            loss = loss_aero + loss_tc + loss_reg
            
            # E. Retropropagación del gradiente y ajuste de z
            loss.backward()
            optimizer.step()
            
            # Registro
            ld_val = (cl_pred / torch.clamp(cd_pred, min=1e-4)).item()
            history["loss"].append(loss.item())
            history["cl"].append(cl_pred.item())
            history["cd"].append(cd_pred.item())
            history["ld"].append(ld_val)
            history["tc"].append(tc_pred.item())
            
            if it % 20 == 0 or it == 1 or it == n_iters:
                print(f"  Iter {it:03d}/{n_iters:03d} | Loss: {loss.item():.4f} | Cl: {cl_pred.item():.3f} | Cd: {cd_pred.item():.4f} | L/D: {ld_val:.1f} | t/c: {tc_pred.item()*100:.2f}%")
                
        # 5. Extraer perfil óptimo definitivo
        with torch.no_grad():
            opt_cst = self.cvae.decode(z, norm_cond).squeeze(0).cpu().numpy()
            
        opt_tc, opt_xtc, x_pts, y_up, y_low = calculate_thickness_properties(self.cst, opt_cst)
        
        final_results = {
            "z_opt": z.detach().cpu().numpy().tolist()[0],
            "cst_params": opt_cst.tolist(),
            "target": {
                "cl": target_cl,
                "cd": target_cd,
                "tc": target_tc,
                "reynolds": reynolds,
                "alpha": alpha,
                "maximize_ld": maximize_ld
            },
            "predicted": {
                "cl": history["cl"][-1],
                "cd": history["cd"][-1],
                "ld": history["ld"][-1],
                "tc": opt_tc,
                "xtc": opt_xtc
            },
            "coordinates": {
                "x": x_pts.tolist(),
                "y_upper": y_up.tolist(),
                "y_lower": y_low.tolist()
            },
            "history": history
        }
        
        print("\n>> ¡Optimización completada con éxito!")
        print(f"   - Cl Final: {final_results['predicted']['cl']:.4f}")
        print(f"   - Cd Final: {final_results['predicted']['cd']:.5f}")
        print(f"   - L/D Final: {final_results['predicted']['ld']:.1f}")
        print(f"   - Espesor Relativo: {final_results['predicted']['tc']*100:.2f}% @ x/c={final_results['predicted']['xtc']*100:.1f}%")
        
        return final_results
        
    def export_results(self, results, name="optimized_airfoil"):
        """
        Exporta las coordenadas en formato .dat y guarda el gráfico de convergencia y forma.
        """
        export_dir = self.root_dir / "outputs" / "optimized_airfoils"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        dat_path = export_dir / f"{name}.dat"
        png_path = export_dir / f"{name}.png"
        
        # 1. Guardar archivo .dat (Formato estándar Selig / UIUC)
        x_pts = np.array(results["coordinates"]["x"])
        y_up = np.array(results["coordinates"]["y_upper"])
        y_low = np.array(results["coordinates"]["y_lower"])
        
        with open(dat_path, "w", encoding="utf-8") as f:
            f.write(f"DIPAS Optimized Airfoil - {name}\n")
            f.write(f"# Cl={results['predicted']['cl']:.3f}, Cd={results['predicted']['cd']:.4f}, L/D={results['predicted']['ld']:.1f}, t/c={results['predicted']['tc']*100:.1f}%\n")
            # De borde de fuga a borde de ataque por extradós
            for x, y in zip(reversed(x_pts), reversed(y_up)):
                f.write(f"  {x:.7f}   {y:.7f}\n")
            # De borde de ataque a borde de fuga por intradós (omitiendo x=0 repetido)
            for x, y in zip(x_pts[1:], y_low[1:]):
                f.write(f"  {x:.7f}   {y:.7f}\n")
                
        print(f">> Coordenadas .dat guardadas en: {dat_path.resolve()}")
        
        # 2. Guardar Gráfico Resumen de Optimización
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)
        
        # Panel A: Geometría del Perfil
        ax_geom = fig.add_subplot(gs[0, :])
        ax_geom.plot(x_pts, y_up, 'b-', linewidth=2.2, label="Extradós (Suction Side)")
        ax_geom.plot(x_pts, y_low, 'b-', linewidth=2.2, label="Intradós (Pressure Side)")
        # Línea de curvatura media (Camber line)
        camber = (y_up + y_low) / 2.0
        ax_geom.plot(x_pts, camber, 'k--', linewidth=1.2, label="Línea Media (Camber)")
        ax_geom.axvline(x=results["predicted"]["xtc"], color='gray', linestyle=':', label=f"t/c max ({results['predicted']['tc']*100:.1f}%)")
        ax_geom.set_aspect("equal", adjustable="box")
        ax_geom.set_title(f"Perfil Optimizado: {name}\n"
                          f"Cl={results['predicted']['cl']:.3f} | Cd={results['predicted']['cd']:.4f} | L/D={results['predicted']['ld']:.1f} | t/c={results['predicted']['tc']*100:.1f}% @ Re={int(results['target']['reynolds']):,}", fontsize=11)
        ax_geom.set_xlabel("x/c")
        ax_geom.set_ylabel("y/c")
        ax_geom.grid(True, alpha=0.3)
        ax_geom.legend(loc="upper right")
        
        # Panel B: Convergencia de Pérdida
        ax_loss = fig.add_subplot(gs[1, 0])
        ax_loss.plot(results["history"]["loss"], 'r-', linewidth=1.8)
        ax_loss.set_title("Convergencia de la Función Objetivo (Adam)")
        ax_loss.set_xlabel("Iteraciones")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)
        
        # Panel C: Evolución de Cl y L/D durante la optimización
        ax_ld = fig.add_subplot(gs[1, 1])
        ax_ld.plot(results["history"]["ld"], 'g-', linewidth=1.8, label="Finura L/D")
        ax_ld.set_title("Evolución de la Eficiencia Aerodinámica (L/D)")
        ax_ld.set_xlabel("Iteraciones")
        ax_ld.set_ylabel("L/D")
        ax_ld.grid(True, alpha=0.3)
        ax_ld.legend()
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f">> Gráfico de optimización guardado en: {png_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Optimizador de Perfiles Aerodinámicos en Espacio Latente - DIPAS")
    parser.add_argument("--cl", type=float, default=0.70, help="Sustentación objetivo (Cl)")
    parser.add_argument("--cd", type=float, default=None, help="Arrastre objetivo (Cd). Si no se pasa, se maximiza L/D.")
    parser.add_argument("--tc", type=float, default=0.12, help="Espesor relativo objetivo (t/c, ej. 0.12 para 12 pct)")
    parser.add_argument("--reynolds", type=float, default=150000.0, help="Número de Reynolds de diseño")
    parser.add_argument("--alpha", type=float, default=3.0, help="Ángulo de ataque de crucero (grados)")
    parser.add_argument("--maximize_ld", action="store_true", default=True, help="Maximizar la finura aerodinámica (L/D)")
    parser.add_argument("--iters", type=int, default=80, help="Número de iteraciones de Adam")
    parser.add_argument("--name", type=str, default="dipas_uav_cruise", help="Nombre del perfil a exportar")
    
    args = parser.parse_args()
    
    optimizer = LatentAirfoilOptimizer()
    results = optimizer.optimize(
        target_cl=args.cl,
        target_cd=args.cd,
        target_tc=args.tc,
        reynolds=args.reynolds,
        alpha=args.alpha,
        maximize_ld=args.maximize_ld,
        n_iters=args.iters
    )
    
    optimizer.export_results(results, name=args.name)

if __name__ == "__main__":
    main()
