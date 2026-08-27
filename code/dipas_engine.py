# -*- coding: utf-8 -*-
"""
DIPAS Engine - Backend de Inferencia, Optimización y Validación Aerodinámica
DIPAS: Diseño Inverso para Perfiles con Autoencoder y Simulación
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Agregar directorio actual al PATH
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR / "cfd_automation"))

from cvae_model import CVAE
from surrogate_model import AerodynamicSurrogate
from cst_generator import CSTParametrization
from xfoil_wrapper import XFoilWrapper

class DIPASEngine:
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.absolute()
        else:
            self.project_root = Path(project_root)
            
        self.code_dir = self.project_root / "code"
        self.outputs_dir = self.project_root / "outputs"
        self.data_dir = self.project_root / "data"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Parámetros exactos de normalización derivados del dataset de entrenamiento
        # Cond: [reynolds, cl, cd, t_c]
        self.cond_min = np.array([100000.0, -0.56390, 0.00000, 0.080023], dtype=np.float32)
        self.cond_max = np.array([300000.0,  1.56460, 0.07072, 0.147092], dtype=np.float32)
        
        # Instanciar generador CST
        self.cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
        
        # Modelos
        self.cvae_model = None
        self.current_cvae_name = None
        self.surrogate_model = None
        self.surrogate_scalers = None
        
        # Inicializar wrapper XFOIL multiplataforma (Windows/Linux)
        try:
            self.xfoil = XFoilWrapper()
            print(f"[DIPAS Init] XFOIL Wrapper inicializado exitosamente: {self.xfoil.xfoil_path}")
        except Exception as e:
            print(f"[DIPAS Init] Advertencia: XFOIL no inicializado: {e}")
            self.xfoil = None
        
        # Cargar Surrogate por defecto si existe
        self._init_surrogate()

    def get_available_cvae_models(self):
        """Retorna la lista de modelos CVAE disponibles en outputs/"""
        models = []
        if self.outputs_dir.exists():
            for f in self.outputs_dir.glob("*.pth"):
                if "surrogate" not in f.name.lower() and "autoencoder" not in f.name.lower():
                    models.append(f.name)
        if not models:
            models = ["dipas_tl_exp_model.pth", "dipas_tl_model.pth", "dipas_base_model.pth"]
        return sorted(models)

    def load_cvae(self, model_name="dipas_tl_exp_model.pth"):
        """Carga el modelo CVAE especificado."""
        model_path = self.outputs_dir / model_name
        if not model_path.exists():
            candidates = list(self.project_root.glob(f"**/{model_name}"))
            if candidates:
                model_path = candidates[0]
            else:
                raise FileNotFoundError(f"No se encontró el modelo {model_name} en {self.outputs_dir}")
                
        model = CVAE(cst_dim=12, cond_dim=4, latent_dim=6, hidden_dims=[256, 128, 64]).to(self.device)
        state_dict = torch.load(str(model_path), map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        self.cvae_model = model
        self.current_cvae_name = model_name
        return True

    def _init_surrogate(self):
        """Carga los pesos y escaladores del modelo Surrogate de alta precisión."""
        scalers_path = self.outputs_dir / "surrogate_precision_scalers.json"
        weights_path = self.outputs_dir / "surrogate_precision.pth"
        
        if not weights_path.exists():
            scalers_path = self.outputs_dir / "surrogate_scalers.json"
            weights_path = self.outputs_dir / "surrogate_numeric.pth"
            if not weights_path.exists():
                weights_path = self.outputs_dir / "surrogate_hybrid.pth"
            
        if scalers_path.exists() and weights_path.exists():
            try:
                with open(scalers_path, "r") as f:
                    self.surrogate_scalers = json.load(f)
                    
                model = AerodynamicSurrogate(input_dim=14, hidden_dims=[256, 128, 64], output_dim=2).to(self.device)
                model.load_state_dict(torch.load(str(weights_path), map_location=self.device))
                model.eval()
                self.surrogate_model = model
            except Exception as e:
                print(f"[Aviso] No se pudo cargar el Surrogate Model: {e}")

    def normalize_condition(self, reynolds, cl, cd, tc):
        """Normaliza el vector de condición a [0, 1]."""
        cond = np.array([reynolds, cl, cd, tc], dtype=np.float32)
        denom = self.cond_max - self.cond_min
        denom[denom == 0] = 1.0
        cond_norm = np.clip((cond - self.cond_min) / denom, 0.0, 1.0)
        return torch.tensor(cond_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

    def generate_airfoils(self, cl_target, cd_target, reynolds, tc_target, n_samples=50, seed=None, custom_z=None, eval_alpha=3.0, **kwargs):
        """
        Genera N variantes de perfiles a partir de las condiciones de diseño.
        Retorna una lista de diccionarios ordenados por ranking aerodinámico.
        """
        if self.cvae_model is None:
            available = self.get_available_cvae_models()
            default_model = "dipas_tl_model.pth" if "dipas_tl_model.pth" in available else available[0]
            self.load_cvae(default_model)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        cond_tensor = self.normalize_condition(reynolds, cl_target, cd_target, tc_target)
        
        # 1. Optimización multi-arquetipo de ingeniería aeroespacial
        # Cada variante persigue un objetivo de diseño físico específico:
        # #1: Nominal balanceado, #2: Mínimo drag, #3: Mayor espesor, #4: Perfil delgado, #5: Bajo momento de cabeceo
        archetypes = [
            {
                "name": "Óptimo Balanceado Nominal",
                "tag": "Nominal / Balanceado",
                "desc": "Compromiso de diseño con máxima eficiencia L/D y espesor estándar.",
                "tc_t": float(np.clip(tc_target, 0.08, 0.16)),
                "w_cd": 12.0,
                "w_tc": 80.0,
                "w_cm": 0.0
            },
            {
                "name": "Mínimo Arrastre (High L/D)",
                "tag": "Mínimo Arrastre / Alto L/D",
                "desc": "Minimización agresiva de arrastre (CD) para planeo y bajo consumo.",
                "tc_t": float(np.clip(tc_target * 0.95, 0.075, 0.15)),
                "w_cd": 45.0,
                "w_tc": 40.0,
                "w_cm": 0.0
            },
            {
                "name": "Mayor Espesor (Estructural)",
                "tag": "Mayor Espesor / Estructural",
                "desc": "Mayor espesor t/c (+20%) para volumen interno y rigidez de larguero.",
                "tc_t": float(np.clip(tc_target * 1.22, 0.10, 0.165)),
                "w_cd": 8.0,
                "w_tc": 320.0,
                "w_cm": 0.0
            },
            {
                "name": "Perfil Delgado (Alta Velocidad)",
                "tag": "Perfil Delgado / Velocidad",
                "desc": "Perfil fino (-20% t/c) con mínima resistencia de forma y área frontal.",
                "tc_t": float(np.clip(tc_target * 0.78, 0.070, 0.12)),
                "w_cd": 15.0,
                "w_tc": 320.0,
                "w_cm": 0.0
            },
            {
                "name": "Bajo Momento (Estabilidad / Trim)",
                "tag": "Bajo Momento / Fácil Trim",
                "desc": "Reducción del momento de cabeceo |CM| para facilitar el trimado.",
                "tc_t": float(np.clip(tc_target, 0.08, 0.15)),
                "w_cd": 10.0,
                "w_tc": 60.0,
                "w_cm": 200.0
            }
        ]

        top_cst_list = []
        top_meta = []

        if custom_z is not None:
            z_opt = torch.tensor(custom_z, dtype=torch.float32, device=self.device).unsqueeze(0)
            cst_generated = self.cvae_model.decode(z_opt, cond_tensor).cpu().detach().numpy()
            for c in cst_generated:
                top_cst_list.append(c)
                top_meta.append(archetypes[0])
        else:
            if self.surrogate_model is not None and self.surrogate_scalers is not None:
                mean_in = torch.tensor(self.surrogate_scalers["input_mean"], dtype=torch.float32, device=self.device)
                std_in = torch.tensor(self.surrogate_scalers["input_std"], dtype=torch.float32, device=self.device)
                mean_out = torch.tensor(self.surrogate_scalers["output_mean"], dtype=torch.float32, device=self.device)
                std_out = torch.tensor(self.surrogate_scalers["output_std"], dtype=torch.float32, device=self.device)
                alpha_single = torch.tensor([eval_alpha, reynolds], dtype=torch.float32, device=self.device)

                for arch in archetypes:
                    tc_t = arch["tc_t"]
                    cond_arch = self.normalize_condition(reynolds, cl_target, cd_target, tc_t)
                    
                    # Multi-start en espacio latente
                    K = 8
                    z_var = torch.randn((K, 6), dtype=torch.float32, device=self.device, requires_grad=True)
                    opt_z = torch.optim.Adam([z_var], lr=0.10)
                    alpha_re = alpha_single.repeat(K, 1)
                    cond_K = cond_arch.repeat(K, 1)
                    
                    for _ in range(45):
                        opt_z.zero_grad()
                        cst_p = self.cvae_model.decode(z_var, cond_K)
                        yu_t, yl_t = self.cvae_model.cst_to_coord(cst_p)
                        tc_t_p = torch.max(yu_t - yl_t, dim=1)[0]
                        camb_p = torch.max(0.5 * (yu_t + yl_t), dim=1)[0]
                        
                        s_in = (torch.cat([cst_p, alpha_re], dim=1) - mean_in) / std_in
                        pred = self.surrogate_model(s_in) * std_out + mean_out
                        cl_t = pred[:, 0]
                        cd_t = pred[:, 1]
                        
                        loss = torch.mean(
                            1400.0 * (cl_t - cl_target)**2 + 
                            arch["w_cd"] * cd_t + 
                            arch["w_tc"] * (tc_t_p - tc_t)**2 + 
                            arch["w_cm"] * (camb_p - 0.025)**2 + 
                            0.002 * torch.sum(z_var**2, dim=1)
                        )
                        loss.backward()
                        opt_z.step()
                        
                    # Seleccionar mejor z
                    cst_cand_k = self.cvae_model.decode(z_var, cond_K)
                    s_in_k = (torch.cat([cst_cand_k, alpha_re], dim=1) - mean_in) / std_in
                    p_k = self.surrogate_model(s_in_k) * std_out + mean_out
                    best_k = torch.argmin((p_k[:, 0] - cl_target)**2 + 0.1 * p_k[:, 1])
                    best_z = z_var[best_k:best_k+1].detach()
                    
                    # Fine-tuning sobre CST
                    cst_init = self.cvae_model.decode(best_z, cond_arch).detach()
                    cst_fine = cst_init.clone().requires_grad_(True)
                    opt_c = torch.optim.Adam([cst_fine], lr=0.015)
                    
                    for _ in range(40):
                        opt_c.zero_grad()
                        yu_s, yl_s = self.cvae_model.cst_to_coord(cst_fine)
                        tc_s = torch.max(yu_s - yl_s, dim=1)[0]
                        camb_s = torch.max(0.5 * (yu_s + yl_s), dim=1)[0]
                        
                        s_in_s = (torch.cat([cst_fine.squeeze(0), alpha_single], dim=0) - mean_in) / std_in
                        pred_s = self.surrogate_model(s_in_s.unsqueeze(0)) * std_out + mean_out
                        cl_s = pred_s[0, 0]
                        cd_s = pred_s[0, 1]
                        
                        loss_c = (
                            1600.0 * (cl_s - cl_target)**2 + 
                            arch["w_cd"] * cd_s + 
                            arch["w_tc"] * (tc_s - tc_t)**2 + 
                            arch["w_cm"] * (camb_s - 0.025)**2 + 
                            25.0 * torch.mean((cst_fine - cst_init)**2)
                        )
                        loss_c.backward()
                        opt_c.step()
                        
                    top_cst_list.append(cst_fine.detach().cpu().numpy()[0])
                    top_meta.append(arch)
            else:
                for arch in archetypes:
                    z_dummy = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
                    cond_arch = self.normalize_condition(reynolds, cl_target, cd_target, arch["tc_t"])
                    cst_d = self.cvae_model.decode(z_dummy, cond_arch).detach().cpu().numpy()[0]
                    top_cst_list.append(cst_d)
                    top_meta.append(arch)

        # Generación de candidatos con metadatos de arquetipo
        candidates = []
        for i, cst_vec in enumerate(top_cst_list):
            arch_info = top_meta[i]
            cst_u = cst_vec[:6]
            cst_l = cst_vec[6:]
            
            x, y_u, y_l = self.cst.generate_coordinates(cst_u, cst_l, n_points=160)
            
            thickness = y_u - y_l
            max_tc_idx = np.argmax(thickness)
            max_tc = float(thickness[max_tc_idx])
            x_tc_max = float(x[max_tc_idx])
            
            camber = 0.5 * (y_u + y_l)
            max_camber_idx = np.argmax(camber)
            max_camber = float(camber[max_camber_idx])
            x_camber_max = float(x[max_camber_idx])
            
            te_gap = float(y_u[-1] - y_l[-1])
            is_valid = bool(np.all(thickness >= -0.001) and te_gap >= 0.001)
            
            pred_cl, pred_cd, pred_ld = self.evaluate_with_surrogate(cst_vec, alpha=eval_alpha, reynolds=reynolds)
            
            candidate = {
                "id": i + 1,
                "rank": i + 1,
                "archetype_name": arch_info["name"],
                "archetype_tag": arch_info["tag"],
                "archetype_desc": arch_info["desc"],
                "cst_upper": cst_u.tolist(),
                "cst_lower": cst_l.tolist(),
                "cst_all": cst_vec.tolist(),
                "x": x.tolist(),
                "y_upper": y_u.tolist(),
                "y_lower": y_l.tolist(),
                "camber": camber.tolist(),
                "thickness": thickness.tolist(),
                "max_tc": max_tc,
                "x_tc_max": x_tc_max,
                "max_camber": max_camber,
                "x_camber_max": x_camber_max,
                "te_gap": te_gap,
                "is_valid": is_valid,
                "surrogate_cl": pred_cl,
                "surrogate_cd": pred_cd,
                "surrogate_ld": pred_ld,
                "surrogate_cm": float(-2.0 * max_camber * (1.0 - x_camber_max)),
                "tc_error": abs(max_tc - tc_target),
                "cl_error": abs(pred_cl - cl_target) if pred_cl is not None else 0.0
            }
            candidates.append(candidate)

        return candidates

    def evaluate_with_surrogate(self, cst_12, alpha=3.0, reynolds=200000.0):
        """Predice CL, CD y L/D usando el modelo sustituto."""
        if self.surrogate_model is None or self.surrogate_scalers is None:
            return None, None, None
            
        try:
            inp_raw = np.concatenate([cst_12, [alpha, reynolds]]).astype(np.float32)
            mean_in = np.array(self.surrogate_scalers["input_mean"], dtype=np.float32)
            std_in = np.array(self.surrogate_scalers["input_std"], dtype=np.float32)
            inp_norm = (inp_raw - mean_in) / std_in
            
            with torch.no_grad():
                tensor_in = torch.tensor(inp_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
                out_norm = self.surrogate_model(tensor_in).cpu().numpy()[0]
                
            mean_out = np.array(self.surrogate_scalers["output_mean"], dtype=np.float32)
            std_out = np.array(self.surrogate_scalers["output_std"], dtype=np.float32)
            out_raw = out_norm * std_out + mean_out
            
            cl = float(out_raw[0])
            cd = float(max(out_raw[1], 0.003))
            ld = float(cl / cd) if cd > 0 else 0.0
            return cl, cd, ld
        except Exception:
            return None, None, None

    def run_xfoil_validation(self, candidate, reynolds=200000, alpha_start=-4.0, alpha_end=14.0, alpha_step=1.0, eval_alpha=3.0):
        """Ejecuta una corrida de polar y Cp con XFOIL, con respaldo continuo multi-fidelidad."""
        print(f"[DIPAS Validation] Iniciando simulación XFOIL. Wrapper path: {self.xfoil.xfoil_path if self.xfoil else 'None'}")
        polar_df = None
        cp_data = None
        cp_alpha = float(eval_alpha)
        is_fallback = False
        
        x = np.array(candidate["x"])
        y_u = np.array(candidate["y_upper"])
        y_l = np.array(candidate["y_lower"])
        cst_vec = np.array(candidate["cst_all"])

        if self.xfoil is not None:
            try:
                polar_df = self.xfoil.run_simulation(
                    x, y_u, y_l, 
                    reynolds=reynolds, 
                    alpha_start=alpha_start, 
                    alpha_end=alpha_end, 
                    alpha_step=alpha_step,
                    file_prefix="dipas_ui"
                )
                cp_data = self.xfoil.get_cp_distribution(x, y_u, y_l, reynolds=reynolds, alpha=cp_alpha, file_prefix="dipas_ui")
            except Exception as ex:
                print(f"[XFOIL Aviso] Error en subproceso XFOIL: {ex}")
                polar_df = None

        # Si XFOIL no convergió o no está disponible en el entorno, generar polar física multi-fidelidad
        if polar_df is None or not polar_df.get("alpha") or len(polar_df["alpha"]) < 2:
            is_fallback = True
            alphas = np.arange(alpha_start, alpha_end + 0.1, alpha_step)
            p_dict = {"alpha": [], "CL": [], "CD": [], "CM": []}
            
            for a in alphas:
                pcl, pcd, pld = self.evaluate_with_surrogate(cst_vec, alpha=float(a), reynolds=float(reynolds))
                if pcl is not None:
                    p_dict["alpha"].append(float(a))
                    p_dict["CL"].append(float(pcl))
                    p_dict["CD"].append(float(pcd))
                    cm_val = float(candidate["surrogate_cm"] - 0.002 * (a - eval_alpha))
                    p_dict["CM"].append(cm_val)
                    
            polar_df = p_dict
            
            # Generar distribución de Cp teórica/numérica consistente con formato estándar
            if cp_data is None:
                x_pts = np.linspace(0.001, 1.0, 80)
                thickness = np.interp(x_pts, x, candidate["thickness"])
                cl_curr = float(candidate["surrogate_cl"]) if candidate.get("surrogate_cl") is not None else 1.0
                
                # Distribución analítica Cp
                cp_u = 1.0 - (1.0 + 1.8 * thickness + 0.45 * cl_curr * np.sqrt((1.0 - x_pts) / x_pts))**2
                cp_l = 1.0 - (1.0 + 1.1 * thickness - 0.25 * cl_curr * np.sqrt((1.0 - x_pts) / x_pts))**2
                cp_u = np.clip(cp_u, -5.5, 1.0)
                cp_l = np.clip(cp_l, -1.8, 1.0)

                x_comb = np.concatenate([np.flip(x_pts), x_pts])
                y_u_interp = np.interp(x_pts, x, candidate["y_upper"])
                y_l_interp = np.interp(x_pts, x, candidate["y_lower"])
                y_comb = np.concatenate([np.flip(y_u_interp), y_l_interp])
                cp_comb = np.concatenate([np.flip(cp_u), cp_l])
                
                cp_data = {
                    "x": x_comb.tolist(),
                    "y": y_comb.tolist(),
                    "Cp": cp_comb.tolist()
                }

        return {
            "polar": polar_df,
            "cp": cp_data,
            "reynolds": reynolds,
            "evaluated_alpha_cp": cp_alpha,
            "is_fallback": is_fallback,
            "solver_name": "XFOIL 6.99 (Método de Paneles Viscosos eⁿ — Mark Drela, MIT)" if not is_fallback else "Surrogate Multi-Fidelidad (Red Tensorial RANS / Túnel UIUC)"
        }

    def run_surrogate_validation(self, candidate, reynolds=200000, alpha_start=-4.0, alpha_end=14.0, alpha_step=1.0, eval_alpha=3.0):
        """Ejecuta una corrida instantánea con el Modelo Sustituto Físico (RANS/XFOIL/UIUC)."""
        x = np.array(candidate["x"])
        cst_vec = np.array(candidate["cst_all"])
        cp_alpha = float(eval_alpha)
        
        alphas = np.arange(alpha_start, alpha_end + 0.1, alpha_step)
        p_dict = {"alpha": [], "CL": [], "CD": [], "CM": []}
        
        for a in alphas:
            pcl, pcd, pld = self.evaluate_with_surrogate(cst_vec, alpha=float(a), reynolds=float(reynolds))
            if pcl is not None:
                p_dict["alpha"].append(float(a))
                p_dict["CL"].append(float(pcl))
                p_dict["CD"].append(float(pcd))
                cm_val = float(candidate["surrogate_cm"] - 0.002 * (a - eval_alpha))
                p_dict["CM"].append(cm_val)
                
        # Distribución de Cp analítica
        x_pts = np.linspace(0.001, 1.0, 80)
        thickness = np.interp(x_pts, x, candidate["thickness"])
        cl_curr = float(candidate["surrogate_cl"]) if candidate.get("surrogate_cl") is not None else 1.0
        
        cp_u = 1.0 - (1.0 + 1.8 * thickness + 0.45 * cl_curr * np.sqrt((1.0 - x_pts) / x_pts))**2
        cp_l = 1.0 - (1.0 + 1.1 * thickness - 0.25 * cl_curr * np.sqrt((1.0 - x_pts) / x_pts))**2
        cp_u = np.clip(cp_u, -5.5, 1.0)
        cp_l = np.clip(cp_l, -1.8, 1.0)

        x_comb = np.concatenate([np.flip(x_pts), x_pts])
        y_u_interp = np.interp(x_pts, x, candidate["y_upper"])
        y_l_interp = np.interp(x_pts, x, candidate["y_lower"])
        y_comb = np.concatenate([np.flip(y_u_interp), y_l_interp])
        cp_comb = np.concatenate([np.flip(cp_u), cp_l])
        
        cp_data = {
            "x": x_comb.tolist(),
            "y": y_comb.tolist(),
            "Cp": cp_comb.tolist()
        }

        return {
            "polar": p_dict,
            "cp": cp_data,
            "reynolds": reynolds,
            "evaluated_alpha_cp": cp_alpha,
            "is_fallback": True,
            "solver_name": "Surrogate Multi-Fidelidad (Red Tensorial RANS / Túnel UIUC)"
        }

    def detect_ansys_fluent(self):
        """
        Escanea el sistema en busca de una instalación válida de ANSYS Fluent.
        Prioriza la versión más reciente instalada (ej. v261 sobre v252).
        Retorna (is_installed: bool, fluent_path: str, version: str).
        """
        detected = []
        for k, v in os.environ.items():
            if k.startswith("AWP_ROOT"):
                version_str = k.replace("AWP_ROOT", "v")
                candidate_exe = Path(v) / "fluent" / "ntbin" / "win64" / "fluent.exe"
                if candidate_exe.exists():
                    detected.append((version_str, str(candidate_exe)))
                    
        common_paths = [
            r"C:\Program Files\ANSYS Inc\ANSYS Student\v261\fluent\ntbin\win64\fluent.exe",
            r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\fluent\ntbin\win64\fluent.exe",
            r"C:\Program Files\ANSYS Inc\ANSYS Student\v241\fluent\ntbin\win64\fluent.exe",
            r"C:\Program Files\ANSYS Inc\v261\fluent\ntbin\win64\fluent.exe",
            r"C:\Program Files\ANSYS Inc\v252\fluent\ntbin\win64\fluent.exe",
            r"C:\Program Files\ANSYS Inc\v241\fluent\ntbin\win64\fluent.exe",
            r"C:\Program Files\ANSYS Inc\v232\fluent\ntbin\win64\fluent.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                parts = p.split(os.sep)
                ver = [part for part in parts if part.startswith("v") and len(part) == 4]
                version_str = ver[0] if ver else "Detected"
                detected.append((version_str, p))
                
        if detected:
            # Eliminar duplicados manteniendo orden
            seen = set()
            unique_detected = []
            for ver, p in detected:
                if p not in seen:
                    seen.add(p)
                    unique_detected.append((ver, p))
            # Ordenar por versión descendente (ej. v261 antes de v252)
            unique_detected.sort(key=lambda x: x[0], reverse=True)
            best_ver, best_path = unique_detected[0]
            return True, best_path, best_ver
            
        return False, None, None

    def launch_ansys_gui(self, candidate, reynolds=200000, alpha=0.0):
        """
        Genera la malla CFD 2D del perfil y lanza ANSYS Fluent precargando el caso y la física.
        """
        is_inst, fluent_exe, _ = self.detect_ansys_fluent()
        if not is_inst:
            raise FileNotFoundError("ANSYS Fluent no está instalado en este sistema.")
            
        export_file = self.data_dir / "fluent_active_airfoil.dat"
        self.export_to_selig_format(candidate, str(export_file), chord_mm=200.0)
        
        try:
            from test_gmsh_fluent import build_and_launch_fluent_case
            return build_and_launch_fluent_case(
                candidate=candidate,
                data_dir=self.data_dir,
                fluent_exe=fluent_exe,
                reynolds=reynolds,
                alpha=alpha,
                chord=0.200
            )
        except Exception as e:
            print(f"Error en launch_ansys_gui: {e}")
            cmd = [fluent_exe, "2d", "-t4"]
            subprocess.Popen(cmd, cwd=str(self.data_dir))
            return True

    def export_to_selig_format(self, candidate, file_path=None, chord_mm=200.0, name="DIPAS_AIRFOIL"):
        """Exporta el perfil a formato estándar .dat (Selig / UIUC / XFOIL)."""
        x = np.array(candidate["x"])
        y_u = np.array(candidate["y_upper"])
        y_l = np.array(candidate["y_lower"])
        
        x_up_inv = np.flip(x)
        y_up_inv = np.flip(y_u)
        x_low = x[1:]
        y_low = y_l[1:]
        
        x_coords = np.concatenate([x_up_inv, x_low])
        y_coords = np.concatenate([y_up_inv, y_low])
        
        content = [f"{name} (DIPAS AI Generated, Chord={chord_mm:.1f}mm)"]
        for xi, yi in zip(x_coords, y_coords):
            content.append(f"  {xi:10.6f}   {yi:10.6f}")
            
        dat_text = "\n".join(content)
        
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(dat_text)
                
        return dat_text

    def export_to_csv(self, candidate, file_path=None, chord_mm=200.0):
        """Exporta tabla de coordenadas dimensionales y adimensionales en CSV."""
        x = np.array(candidate["x"])
        y_u = np.array(candidate["y_upper"])
        y_l = np.array(candidate["y_lower"])
        t = np.array(candidate["thickness"])
        c = np.array(candidate["camber"])
        
        df = pd.DataFrame({
            "x/c": x,
            "y_upper/c": y_u,
            "y_lower/c": y_l,
            "thickness/c": t,
            "camber/c": c,
            "X_mm": x * chord_mm,
            "Y_upper_mm": y_u * chord_mm,
            "Y_lower_mm": y_l * chord_mm
        })
        
        if file_path:
            df.to_csv(file_path, index=False)
            
        return df.to_csv(index=False)
