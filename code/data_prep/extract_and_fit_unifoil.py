# -*- coding: utf-8 -*-
"""
Preprocesamiento de UniFoil:
Lee las coordenadas de las geometrías de UniFoil (airfoil_ft_geom y airfoil_nlf_geom)
en la raíz del proyecto, ajusta coeficientes CST y guarda el dataset en data/unifoil_cst_dataset.csv.
"""

import os
import sys
import numpy as np
import csv
from pathlib import Path

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

from cst_generator import CSTParametrization

def parse_coordinate_file(file_path, n_points=100):
    """
    Lee las coordenadas (x, y) de un archivo .dat e interpola extradós e intradós.
    """
    x_raw, y_raw = [], []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    if not lines:
        return None
        
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                # Si tiene 3 columnas (ej: idx x y), tomamos las dos últimas
                if len(parts) >= 3:
                    val_x = float(parts[1])
                    val_y = float(parts[2])
                else:
                    val_x = float(parts[0])
                    val_y = float(parts[1])
                
                if val_x <= 1.05:
                    x_raw.append(val_x)
                    y_raw.append(val_y)
            except ValueError:
                continue
                
    if len(x_raw) < 10:
        return None
        
    x_raw = np.array(x_raw)
    y_raw = np.array(y_raw)
    
    # Formato Lednicer o UIUC estándar
    if x_raw[0] < 0.1:
        split_idx = len(x_raw) // 2
        for i in range(1, len(x_raw)):
            if x_raw[i] < x_raw[i-1]:
                split_idx = i
                break
        x_upper = x_raw[:split_idx]
        y_upper = y_raw[:split_idx]
        x_lower = x_raw[split_idx:]
        y_lower = y_raw[split_idx:]
        
        theta = np.linspace(0, np.pi, n_points)
        x_grid = 0.5 * (1.0 - np.cos(theta))
        y_upper_grid = np.interp(x_grid, x_upper, y_upper)
        y_lower_grid = np.interp(x_grid, x_lower, y_lower)
    else:
        idx_min_x = np.argmin(x_raw)
        x_upper = x_raw[:idx_min_x+1]
        y_upper = y_raw[:idx_min_x+1]
        x_lower = x_raw[idx_min_x:]
        y_lower = y_raw[idx_min_x:]
        
        theta = np.linspace(0, np.pi, n_points)
        x_grid = 0.5 * (1.0 - np.cos(theta))
        y_upper_grid = np.interp(x_grid, np.flip(x_upper), np.flip(y_upper))
        y_lower_grid = np.interp(x_grid, x_lower, y_lower)
        
    return x_grid, y_upper_grid, y_lower_grid

def main():
    root_dir = SCRIPT_DIR.parent
    ft_geom_dir = root_dir / "airfoil_ft_geom"
    nlf_geom_dir = root_dir / "airfoil_nlf_geom"
    output_file = root_dir / "data" / "unifoil_cst_dataset.csv"
    
    # Aseguramos carpeta data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(">> Iniciando ajuste CST de geometrías UniFoil...")
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    
    headers = [
        "airfoil_name", "source",
        "au0", "au1", "au2", "au3", "au4", "au5",
        "al0", "al1", "al2", "al3", "al4", "al5"
    ]
    
    success_count = 0
    fail_count = 0
    
    # Obtener lista de archivos
    ft_files = [(ft_geom_dir / f, "ft") for f in os.listdir(ft_geom_dir) if f.endswith(".dat")] if ft_geom_dir.exists() else []
    nlf_files = [(nlf_geom_dir / f, "nlf") for f in os.listdir(nlf_geom_dir) if f.endswith(".dat")] if nlf_geom_dir.exists() else []
    
    all_files = ft_files + nlf_files
    print(f"Total perfiles encontrados en UniFoil: {len(all_files)} (FT: {len(ft_files)}, NLF: {len(nlf_files)})")
    
    # Limitaremos a un máximo de 10.000 para balancear velocidad y representatividad
    # (más que suficiente para preentrenar el espacio latente geométrico)
    max_pretrain_geoms = 10000
    if len(all_files) > max_pretrain_geoms:
        print(f"   Limitando a {max_pretrain_geoms} geometrías aleatorias para pre-entrenamiento rápido.")
        np.random.seed(42)
        indices = np.random.choice(len(all_files), max_pretrain_geoms, replace=False)
        all_files = [all_files[i] for i in indices]
        
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for idx, (file_path, source) in enumerate(all_files, 1):
            airfoil_name = f"{source}_{file_path.stem}"
            try:
                parsed = parse_coordinate_file(file_path, n_points=120)
                if parsed is None:
                    fail_count += 1
                    continue
                    
                x_grid, y_upper, y_lower = parsed
                coefs_upper, coefs_lower = cst.fit_airfoil(x_grid, y_upper, y_lower)
                
                row = [airfoil_name, source] + list(coefs_upper) + list(coefs_lower)
                writer.writerow(row)
                success_count += 1
                
                if idx % 1000 == 0 or idx == len(all_files):
                    print(f"  Procesados {idx}/{len(all_files)} perfiles...")
                    
            except Exception as e:
                fail_count += 1
                
    print(f"\n>> ¡Ajuste completado!")
    print(f"   - Geometrías procesadas exitosamente: {success_count}")
    print(f"   - Fallidos: {fail_count}")
    print(f"   - Dataset guardado en: {output_file.resolve()}")

if __name__ == "__main__":
    main()
