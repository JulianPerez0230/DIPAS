# -*- coding: utf-8 -*-
"""
Fase 1: Preprocesamiento y Ajuste CST de la base de datos UIUC.
Lee las coordenadas (.dat) de data/uiuc_airfoils/, interpola las superficies
extradós e intradós en una grilla de coseno, calcula los 12 parámetros CST
por mínimos cuadrados y los guarda en uiuc_cst_dataset.csv.
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

def parse_uiuc_file(file_path, n_points=100):
    """
    Lee un archivo de perfil UIUC (.dat) y separa las coordenadas de extradós/intradós,
    interpolando en una grilla de coseno estándar.
    """
    x_raw, y_raw = [], []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    if not lines:
        return None
        
    # Omitimos la primera línea (cabecera con el nombre del perfil)
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2:
            try:
                val_x = float(parts[0])
                val_y = float(parts[1])
                if val_x <= 1.05:  # Filtro de coordenadas normalizadas
                    x_raw.append(val_x)
                    y_raw.append(val_y)
            except ValueError:
                continue
                
    if len(x_raw) < 10:
        return None
        
    x_raw = np.array(x_raw)
    y_raw = np.array(y_raw)
    
    # Detección de formato: Lednicer (extradós va de 0 a 1, luego intradós de 0 a 1)
    # o Estándar (un solo recorrido de 1 -> 0 -> 1)
    if x_raw[0] < 0.1:
        # Formato Lednicer
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
        # Formato Estándar UIUC (1 -> 0 -> 1)
        idx_min_x = np.argmin(x_raw)
        
        x_upper = x_raw[:idx_min_x+1]
        y_upper = y_raw[:idx_min_x+1]
        x_lower = x_raw[idx_min_x:]
        y_lower = y_raw[idx_min_x:]
        
        theta = np.linspace(0, np.pi, n_points)
        x_grid = 0.5 * (1.0 - np.cos(theta))
        
        # En UIUC estándar, el extradós va de 1 a 0, lo invertimos para ordenar de 0 a 1
        y_upper_grid = np.interp(x_grid, np.flip(x_upper), np.flip(y_upper))
        y_lower_grid = np.interp(x_grid, x_lower, y_lower)
        
    return x_grid, y_upper_grid, y_lower_grid

def main():
    uiuc_dir = SCRIPT_DIR.parent / "data" / "uiuc_airfoils"
    output_file = SCRIPT_DIR.parent / "data" / "uiuc_cst_dataset.csv"
    
    if not uiuc_dir.exists():
        print(f"❌ La carpeta de perfiles no existe: {uiuc_dir}")
        sys.exit(1)
        
    print(f">> Iniciando ajuste CST sobre base de datos UIUC en: {uiuc_dir}")
    
    # 6 coeficientes arriba, 6 abajo, espesor de borde de fuga nominal de 0.003
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    
    dat_files = sorted([f for f in os.listdir(uiuc_dir) if f.endswith(".dat")])
    print(f"Se encontraron {len(dat_files)} archivos .dat para procesar.")
    
    headers = [
        "airfoil_name",
        "au0", "au1", "au2", "au3", "au4", "au5",
        "al0", "al1", "al2", "al3", "al4", "al5"
    ]
    
    success_count = 0
    fail_count = 0
    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for idx, fname in enumerate(dat_files, 1):
            file_path = uiuc_dir / fname
            airfoil_name = fname.replace(".dat", "")
            
            try:
                parsed = parse_uiuc_file(file_path, n_points=120)
                if parsed is None:
                    fail_count += 1
                    continue
                    
                x_grid, y_upper, y_lower = parsed
                
                # Ajustamos coeficientes CST
                coefs_upper, coefs_lower = cst.fit_airfoil(x_grid, y_upper, y_lower)
                
                # Armar fila
                row = [airfoil_name] + list(coefs_upper) + list(coefs_lower)
                writer.writerow(row)
                success_count += 1
                
                if idx % 200 == 0 or idx == len(dat_files):
                    print(f"  Procesados {idx}/{len(dat_files)} perfiles...")
                    
            except Exception as e:
                # Omitir errores de archivos corruptos o singulares
                fail_count += 1
                
    print(f"\n>> ¡Ajuste completado con éxito!")
    print(f"   - Perfiles procesados exitosamente: {success_count}")
    print(f"   - Perfiles fallidos/omitidos: {fail_count}")
    print(f"   - Dataset guardado en: {output_file.resolve()}")

if __name__ == "__main__":
    main()
