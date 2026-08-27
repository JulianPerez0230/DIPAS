# -*- coding: utf-8 -*-
"""
Parser de datos de túnel de viento UIUC.
Recorre las carpetas volume01..06 de data/uiuc_tunnel_data/, extrae las curvas polares 
experimentales de DRAG*.TXT y las asocia con los parámetros CST de uiuc_cst_dataset.csv.
Guarda los resultados unificados en data/uiuc_polar_dataset.csv.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

def normalize_name(name):
    """
    Normaliza los nombres de los perfiles para facilitar el emparejamiento.
    E.g. 'E387 (C)' -> 'e387c', 'NACA 0012' -> 'naca0012'
    """
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def parse_drag_file(file_path):
    """
    Parsea un archivo DRAG*.TXT de la UIUC que contiene bloques de polares.
    Retorna una lista de diccionarios con {airfoil_raw, reynolds, alpha, cl, cd}
    """
    data_points = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    # Expresión regular para separar cada bloque de perfil
    # Cada bloque empieza con 'Airfoil:'
    blocks = content.split("Airfoil:")
    
    for block in blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue
            
        airfoil_name = lines[0].split("Builder:")[0].strip()
        
        # Buscar Reynolds y datos polares en el bloque
        current_re = None
        in_data_block = False
        
        for line in lines[1:]:
            line_strip = line.strip()
            
            # Buscar Reynolds promedio
            if "Average Reynolds #:" in line or "Reynolds #:" in line:
                in_data_block = False
                continue
            
            # Si la línea anterior definió el Reynolds, buscamos el valor numérico
            if current_re is None or not in_data_block:
                re_match = re.search(r'^\s*(\d+)\s*$', line_strip)
                if re_match:
                    current_re = float(re_match.group(1))
                    continue
                    
            # Detectar cabecera de datos
            if "alpha" in line_strip and ("Cl" in line_strip or "Cl/Cd" in line_strip or "Cd" in line_strip):
                in_data_block = True
                continue
                
            # Fin del bloque de datos por archivo o firma
            if "Tabulated from" in line_strip or "Average Reynolds" in line_strip:
                in_data_block = False
                current_re = None
                continue
                
            # Procesar fila de datos
            if in_data_block and line_strip:
                parts = line_strip.split()
                if len(parts) >= 3:
                    try:
                        alpha = float(parts[0])
                        cl = float(parts[1])
                        cd = float(parts[2])
                        
                        data_points.append({
                            "airfoil_raw": airfoil_name,
                            "reynolds": current_re,
                            "alpha": alpha,
                            "cl": cl,
                            "cd": cd
                        })
                    except ValueError:
                        continue # Saltamos líneas no numéricas
                        
    return data_points

def main():
    root_dir = SCRIPT_DIR.parent
    tunnel_dir = root_dir / "data" / "uiuc_tunnel_data"
    uiuc_cst_path = root_dir / "data" / "uiuc_cst_dataset.csv"
    output_path = root_dir / "data" / "uiuc_polar_dataset.csv"
    
    if not uiuc_cst_path.exists():
        print(f"❌ No se encuentra el dataset CST de UIUC: {uiuc_cst_path}")
        sys.exit(1)
        
    print(">> Cargando mapa CST de la UIUC...")
    cst_df = pd.read_csv(uiuc_cst_path)
    # Crear un diccionario de búsqueda indexando por el nombre normalizado
    cst_df["norm_name"] = cst_df["airfoil_name"].apply(normalize_name)
    cst_map = cst_df.set_index("norm_name").to_dict(orient="index")
    
    print(">> Escaneando archivos de túnel UIUC...")
    all_points = []
    
    # Recorrer volúmenes
    for vol_dir in tunnel_dir.iterdir():
        if vol_dir.is_dir() and vol_dir.name.startswith("volume"):
            for f in vol_dir.iterdir():
                if f.is_file() and f.name.startswith("DRAG") and f.suffix.lower() == ".txt":
                    print(f"   Parsea: {f.relative_to(root_dir)}")
                    points = parse_drag_file(f)
                    all_points.extend(points)
                    
    print(f">> Total puntos polares crudos leídos: {len(all_points)}")
    
    # Emparejar con CST y consolidar
    consolidated_rows = []
    cst_cols = [f"au{i}" for i in range(6)] + [f"al{i}" for i in range(6)]
    
    matched_set = set()
    unmatched_set = set()
    
    for pt in all_points:
        norm = normalize_name(pt["airfoil_raw"])
        # Intentar coincidencia exacta o coincidencia parcial
        match = None
        if norm in cst_map:
            match = cst_map[norm]
        else:
            # Buscar si el nombre normalizado del túnel está contenido en el nombre CST o viceversa
            for cst_norm, cst_row in cst_map.items():
                if norm in cst_norm or cst_norm in norm:
                    match = cst_row
                    break
                    
        if match:
            matched_set.add(pt["airfoil_raw"])
            row = {
                "airfoil_name": match["airfoil_name"],
                "reynolds": pt["reynolds"],
                "alpha": pt["alpha"],
                "cl": pt["cl"],
                "cd": pt["cd"]
            }
            # Copiar coeficientes CST
            for col in cst_cols:
                row[col] = match[col]
            consolidated_rows.append(row)
        else:
            unmatched_set.add(pt["airfoil_raw"])
            
    print(f"\n>> Resumen de Emparejamiento:")
    print(f"   - Perfiles emparejados exitosamente: {len(matched_set)}")
    print(f"   - Perfiles no emparejados: {len(unmatched_set)}")
    print(f"   - Puntos polares consolidados: {len(consolidated_rows)}")
    
    if not consolidated_rows:
        print("❌ Error: No se logró emparejar ningún punto polar con sus coordenadas CST.")
        sys.exit(1)
        
    out_df = pd.DataFrame(consolidated_rows)
    out_df.to_csv(output_path, index=False)
    print(f">> Dataset polar UIUC experimental guardado en: {output_path.resolve()}")

if __name__ == "__main__":
    main()
