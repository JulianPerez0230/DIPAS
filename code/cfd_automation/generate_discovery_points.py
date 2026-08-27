import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_builder import DatasetBuilder

def generate_discovery_points_sharp_fixed():
    db = DatasetBuilder()
    s3021_path = os.path.abspath("../data/seeds/s3021.dat")
    
    # Leemos y alineamos coordenadas
    x, y_up, y_low = db.parse_uiuc_file(s3021_path, n_points=80)
    
    # FILTRO ANTICOLISIÓN: extradós >= intradós
    y_up_clean = np.maximum(y_up, y_low)
    y_low_clean = np.minimum(y_up, y_low)
    
    # CORRECCIÓN DE ÍNDICE: En la grilla de coseno, x=1.0 es el último elemento (índice -1)
    y_up_clean[-1] = 0.0   # En x=1.0 (borde de fuga)
    y_low_clean[-1] = 0.0  # En x=1.0 (borde de fuga)
    
    # Preparamos las curvas por separado
    x_up_inv = np.flip(x)
    y_up_inv = np.flip(y_up_clean)
    
    x_low = x[1:]
    y_low = y_low_clean[1:]
    
    output_path = os.path.abspath("../data/s3021_discovery.txt")
    cuerda = 200.0 # 200 mm
    
    with open(output_path, "w") as f:
        # Cabeceras globales
        f.write("3d=true\n")
        f.write("polyline=false\n")
        f.write("fit=false\n\n")
        
        # Grupo 1: Extradós
        for xi, yi in zip(x_up_inv, y_up_inv):
            f.write(f"0.000000   {xi*cuerda:.6f}   {yi*cuerda:.6f}\n")
            
        # Línea en blanco
        f.write("\n")
        
        # Grupo 2: Intradós
        f.write(f"0.000000   {x_up_inv[-1]*cuerda:.6f}   {y_up_inv[-1]*cuerda:.6f}\n")
        for xi, yi in zip(x_low, y_low):
            f.write(f"0.000000   {xi*cuerda:.6f}   {yi*cuerda:.6f}\n")
            
    print(f"Archivo de importación corregido (borde de fuga afilado) generado en: {output_path}")

if __name__ == "__main__":
    generate_discovery_points_sharp_fixed()
