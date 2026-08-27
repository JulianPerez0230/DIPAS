# -*- coding: utf-8 -*-
# Script to generate a thick-trailing-edge coordinates file for SpaceClaim/Discovery (YZ-plane, 200mm chord)
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_builder import DatasetBuilder

def generate_discovery_thick():
    db = DatasetBuilder()
    s3021_path = os.path.abspath("../data/seeds/s3021.dat")
    output_path = os.path.abspath("../data/s3021_discovery.txt")
    
    # 1. Leemos con 80 puntos por superficie (grilla de coseno)
    x, y_up, y_low = db.parse_uiuc_file(s3021_path, n_points=80)
    
    # 2. Aplicamos espesor de cola de 0.003 (lo que da 0.6 mm real para cuerda de 200 mm)
    te_half_thickness = 0.0015
    y_up_thick = y_up + (x * te_half_thickness)
    y_low_thick = y_low - (x * te_half_thickness)
    
    x_up_inv = np.flip(x)
    y_up_inv = np.flip(y_up_thick)
    
    x_low = x[1:]
    y_low = y_low_thick[1:]
    
    # Escala a 200 mm
    cuerda = 200.0
    
    with open(output_path, "w") as f:
        # Cabeceras de SpaceClaim
        f.write("3d=true\n")
        f.write("polyline=false\n")
        f.write("fit=false\n\n")
        
        # Extradós (Grupo 1)
        for xi, yi in zip(x_up_inv, y_up_inv):
            f.write(f"0.000000   {xi*cuerda:.6f}   {yi*cuerda:.6f}\n")
            
        # Separación
        f.write("\n")
        
        # Intradós (Grupo 2)
        f.write(f"0.000000   {x_up_inv[-1]*cuerda:.6f}   {y_up_inv[-1]*cuerda:.6f}\n")
        for xi, yi in zip(x_low, y_low):
            f.write(f"0.000000   {xi*cuerda:.6f}   {yi*cuerda:.6f}\n")
            
    print("Archivo para SpaceClaim con espesor de 0.6 mm generado en: " + output_path)
    print("Espesor en X=200: %.2f mm" % ((y_up_thick[-1] - y_low_thick[-1])*cuerda))

if __name__ == "__main__":
    generate_discovery_thick()
