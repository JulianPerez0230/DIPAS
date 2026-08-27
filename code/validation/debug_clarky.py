import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_builder import DatasetBuilder
from cst_generator import CSTParametrization

def debug():
    print("=== DEBAGUEANDO EL PERFIL CLARKY ===")
    
    db = DatasetBuilder()
    clarky_path = os.path.abspath("../data/seeds/clarky.dat")
    
    # 1. Intentar parsear
    print("\n1. Probando parser...")
    try:
        x, yu, yl = db.parse_uiuc_file(clarky_path)
        print(f"  Puntos leídos: x={len(x)}, yu={len(yu)}, yl={len(yl)}")
        print(f"  Rango x: [{x.min():.4f}, {x.max():.4f}]")
        print(f"  Rango yu: [{yu.min():.4f}, {yu.max():.4f}]")
        print(f"  Rango yl: [{yl.min():.4f}, {yl.max():.4f}]")
        
        # Comprobar NaNs
        print(f"  NaNs en x: {np.isnan(x).sum()}")
        print(f"  NaNs en yu: {np.isnan(yu).sum()}")
        print(f"  NaNs en yl: {np.isnan(yl).sum()}")
    except Exception as e:
        print(f"  [ERROR] Falló el parser: {e}")
        return

    # 2. Intentar ajustar CST
    print("\n2. Probando ajuste CST...")
    try:
        coefs_u, coefs_l = db.cst.fit_airfoil(x, yu, yl)
        print("  Coefs Upper:", np.round(coefs_u, 4))
        print("  Coefs Lower:", np.round(coefs_l, 4))
    except Exception as e:
        print(f"  [ERROR] Falló el ajuste CST: {e}")
        return

    # 3. Probando simulación de XFOIL de la variante
    print("\n3. Probando simulación de la semilla Clarky en XFOIL...")
    # Creamos un archivo temporal para simular manualmente
    airfoil_file = "temp_debug_clarky.dat"
    polar_file = "temp_debug_polar.txt"
    
    # Generamos coordenadas suaves a partir del ajuste
    x_coords, y_u_cst, y_l_cst = db.cst.generate_coordinates(coefs_u, coefs_l, n_points=100)
    db.xfoil.write_airfoil_file(x_coords, y_u_cst, y_l_cst, airfoil_file)
    
    # Escribimos los comandos de XFOIL para correr interactivamente y ver el output
    # pero sin redireccionar para ver si hay algún error
    commands = [
        "plop",
        "g",
        "",
        f"load {airfoil_file}",
        "oper",
        "visc 150000",
        "iter 50",
        "pacc",
        polar_file,
        "",
        "aseq 0 2 1",
        "pacc",
        "",
        "quit"
    ]
    
    input_file = "temp_debug_input.txt"
    with open(input_file, "w") as f:
        f.write("\n".join(commands) + "\n")
        
    print("  Corriendo XFOIL en consola interactiva para ver los mensajes de error...")
    os.system(f'"{db.xfoil.xfoil_path}" < "{input_file}"')
    
    # Limpieza
    for f in [airfoil_file, input_file, polar_file]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    debug()
