# -*- coding: utf-8 -*-
"""
DIPAS - fluent_case_builder.py
Automatiza la creación de la malla, configuración física y apertura
de ANSYS Fluent con el caso (.cas.h5) 100% precargado y visible.
"""

import os
import subprocess
import numpy as np
from pathlib import Path
from fluent_mesh_generator import generate_airfoil_mesh

def build_and_launch_fluent_case(candidate, data_dir, fluent_exe, reynolds=200000, alpha=3.0, chord=0.200):
    """
    1. Genera la malla 2D (.msh)
    2. Ejecuta Fluent batch para compilar el caso (.cas.h5) con física Transition SST
    3. Abre Fluent GUI directamente con el caso .cas.h5 cargado en pantalla.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # 1. Coordenadas a escala de cuerda
    x = np.array(candidate["x"])
    y_u = np.array(candidate["y_upper"])
    y_l = np.array(candidate["y_lower"])
    
    x_up_inv = np.flip(x)
    y_up_inv = np.flip(y_u)
    x_low = x[1:]
    y_low = y_l[1:]
    
    x_coords = np.concatenate([x_up_inv, x_low]) * chord
    y_coords = np.concatenate([y_up_inv, y_low]) * chord
    coords_xy = list(zip(x_coords, y_coords))
    
    msh_path = data_dir / "current_airfoil.msh"
    cas_path = data_dir / "current_airfoil.cas.h5"
    jou_batch_path = data_dir / "setup_batch.jou"
    
    # Generar malla Gmsh -> Fluent
    generate_airfoil_mesh(coords_xy, str(msh_path), chord=chord)
    
    # Velocidad de entrada para el Reynolds dado (nu = 1.5e-5 m2/s)
    v_inf = float(reynolds * 1.5e-5 / chord)
    rad = np.radians(alpha)
    vx = float(v_inf * np.cos(rad))
    vy = float(v_inf * np.sin(rad))
    
    msh_forward = str(msh_path).replace("\\", "/")
    cas_forward = str(cas_path).replace("\\", "/")
    
    # Journal para compilar caso .cas.h5
    jou_content = f"""/file/read-case "{msh_forward}"
/define/models/viscous/transition-sst yes
/report/reference-values/area {chord:.4f}
/report/reference-values/length {chord:.4f}
/report/reference-values/velocity {v_inf:.4f}
/solve/report-definitions/add cd-report drag force-vector {np.cos(rad):.6f} {np.sin(rad):.6f} thread-names airfoil () quit
/solve/report-definitions/add cl-report lift force-vector {-np.sin(rad):.6f} {np.cos(rad):.6f} thread-names airfoil () quit
/file/write-case "{cas_forward}"
(exit)
"""
    with open(jou_batch_path, "w", encoding="utf-8") as f:
        f.write(jou_content)
        
    jou_interactive_path = data_dir / "setup_fluent_case.jou"
    with open(jou_interactive_path, "w", encoding="utf-8") as f:
        f.write(f"""/file/read-case "{msh_forward}"
/define/models/viscous/transition-sst yes
/report/reference-values/area {chord:.4f}
/report/reference-values/length {chord:.4f}
/report/reference-values/velocity {v_inf:.4f}
/solve/report-definitions/add cd-report drag force-vector {np.cos(rad):.6f} {np.sin(rad):.6f} thread-names airfoil () quit
/solve/report-definitions/add cl-report lift force-vector {-np.sin(rad):.6f} {np.cos(rad):.6f} thread-names airfoil () quit
""")
        
    # Ejecución rápida en batch para generar el .cas.h5
    env = os.environ.copy()
    for old in ["AWP_ROOT252", "ANSYS252_DIR"]:
        env.pop(old, None)
        
    try:
        subprocess.run(
            [fluent_exe, "2d", "-g", "-t2", "-i", str(jou_batch_path)],
            cwd=str(data_dir),
            capture_output=True,
            timeout=25,
            env=env
        )
    except Exception as e:
        print(f"Advertencia en compilación de caso: {e}")
        
    # Si se generó el caso .cas.h5, abrir Fluent GUI cargando el caso directamente
    if cas_path.exists():
        gui_cmd = [fluent_exe, "2d", "-t4", "-case", str(cas_path)]
    else:
        # Fallback con journal interactivo
        gui_cmd = [fluent_exe, "2d", "-t4", "-i", str(jou_batch_path)]
        
    subprocess.Popen(gui_cmd, cwd=str(data_dir), env=env)
    return True
