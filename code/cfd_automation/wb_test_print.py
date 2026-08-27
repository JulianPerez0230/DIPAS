import sys, os
from pathlib import Path
sys.path.append(r"C:\Users\JULIAN\JunoWorkspace\projects\DIPAS\code")
sys.path.append(r"C:\Users\JULIAN\JunoWorkspace\projects\DIPAS\code\cfd_automation")
from dipas_engine import DIPASEngine

engine = DIPASEngine()
cands = engine.generate_airfoils(cl_target=1.1, cd_target=0.015, reynolds=100000, tc_target=0.12, eval_alpha=3.5)
cand = cands[0]

fluent_exe = r"C:\Program Files\ANSYS Inc\ANSYS Student\v261\fluent\ntbin\win64\fluent.exe"
data_dir = engine.data_dir

jou_batch = data_dir / "setup_batch.jou"
cas_path = data_dir / "current_airfoil.cas.h5"
if cas_path.exists():
    try:
        os.remove(cas_path)
    except:
        pass

print("Generando malla...")
from fluent_mesh_generator import generate_airfoil_mesh
import numpy as np

chord = 0.200
x = np.array(cand["x"])
y_u = np.array(cand["y_upper"])
y_l = np.array(cand["y_lower"])
x_coords = np.concatenate([np.flip(x), x[1:]]) * chord
y_coords = np.concatenate([np.flip(y_u), y_l[1:]]) * chord
coords_xy = list(zip(x_coords, y_coords))
msh_path = data_dir / "current_airfoil.msh"
generate_airfoil_mesh(coords_xy, str(msh_path), chord=chord)

msh_forward = str(msh_path).replace("\\", "/")
cas_forward = str(cas_path).replace("\\", "/")

jou_content = f"""/file/read-case "{msh_forward}"
/define/models/viscous/transition-sst yes
/report/reference-values/area {chord:.4f}
/report/reference-values/length {chord:.4f}
/report/reference-values/velocity 15.0
/file/write-case "{cas_forward}"
(exit)
"""
with open(jou_batch, "w", encoding="utf-8") as f:
    f.write(jou_content)

jou_interactive = data_dir / "setup_fluent_case.jou"
with open(jou_interactive, "w", encoding="utf-8") as f:
    f.write(f"""/file/read-case "{msh_forward}"
/define/models/viscous/transition-sst yes
/report/reference-values/area {chord:.4f}
/report/reference-values/length {chord:.4f}
/report/reference-values/velocity 15.0
/solve/report-definitions/add cd-report drag force-vector 1 0 thread-names airfoil () quit
/solve/report-definitions/add cl-report lift force-vector 0 1 thread-names airfoil () quit
""")
print("setup_fluent_case.jou escrito con exito!")

env = os.environ.copy()
env.pop("AWP_ROOT252", None)
env.pop("ANSYS252_DIR", None)

import subprocess
print("Ejecutando Fluent batch...")
res = subprocess.run([fluent_exe, "2d", "-g", "-t2", "-i", str(jou_batch)], cwd=str(data_dir), capture_output=True, text=True, timeout=35, env=env)
print("Returncode:", res.returncode)
print("STDOUT Ultimas lineas:\n", res.stdout[-1500:])
print("STDERR:\n", res.stderr)
print("cas_path exists:", cas_path.exists())
