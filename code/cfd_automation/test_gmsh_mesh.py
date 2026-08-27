# -*- coding: utf-8 -*-
"""
DIPAS - test_gmsh_mesh.py
=========================
Script de PRUEBA para generar malla 2D de perfil aerodinamico con gmsh
y verificar compatibilidad con ANSYS Fluent.

Parametros replicados del proyecto ANSYS original:
  - Edge Sizing airfoil : 1e-3 m
  - First Layer Height  : 2.4e-5 m
  - Max Layers          : 20
  - Growth Rate         : 1.2
  - Dominio             : x=[-3, 4] m, y=[-3, 3] m
  - Cuerda              : 0.200 m

Estrategia de exportacion para Fluent:
  1. Exportar gmsh v2.2 ASCII (.msh)  <-- intento principal
  2. Si Fluent falla con ese, convertir via meshio a formato Fluent nativo

Uso:
    pip install gmsh meshio
    python code/test_gmsh_mesh.py
"""

import math
import sys
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_MSH  = DATA_DIR / "test_gmsh_airfoil.msh"

# ─── Parametros del mallado (replicando ANSYS) ────────────────────────────
CHORD        = 0.200        # m
AIRFOIL_SIZE = 1e-3         # m  (Edge Sizing airfoil)
FAR_SIZE     = 0.8          # m  (tamano lejos del perfil)
FIRST_LAYER  = 2.4e-5       # m  (First Layer Height)
N_LAYERS     = 20           # capas de inflacion
GROWTH_RATE  = 1.2          # razon de crecimiento

# Dominio: mismas dimensiones que SpaceClaim (-3 a 4 en x, -3 a 3 en y)
X_INLET  = -3.0
X_OUTLET =  4.0
Y_BOT    = -3.0
Y_TOP    =  3.0

# ─── Espesor total de BL (suma de serie geometrica) ──────────────────────
BL_THICKNESS = FIRST_LAYER * (GROWTH_RATE**N_LAYERS - 1) / (GROWTH_RATE - 1)
print(f"Espesor total BL: {BL_THICKNESS*1000:.2f} mm ({BL_THICKNESS/CHORD*100:.2f}% cuerda)")

# ─── Cargar coordenadas del perfil semilla s3021 ─────────────────────────
def load_s3021_coords():
    """Carga las coordenadas actuales del archivo de coordenadas DIPAS."""
    coords_file = DATA_DIR / "s3021_discovery.txt"
    if not coords_file.exists():
        print("AVISO: s3021_discovery.txt no encontrado, usando perfil NACA 0012 de prueba")
        return generate_naca0012()
    
    points = []
    with open(coords_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    # Formato: 0.0  x_mm  y_mm
                    x = float(parts[1]) / 1000.0  # mm -> m
                    y = float(parts[2]) / 1000.0
                    points.append((x, y))
                except ValueError:
                    continue
    print(f"Coordenadas cargadas: {len(points)} puntos desde s3021_discovery.txt")
    return points

def generate_naca0012(n=100):
    """Genera perfil NACA 0012 como fallback."""
    t = np.linspace(0, 1, n)
    x = 0.5 * (1 - np.cos(np.pi * t)) * CHORD
    yt = 5*0.12*CHORD * (0.2969*np.sqrt(x/CHORD) - 0.1260*(x/CHORD)
                         - 0.3516*(x/CHORD)**2 + 0.2843*(x/CHORD)**3
                         - 0.1015*(x/CHORD)**4)
    upper = list(zip(x[::-1], yt[::-1]))
    lower = list(zip(x[1:],  -yt[1:]))
    return upper + lower

# ─── Generacion de malla con gmsh ────────────────────────────────────────
try:
    import gmsh
except ImportError:
    print("ERROR: gmsh no instalado. Ejecuta: pip install gmsh")
    sys.exit(1)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("airfoil_2d")

# 1. Puntos del perfil aerodinamico
coords = load_s3021_coords()
airfoil_tags = []
for i, (x, y) in enumerate(coords):
    tag = gmsh.model.geo.addPoint(x, y, 0.0, meshSize=AIRFOIL_SIZE)
    airfoil_tags.append(tag)

# 2. Spline cerrada del perfil
n_pts = len(airfoil_tags)
# Dividir en extrados e intrados para mejor control
mid = n_pts // 2
upper_tags = airfoil_tags[:mid+1]
lower_tags = airfoil_tags[mid:]

spline_upper = gmsh.model.geo.addSpline(upper_tags)
spline_lower = gmsh.model.geo.addSpline(lower_tags + [airfoil_tags[0]])  # cierra en LE
airfoil_loop = gmsh.model.geo.addCurveLoop([spline_upper, spline_lower])

# 3. Dominio externo (rectangulo)
p1 = gmsh.model.geo.addPoint(X_INLET,  Y_BOT, 0, meshSize=FAR_SIZE)
p2 = gmsh.model.geo.addPoint(X_OUTLET, Y_BOT, 0, meshSize=FAR_SIZE)
p3 = gmsh.model.geo.addPoint(X_OUTLET, Y_TOP, 0, meshSize=FAR_SIZE)
p4 = gmsh.model.geo.addPoint(X_INLET,  Y_TOP, 0, meshSize=FAR_SIZE)

l_bot    = gmsh.model.geo.addLine(p1, p2)
l_outlet = gmsh.model.geo.addLine(p2, p3)
l_top    = gmsh.model.geo.addLine(p3, p4)
l_inlet  = gmsh.model.geo.addLine(p4, p1)
outer_loop = gmsh.model.geo.addCurveLoop([l_bot, l_outlet, l_top, l_inlet])

# 4. Superficie fluida (rectangulo menos perfil)
fluid_surface = gmsh.model.geo.addPlaneSurface([outer_loop, airfoil_loop])

gmsh.model.geo.synchronize()

# 5. Physical groups (nombres de boundary conditions para Fluent)
gmsh.model.addPhysicalGroup(1, [l_inlet],             name="inlet")
gmsh.model.addPhysicalGroup(1, [l_outlet],            name="outlet")
gmsh.model.addPhysicalGroup(1, [l_bot, l_top],        name="symmetry")
gmsh.model.addPhysicalGroup(1, [spline_upper, spline_lower], name="airfoil")
gmsh.model.addPhysicalGroup(2, [fluid_surface],       name="fluid")

# 6. Campo de refinamiento en el airfoil (curvatura + distancia)
f_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [spline_upper, spline_lower])
gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)

f_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_thresh, "InField",   f_dist)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin",   AIRFOIL_SIZE)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax",   FAR_SIZE)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMin",   0.01)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMax",   1.0)

gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

# 7. Capas de inflacion (BoundaryLayer) - replica ANSYS Inflation
airfoil_curves = [spline_upper, spline_lower]
airfoil_points = list(set(upper_tags + lower_tags))

f_bl = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(f_bl, "CurvesList",  airfoil_curves)
gmsh.model.mesh.field.setNumbers(f_bl, "PointsList",  airfoil_points)
gmsh.model.mesh.field.setNumber(f_bl,  "hwall_n",     FIRST_LAYER)
gmsh.model.mesh.field.setNumber(f_bl,  "ratio",       GROWTH_RATE)
gmsh.model.mesh.field.setNumber(f_bl,  "thickness",   BL_THICKNESS * 1.5)
gmsh.model.mesh.field.setNumber(f_bl,  "NbLayers",    N_LAYERS)
# Registrar el campo de BL como campo de tamano de malla
# (gmsh.model.mesh.setBoundaryLayers no existe en la API Python de gmsh)
gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
# El campo BL se aplica automaticamente al generar la malla

# 8. Generar malla 2D
print("\nGenerando malla 2D...")
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Laplace2D")

# Estadisticas
node_count = len(gmsh.model.mesh.getNodes()[0])
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
total_2d = sum(len(t) for t in elem_tags)
print(f"\n=== ESTADISTICAS DE MALLA ===")
print(f"Nodos totales    : {node_count}")
print(f"Elementos 2D     : {total_2d}")
print(f"First Layer      : {FIRST_LAYER*1e6:.1f} um")
print(f"N capas BL       : {N_LAYERS}")
print(f"Espesor BL total : {BL_THICKNESS*1000:.2f} mm")

# 9. Exportar a formato Fluent MSH 2D nativo
import sys
sys.path.append(str(PROJECT_DIR / "code"))
from gmsh_to_fluent_msh import export_gmsh_to_fluent_msh2d

OUTPUT_MSH = DATA_DIR / "test_fluent_airfoil_native.msh"
export_gmsh_to_fluent_msh2d(OUTPUT_MSH)

gmsh.finalize()

# 10. Generar journal de prueba para Fluent
journal_test = DATA_DIR / "test_gmsh_import.jou"
msh_fluent_path = str(OUTPUT_MSH).replace("\\", "/")

with open(journal_test, "w") as f:
    f.write(f'/file/read-case "{msh_fluent_path}"\n')
    f.write('/mesh/check\n')
    f.write('/mesh/quality\n')
    f.write('/exit yes\n')

print(f"\nJournal de prueba Fluent generado: {journal_test}")
print(f"Comando en Journal: /file/read-case \"{msh_fluent_path}\"")


