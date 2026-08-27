# -*- coding: utf-8 -*-
"""
DIPAS - fluent_mesh_generator.py
================================
Generador automatico y ultrarrapido de mallas 2D en formato nativo ANSYS Fluent MSH
utilizando Gmsh API y conversion directa.

Parametros ajustados al estandar aerodinamico de alta fidelidad:
  - Cuerda c = 0.200 m (200 mm)
  - Edge sizing airfoil = 1 mm (0.001 m)
  - Dominio fluido: X = [-3.0, 4.0] m, Y = [-3.0, 3.0] m
  - Capa limite (Inflation):
      * First layer height: 2.4e-5 m (y+ ~ 1 para Transition SST)
      * Growth rate: 1.2
      * Max layers: 20
"""

import numpy as np
from pathlib import Path
import gmsh
from gmsh_to_fluent_msh import export_gmsh_to_fluent_msh2d

def generate_airfoil_mesh(coords_xy, output_msh_path, chord=0.200):
    """
    coords_xy: lista o array Nx2 de coordenadas (x, y) en metros [0, chord]
    output_msh_path: ruta destino del archivo .msh para Fluent
    """
    output_msh_path = Path(output_msh_path)
    output_msh_path.parent.mkdir(exist_ok=True)
    
    # Parametros de malla
    AIRFOIL_SIZE = 0.001        # 1 mm en superficie
    FAR_SIZE     = 0.8          # 0.8 m en bordes lejanos
    FIRST_LAYER  = 2.4e-5       # 24 um
    N_LAYERS     = 20
    GROWTH_RATE  = 1.2
    
    X_INLET  = -3.0
    X_OUTLET =  4.0
    Y_BOT    = -3.0
    Y_TOP    =  3.0
    
    BL_THICKNESS = FIRST_LAYER * (GROWTH_RATE**N_LAYERS - 1) / (GROWTH_RATE - 1)
    
    # Inicializar Gmsh en modo silencioso
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) # Silencioso
    gmsh.model.add("airfoil_model")
    
    # 1. Crear puntos del perfil
    airfoil_tags = []
    for x, y in coords_xy:
        tag = gmsh.model.geo.addPoint(float(x), float(y), 0.0, meshSize=AIRFOIL_SIZE)
        airfoil_tags.append(tag)
        
    # 2. Spline del perfil
    n_pts = len(airfoil_tags)
    mid = n_pts // 2
    upper_tags = airfoil_tags[:mid+1]
    lower_tags = airfoil_tags[mid:]
    
    spline_upper = gmsh.model.geo.addSpline(upper_tags)
    spline_lower = gmsh.model.geo.addSpline(lower_tags + [airfoil_tags[0]])
    airfoil_loop = gmsh.model.geo.addCurveLoop([spline_upper, spline_lower])
    
    # 3. Rectangulo de dominio exterior
    p1 = gmsh.model.geo.addPoint(X_INLET,  Y_BOT, 0, meshSize=FAR_SIZE)
    p2 = gmsh.model.geo.addPoint(X_OUTLET, Y_BOT, 0, meshSize=FAR_SIZE)
    p3 = gmsh.model.geo.addPoint(X_OUTLET, Y_TOP, 0, meshSize=FAR_SIZE)
    p4 = gmsh.model.geo.addPoint(X_INLET,  Y_TOP, 0, meshSize=FAR_SIZE)
    
    l_bot    = gmsh.model.geo.addLine(p1, p2)
    l_outlet = gmsh.model.geo.addLine(p2, p3)
    l_top    = gmsh.model.geo.addLine(p3, p4)
    l_inlet  = gmsh.model.geo.addLine(p4, p1)
    outer_loop = gmsh.model.geo.addCurveLoop([l_bot, l_outlet, l_top, l_inlet])
    
    # 4. Superficie fluida
    fluid_surface = gmsh.model.geo.addPlaneSurface([outer_loop, airfoil_loop])
    gmsh.model.geo.synchronize()
    
    # 5. Physical groups
    gmsh.model.addPhysicalGroup(1, [l_inlet],             name="inlet")
    gmsh.model.addPhysicalGroup(1, [l_outlet],            name="outlet")
    gmsh.model.addPhysicalGroup(1, [l_bot, l_top],        name="symmetry")
    gmsh.model.addPhysicalGroup(1, [spline_upper, spline_lower], name="airfoil")
    gmsh.model.addPhysicalGroup(2, [fluid_surface],       name="fluid")
    
    # 6. Refinamiento en perfil
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
    
    # 7. Capa limite
    airfoil_curves = [spline_upper, spline_lower]
    airfoil_points = list(set(upper_tags + lower_tags))
    f_bl = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(f_bl, "CurvesList",  airfoil_curves)
    gmsh.model.mesh.field.setNumbers(f_bl, "PointsList",  airfoil_points)
    gmsh.model.mesh.field.setNumber(f_bl,  "hwall_n",     FIRST_LAYER)
    gmsh.model.mesh.field.setNumber(f_bl,  "ratio",       GROWTH_RATE)
    gmsh.model.mesh.field.setNumber(f_bl,  "thickness",   BL_THICKNESS * 1.5)
    gmsh.model.mesh.field.setNumber(f_bl,  "NbLayers",    N_LAYERS)
    
    # 8. Generar y optimizar malla
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    
    # 9. Exportar a formato Fluent nativo
    export_gmsh_to_fluent_msh2d(output_msh_path)
    gmsh.finalize()
    return str(output_msh_path)
