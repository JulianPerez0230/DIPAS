# -*- coding: utf-8 -*-
# Python Script to be executed directly inside SpaceClaim Script Editor - Version 7 (Added Fluid Face Group)
import clr
clr.AddReference("SpaceClaim.Api.V261")
from SpaceClaim.Api.V261 import *

# A. Limpiamos de forma correcta todos los cuerpos y curvas previas
part = GetRootPart()
all_bodies = [b for b in part.Bodies]
all_curves = [c for c in part.Curves]
if all_bodies or all_curves:
    Delete.Execute(Selection.Create(all_bodies + all_curves))

# Limpiamos grupos (Named Selections) anteriores
try:
    for ns in list(part.NamedSelections):
        Delete.Execute(Selection.Create(ns))
except:
    pass

# B. Importamos el nuevo archivo de coordenadas de SpaceClaim
coords_path = r"C:\Users\JULIAN\JunoWorkspace\projects\DIPAS\data\s3021_discovery.txt"
DocumentInsert.Execute(coords_path)

# C. Rellenamos el perfil alar (Fill) para crear la superficie del perfil
curves = [c for c in part.Curves]
selection = Selection.Create(curves)
secondarySelection = Selection.Empty()
options = FillOptions()
result = Fill.Execute(selection, secondarySelection, options, FillMode.Sketch, None)

# D. Creamos el rectangulo del dominio fluido en el plano XY (mismo plano del perfil alar)
# Seteamos el plano de boceto en XY
ViewHelper.SetSketchPlane(Plane.PlaneXY)

# Dibujamos las esquinas exactas del rectangulo de 7.0m x 6.0m en coordenadas XY
point1 = Point2D.Create(MM(-3000), MM(-3000)) # Bottom-Left
point2 = Point2D.Create(MM(4000), MM(-3000))  # Bottom-Right
point3 = Point2D.Create(MM(4000), MM(3000))   # Top-Right
SketchRectangle.Create(point1, point2, point3)

# Salimos al modo 3D (esto nos genera dos cuerpos de superficie en el plano XY)
ViewHelper.SetViewMode(InteractionMode.Solid, None)

# E. Identificamos los dos cuerpos: el Rectangulo (grande) y el Perfil (pequeño)
bodies = [b for b in part.Bodies]
if len(bodies) >= 2:
    rectangle_body = max(bodies, key=lambda b: max(f.Area for f in b.Faces))
    airfoil_body = min(bodies, key=lambda b: max(f.Area for f in b.Faces))
    
    airfoil_face = airfoil_body.Faces[0]
    
    # 1. Proyectamos el ala sobre el rectangulo para dividir su cara (hacer el estampado)
    ProjectToSolid.Execute(Selection.Create(airfoil_face), Selection.Empty(), Selection.Empty())
    
    # 2. Eliminamos el cuerpo original del ala flotante
    Delete.Execute(Selection.Create(airfoil_body))
    
    # 3. Ahora el rectangulo tiene 2 caras. Eliminamos la interna (creando el hueco)
    rect_faces = [f for f in rectangle_body.Faces]
    outer_face = max(rect_faces, key=lambda f: f.Area)
    
    for face in rect_faces:
        if face != outer_face:
            Delete.Execute(Selection.Create(face))
            
    # F. Clasificacion matematica de los bordes para Named Selections (Groups) en la cara remanente
    inlet_edges = []
    outlet_edges = []
    symmetry_edges = []
    airfoil_edges = []
    
    for edge in outer_face.Edges:
        pos1 = edge.Shape.StartVertex.Position
        pos2 = edge.Shape.EndVertex.Position
        
        mid_x = (pos1.X + pos2.X) / 2.0
        mid_y = (pos1.Y + pos2.Y) / 2.0
        
        # Filtramos con tolerancia de 1 mm (0.001 metros)
        if abs(mid_x - (-3.0)) < 1e-3:
            inlet_edges.append(edge)
        elif abs(mid_x - 4.0) < 1e-3:
            outlet_edges.append(edge)
        elif abs(mid_y - 3.0) < 1e-3 or abs(mid_y - (-3.0)) < 1e-3:
            symmetry_edges.append(edge)
        else:
            airfoil_edges.append(edge)
            
    # G. Creamos las Named Selections en el panel de grupos (incluyendo la CARA del fluido)
    # Grupo de la cara del fluido (para el Inflation)
    NamedSelection.Create(Selection.Create(outer_face), Selection.Empty(), "fluid")
    
    # Grupos de los bordes
    if inlet_edges:
        NamedSelection.Create(Selection.Create(inlet_edges), Selection.Empty(), "inlet")
    if outlet_edges:
        NamedSelection.Create(Selection.Create(outlet_edges), Selection.Empty(), "outlet")
    if symmetry_edges:
        NamedSelection.Create(Selection.Create(symmetry_edges), Selection.Empty(), "symmetry")
    if airfoil_edges:
        NamedSelection.Create(Selection.Create(airfoil_edges), Selection.Empty(), "airfoil")

print(">> Reconstruccion completa con grupo 'fluid' exitosa.")
