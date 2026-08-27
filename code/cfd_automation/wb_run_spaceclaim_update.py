# -*- coding: utf-8 -*-
# Flat script executing inside ANSYS Workbench to update SpaceClaim geometry and Meshing (API Direct Export)
import os

# Helper de logging compatible con IronPython (Python 2.7)
def log_message(msg):
    try:
        f = open(r"C:\Users\JULIAN\JunoWorkspace\projects\DIPAS\data\workbench_script.log", "a")
        f.write(msg + "\n")
        f.close()
    except:
        pass

# Inicializar archivo de log limpio
try:
    f = open(r"C:\Users\JULIAN\JunoWorkspace\projects\DIPAS\data\workbench_script.log", "w")
    f.write("=== LOG DE EJECUCION WORKBENCH ===\n")
    f.close()
except:
    pass

log_message(">> Iniciando script de actualizacion...")

# 0. Open the project with forward slashes (ANSYS TUI standard)
project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
log_message(">> Abriendo proyecto: " + project_path)

try:
    Open(FilePath=project_path)
    log_message(">> Proyecto abierto con exito.")
except Exception as e_open:
    log_message(">> ERROR al abrir proyecto: " + str(e_open))
    raise

# 1. Dynamically find the Fluent system in the project
system = None
try:
    all_systems = GetAllSystems()
    log_message(">> Sistemas encontrados en el proyecto: " + str([s.Name for s in all_systems]))
    for s in all_systems:
        if "Fluent" in s.DisplayText or s.Name.startswith("FFF") or s.Name.startswith("SYS"):
            system = s
            break
except Exception as e_sys:
    log_message(">> ERROR al buscar sistemas: " + str(e_sys))
    raise

if not system:
    log_message(">> ERROR: No se encontro ningun sistema de Fluent.")
else:
    log_message(">> Sistema seleccionado: " + system.Name + " (" + system.DisplayText + ")")
    
    try:
        # 2. Open SpaceClaim with standard Interactive=False arguments
        geometry_container = system.GetContainer(ComponentName="Geometry")
        log_message(">> Editando componente de Geometria (SpaceClaim)...")
        geometry_container.Edit(Interactive=False)
        log_message(">> SpaceClaim inicializado en segundo plano.")
        
        # 3. Define instructions for SpaceClaim
        spaceclaim_cmd = """
# Python Script for SpaceClaim - Version 10
# NOTA: en SendCommand, el API de SpaceClaim ya esta pre-cargado en el namespace global.
# NO se hace import clr ni from SpaceClaim.Api import * - eso tira error.
import System

LOG = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/geometry_verify.log"

def sc_log(msg):
    try:
        System.IO.File.AppendAllText(LOG, msg + "\\n")
    except:
        pass

sc_log("=== INICIO SPACECLAIM SCRIPT ===")

# A. LIMPIEZA COMPLETA
part = GetRootPart()
n_b  = len(list(part.Bodies))
n_c  = len(list(part.Curves))
n_cp = len(list(part.Components))
sc_log("ANTES_DELETE: bodies=" + str(n_b) + " curves=" + str(n_c) + " comps=" + str(n_cp))

try:
    all_bodies = list(part.Bodies)
    if all_bodies:
        Delete.Execute(Selection.Create(all_bodies))
        sc_log("Bodies borrados: " + str(len(all_bodies)))
except Exception as ex:
    sc_log("ERROR borrando bodies: " + str(ex))

try:
    all_curves = list(part.Curves)
    if all_curves:
        Delete.Execute(Selection.Create(all_curves))
        sc_log("Curves borrados: " + str(len(all_curves)))
except Exception as ex:
    sc_log("ERROR borrando curves: " + str(ex))

try:
    all_comps = list(part.Components)
    if all_comps:
        Delete.Execute(Selection.Create(all_comps))
        sc_log("Components borrados: " + str(len(all_comps)))
except Exception as ex:
    sc_log("ERROR borrando components: " + str(ex))

try:
    for ns in list(part.NamedSelections):
        Delete.Execute(Selection.Create(ns))
except:
    pass

n_b2 = len(list(part.Bodies))
n_c2 = len(list(part.Curves))
n_cp2 = len(list(part.Components))
sc_log("DESPUES_DELETE: bodies=" + str(n_b2) + " curves=" + str(n_c2) + " comps=" + str(n_cp2))

# B. Leer primera linea de coordenadas para confirmar que el archivo cambio
coords_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/s3021_discovery.txt"
try:
    lines = System.IO.File.ReadAllLines(coords_path)
    sc_log("COORDS_LINE_4: " + (lines[4] if len(lines) > 4 else "???"))
except Exception as ex:
    sc_log("ERROR leyendo coords: " + str(ex))

# C. Importar nuevas coordenadas
sc_log("Ejecutando DocumentInsert...")
try:
    DocumentInsert.Execute(coords_path)
    sc_log("DocumentInsert OK")
except Exception as ex:
    sc_log("ERROR en DocumentInsert: " + str(ex))

n_b3 = len(list(part.Bodies))
n_c3 = len(list(part.Curves))
n_cp3 = len(list(part.Components))
sc_log("POST_IMPORT: bodies=" + str(n_b3) + " curves=" + str(n_c3) + " comps=" + str(n_cp3))

# D. Fill the curves to create the airfoil face
try:
    curves = [c for c in part.Curves]
    sc_log("Curves para Fill: " + str(len(curves)))
    selection = Selection.Create(curves)
    secondarySelection = Selection.Empty()
    options = FillOptions()
    result = Fill.Execute(selection, secondarySelection, options, FillMode.Sketch, None)
    sc_log("Fill OK")
except Exception as ex:
    sc_log("ERROR en Fill: " + str(ex))

# E. Create fluid domain rectangle on PlaneXY
try:
    ViewHelper.SetSketchPlane(Plane.PlaneXY)
    point1 = Point2D.Create(MM(-3000), MM(-3000))
    point2 = Point2D.Create(MM(4000),  MM(-3000))
    point3 = Point2D.Create(MM(4000),  MM(3000))
    SketchRectangle.Create(point1, point2, point3)
    ViewHelper.SetViewMode(InteractionMode.Solid, None)
    sc_log("Rectangulo creado OK")
except Exception as ex:
    sc_log("ERROR creando rectangulo: " + str(ex))

# F. Subtract airfoil from rectangle
try:
    bodies = [b for b in part.Bodies]
    sc_log("Bodies para booleano: " + str(len(bodies)))
    if len(bodies) >= 2:
        rectangle_body = max(bodies, key=lambda b: max(f.Area for f in b.Faces))
        airfoil_body   = min(bodies, key=lambda b: max(f.Area for f in b.Faces))
        airfoil_face   = airfoil_body.Faces[0]
        ProjectToSolid.Execute(Selection.Create(airfoil_face), Selection.Empty(), Selection.Empty())
        Delete.Execute(Selection.Create(airfoil_body))
        rect_faces = [f for f in rectangle_body.Faces]
        outer_face = max(rect_faces, key=lambda f: f.Area)
        for face in rect_faces:
            if face != outer_face:
                Delete.Execute(Selection.Create(face))
        sc_log("Booleano OK")

        # G. Named Selections
        inlet_edges = []
        outlet_edges = []
        symmetry_edges = []
        airfoil_edges = []
        for edge in outer_face.Edges:
            pos1 = edge.Shape.StartVertex.Position
            pos2 = edge.Shape.EndVertex.Position
            mid_x = (pos1.X + pos2.X) / 2.0
            mid_y = (pos1.Y + pos2.Y) / 2.0
            if abs(mid_x - (-3.0)) < 1e-3:
                inlet_edges.append(edge)
            elif abs(mid_x - 4.0) < 1e-3:
                outlet_edges.append(edge)
            elif abs(mid_y - 3.0) < 1e-3 or abs(mid_y - (-3.0)) < 1e-3:
                symmetry_edges.append(edge)
            else:
                airfoil_edges.append(edge)
        NamedSelection.Create(Selection.Create(outer_face), Selection.Empty(), "fluid")
        if inlet_edges:
            NamedSelection.Create(Selection.Create(inlet_edges), Selection.Empty(), "inlet")
        if outlet_edges:
            NamedSelection.Create(Selection.Create(outlet_edges), Selection.Empty(), "outlet")
        if symmetry_edges:
            NamedSelection.Create(Selection.Create(symmetry_edges), Selection.Empty(), "symmetry")
        if airfoil_edges:
            NamedSelection.Create(Selection.Create(airfoil_edges), Selection.Empty(), "airfoil")
        sc_log("Named Selections OK")
    else:
        sc_log("ERROR: menos de 2 bodies para booleano (" + str(len(bodies)) + ")")
except Exception as ex:
    sc_log("ERROR en booleano/NS: " + str(ex))

sc_log("=== FIN SPACECLAIM SCRIPT ===")
"""
        
        log_message(">> Enviando comandos de geometria a SpaceClaim...")
        geometry_container.SendCommand(Language="Python", Command=spaceclaim_cmd)
        log_message(">> Geometry update complete in SpaceClaim.")

        # CRITICO: cerrar SpaceClaim para que Workbench registre los cambios de geometria
        # Sin este Exit(), el grafo de dependencias NO propaga la nueva geometria al Meshing
        log_message(">> Cerrando SpaceClaim para confirmar cambios de geometria...")
        geometry_container.Exit()
        log_message(">> SpaceClaim cerrado. Geometria committed al proyecto.")

        # 4. Open Meshing in batch mode — ahora si ve la geometria actualizada
        mesh_container = system.GetContainer(ComponentName="Mesh")
        log_message(">> Abriendo ANSYS Meshing en segundo plano...")
        mesh_container.Edit(Interactive=False)
        log_message(">> ANSYS Meshing inicializado.")
        
        # 5. Define Mechanical Scripting Command to generate and write the Fluent input file
        mesh_cmd = """
# Mechanical/Meshing Script (Python)
mesh = DataModel.Project.Model.Mesh

# CRITICO: limpiar la malla cacheada antes de regenerar
# Sin esto, WriteFluentInputFile exporta la malla vieja aunque la geometria cambio
try:
    mesh.ClearGeneratedData()
except:
    pass

# Regenerar la malla desde la geometria actualizada de SpaceClaim
mesh.GenerateMesh()

# Export Fluent input file (.msh) directly to the destination path
mesh_path = r"C:\\Users\\JULIAN\\JunoWorkspace\\projects\\DIPAS\\archive\\Simulacion_perfil_prueba_files\\dp0\\FFF-1\\MECH\\FFF-1.msh"
mesh.InternalObject.WriteFluentInputFile(mesh_path)
"""
        
        log_message(">> Enviando comandos de generacion y exportacion de malla...")
        mesh_container.SendCommand(Language="Python", Command=mesh_cmd)
        log_message(">> Malla exportada directamente a FFF-1.msh.")
        
        # 6. Exit Meshing application
        log_message(">> Cerrando ANSYS Meshing...")
        mesh_container.Exit()
        
        # 7. Save project (overwriting safely)
        log_message(">> Guardando cambios del proyecto...")
        Save(Overwrite=True)
        log_message(">> [SUCCESS] Proyecto guardado y actualizado con exito.")
        
    except Exception as e_proc:
        log_message(">> ERROR durante el procesamiento: " + str(e_proc))
        raise
