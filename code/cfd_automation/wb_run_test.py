# -*- coding: utf-8 -*-
# Workbench Journal Script for DIPAS CFD Automation - Phase A
import os

def run_automation():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    geom_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/s3021_discovery.txt"
    
    print(">> Cargando el proyecto de Workbench: " + project_path)
    Open(FilePath=project_path)
    
    # Buscamos el sistema de Fluent (por defecto es "FFF" en Workbench)
    system = GetSystem(Name="FFF")
    if not system:
        print("Error: No se encontro el sistema FFF (Fluid Flow Fluent) en el proyecto.")
        return
        
    print(">> Reemplazando la geometria con el perfil limpio: " + geom_path)
    geometry = system.GetContainer(ComponentName="Geometry")
    geometry.SetFile(FilePath=geom_path)
    
    print(">> Regenerando la malla con la nueva geometria...")
    mesh = system.GetContainer(ComponentName="Mesh")
    mesh.Update()
    
    print(">> Ejecutando la simulacion en ANSYS Fluent...")
    # Update en Setup/Solution ejecuta Fluent con la configuracion cargada (350 iteraciones, etc.)
    setup = system.GetContainer(ComponentName="Setup")
    setup.Update()
    
    print(">> Guardando el proyecto...")
    Save()
    print("[OK] Automatizacion de la Fase A finalizada con exito.")

# Ejecutamos la funcion
run_automation()
