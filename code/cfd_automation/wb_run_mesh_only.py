# -*- coding: utf-8 -*-
# Script to verify geometry replacement and structured mesh update using correct DesignModeler format
import os

def run_mesh_only():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    # Usamos el archivo en formato de DesignModeler (ansys.txt)
    geom_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/s3021_ansys.txt"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/wb_mesh_only.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando prueba de actualizacion de malla estructurada con formato DesignModeler (ansys.txt)...\n")
        try:
            Open(FilePath=project_path)
            log.write(">> Proyecto cargado con exito.\n")
            
            system = GetSystem(Name="FFF")
            if not system:
                log.write(">> Error: No se encontro el sistema FFF.\n")
                return
                
            # 1. Cambiamos el archivo de origen de la geometria al de DesignModeler
            log.write(">> Reemplazando archivo de geometria...\n")
            geometry = system.GetContainer(ComponentName="Geometry")
            geometry.SetFile(FilePath=geom_path)
            log.write(">> Archivo asociado con exito.\n")
            
            # 2. Buscamos la celda de Malla (Mesh)
            log.write(">> Buscando celda de Malla...\n")
            mesh_cell = None
            for cell in system.Cells:
                if cell.Name == "Mesh":
                    mesh_cell = cell
                    break
                    
            if mesh_cell:
                log.write(">> Celda Mesh encontrada. Actualizando celda (mesh_cell.Update())...\n")
                mesh_cell.Update()
                log.write(">> Celda de Malla actualizada con exito.\n")
            else:
                log.write(">> Error: No se encontro la celda 'Mesh' en el sistema FFF.\n")
                return
            
            log.write(">> Guardando proyecto...\n")
            Save()
            log.write(">> [EXITO] Remallado completado y guardado.\n")
            
        except Exception as e:
            log.write(">> Error: " + str(e) + "\n")

run_mesh_only()
