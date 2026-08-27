# -*- coding: utf-8 -*-
# Corrected diagnostic script to list all registered files using Project/project global
import os

def list_files():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/wb_all_files.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando listado de archivos en el proyecto Workbench...\n")
        try:
            Open(FilePath=project_path)
            log.write(">> Proyecto cargado con exito.\n")
            
            # Intentamos obtener los archivos a traves de Project o project
            files = None
            try:
                files = Project.GetFiles()
                log.write(">> Obtenido a traves de 'Project'.\n")
            except Exception as e1:
                log.write("   Fallo 'Project': %s\n" % str(e1))
                try:
                    files = project.GetFiles()
                    log.write(">> Obtenido a traves de 'project'.\n")
                except Exception as e2:
                    log.write("   Fallo 'project': %s\n" % str(e2))
            
            if files:
                log.write(">> Cantidad de archivos encontrados: %d\n" % len(files))
                for idx, file_obj in enumerate(files):
                    log.write("[%d] File:\n" % (idx+1))
                    try:
                        log.write("    - Path: %s\n" % file_obj.Path)
                    except:
                        pass
                    try:
                        log.write("    - Type: %s\n" % file_obj.Type)
                    except:
                        pass
            else:
                log.write(">> [ERROR] No se pudo obtener la lista de archivos.\n")
                    
        except Exception as e:
            log.write(">> Error general: " + str(e) + "\n")

list_files()
