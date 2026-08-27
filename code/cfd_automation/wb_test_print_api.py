# -*- coding: utf-8 -*-
# Diagnostic script to dump all global namespace objects in Workbench
import os

def dump_globals():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/wb_globals.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando volcado de espacio de nombres global...\n")
        try:
            Open(FilePath=project_path)
            log.write(">> Proyecto cargado con exito.\n")
            
            # Listamos todas las claves en el diccionario global
            global_keys = sorted(globals().keys())
            log.write(">> Cantidad de objetos globales: %d\n" % len(global_keys))
            for key in global_keys:
                log.write("   - %s\n" % key)
                
        except Exception as e:
            log.write(">> Error general: " + str(e) + "\n")

dump_globals()
