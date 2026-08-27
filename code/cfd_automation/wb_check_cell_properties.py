# -*- coding: utf-8 -*-
# Corrected cell properties inspector using cell.GetPropertyValue(Name=prop_name)
import os

def check_cells():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/wb_cell_info.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando analisis detallado de propiedades de celdas...\n")
        try:
            Open(FilePath=project_path)
            system = GetSystem(Name="FFF")
            if not system:
                log.write(">> Error: No se encontro el sistema FFF.\n")
                return
                
            for cell in system.Cells:
                log.write("\n>> CELDA: %s (Type: %s)\n" % (cell.Name, cell.Type))
                try:
                    for prop_name in cell.GetProperties():
                        try:
                            val = cell.GetPropertyValue(Name=prop_name)
                            log.write("   - %s = %s\n" % (prop_name, str(val)))
                        except Exception as e_single:
                            log.write("   - %s = [Error reading: %s]\n" % (prop_name, str(e_single)))
                except Exception as e_props:
                    log.write("   No se pudieron listar propiedades: %s\n" % str(e_props))
                
        except Exception as e:
            log.write(">> Error general: " + str(e) + "\n")

check_cells()
