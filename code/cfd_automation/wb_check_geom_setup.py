# -*- coding: utf-8 -*-
# Diagnostic script to check geometry component file association and properties in Workbench
import os

def check_geom():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/wb_geom_info.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando analisis de estructura de Geometria en Workbench...\n")
        try:
            Open(FilePath=project_path)
            system = GetSystem(Name="FFF")
            if not system:
                log.write(">> Error: No se encontro el sistema FFF.\n")
                return
                
            geometry = system.GetContainer(ComponentName="Geometry")
            log.write(">> Componente Geometria obtenido.\n")
            
            # Intentamos listar los archivos asociados a la geometria
            log.write(">> Archivos asociados al componente Geometria:\n")
            try:
                for file_obj in geometry.GetFiles():
                    log.write("   - File Path: %s\n" % file_obj.Path)
            except Exception as e_files:
                log.write("   No se pudo llamar a GetFiles(): %s\n" % str(e_files))
                
            # Intentamos obtener la propiedad del archivo fuente
            log.write(">> Propiedades del componente Geometria:\n")
            try:
                # Listamos las propiedades disponibles para depurar
                for prop in geometry.GetProperties():
                    log.write("   - Prop: %s = %s\n" % (prop.Name, prop.Value))
            except Exception as e_props:
                log.write("   No se pudo listar propiedades: %s\n" % str(e_props))
                
        except Exception as e:
            log.write(">> Error general: " + str(e) + "\n")

check_geom()
