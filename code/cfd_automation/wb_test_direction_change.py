# -*- coding: utf-8 -*-
# Test to verify velocity-inlet configuration using 5 enters to exit the turbulence prompts
import os

def test_direction():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/direction_test.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando prueba fisica con 13-Enter sequence...\n")
        try:
            Open(FilePath=project_path)
            system = GetSystem(Name="FFF")
            setup = system.GetContainer(ComponentName="Setup")
            
            setup.Edit(Interactive=False)
            
            # Comando con 7 Enters, vector X, 1 Enter, vector Y, y 5 Enters para salir
            log.write(">> Enviando comando 13-Enter al inlet...\n")
            cmd = """(ti-menu-load-string "/define/boundary-conditions/velocity-inlet inlet







0.999391

-0.034899





")"""
            setup.SendCommand(Command=cmd)
            
            # Inicializamos y corremos 120 iteraciones
            log.write(">> Inicializando...\n")
            setup.SendCommand(Command='(ti-menu-load-string "/solve/initialize/hyb-initialization yes")')
            log.write(">> Corriendo 120 iteraciones...\n")
            setup.SendCommand(Command='(ti-menu-load-string "/solve/iterate 120")')
            
            # Leemos las fuerzas resultantes
            cd_file = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba_files/dp0/FFF/Fluent/cd_report-rfile.out"
            cl_file = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba_files/dp0/FFF/Fluent/cl_report-rfile.out"
            
            with open(cd_file, "r") as r_file:
                cx_val = float(r_file.readlines()[-1].split()[1])
            with open(cl_file, "r") as r_file:
                cy_val = float(r_file.readlines()[-1].split()[1])
                
            log.write(">> Resultados leidos a la iteracion 120: Cx = %.5f, Cy = %.5f\n" % (cx_val, cy_val))
            if cy_val < 0.25:
                log.write(">> [EXITO] Las fuerzas cambiaron para angulo negativo! Cy = %.4f (Esperado para -2 deg)\n" % cy_val)
            else:
                log.write(">> [FALLO] Cy = %.4f sigue dando el valor de la corrida previa.\n" % cy_val)
                
            setup.Exit()
        except Exception as e:
            log.write(">> Error: " + str(e) + "\n")

test_direction()
