# -*- coding: utf-8 -*-
# Workbench Journal Script for DIPAS CFD Automation - Phase B (Polar Sweep) - Final Confirmed Version
import os
import math

def run_polar_sweep():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    output_csv = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/s3021_polar_cfd.csv"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/wb_polar_run.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando barrido polar por Rotacion Trigonometrica (Metodo Final de Enters)...\n")
        
        try:
            log.write(">> Cargando el proyecto: " + project_path + "\n")
            Open(FilePath=project_path)
            log.write(">> Proyecto cargado con exito.\n")
            
            system = GetSystem(Name="FFF")
            setup = system.GetContainer(ComponentName="Setup")
            
            # Iniciamos el editor de Fluent en segundo plano (Interactive=False)
            log.write(">> Iniciando el editor de Fluent en segundo plano...\n")
            setup.Edit(Interactive=False)
            log.write(">> Editor de Fluent iniciado con exito.\n")
            
            # Definimos los angulos a simular (en grados)
            alphas = [-2, 0, 2, 4, 6, 8, 10, 12]
            
            # Creamos/Limpiamos el archivo CSV de resultados
            with open(output_csv, "w") as f:
                f.write("alpha,cl,cd\n")
            log.write(">> Archivo CSV de salida inicializado.\n")
            
            for alpha in alphas:
                # Convertimos alpha a radianes
                alpha_rad = math.radians(alpha)
                
                # Componentes de direccion del viento
                dir_x = math.cos(alpha_rad)
                dir_y = math.sin(alpha_rad)
                
                log.write(">> Simulando alpha = %d (DirX = %.6f, DirY = %.6f)\n" % (alpha, dir_x, dir_y))
                
                # 1. Cambiar direccion del viento usando Scheme y saltos de linea reales en string multiline
                log.write("   Configurando direccion del viento...\n")
                cmd_vel = """(ti-menu-load-string "/define/boundary-conditions/velocity-inlet inlet







%.6f

%.6f





")""" % (dir_x, dir_y)
                setup.SendCommand(Command=cmd_vel)
                
                # 2. Inicializar e iterar (Forzamos inicializacion limpia sobre datos anteriores)
                log.write("   Inicializando campo de flujo...\n")
                setup.SendCommand(Command='(ti-menu-load-string "/solve/initialize/hyb-initialization yes")')
                log.write("   Corriendo 350 iteraciones...\n")
                setup.SendCommand(Command='(ti-menu-load-string "/solve/iterate 350")')
                
                # 3. Leer Cx y Cy del archivo de reporte generado por Fluent
                cd_file = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba_files/dp0/FFF/Fluent/cd_report-rfile.out"
                cl_file = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba_files/dp0/FFF/Fluent/cl_report-rfile.out"
                
                try:
                    with open(cd_file, "r") as r_file:
                        last_line = r_file.readlines()[-1].split()
                        cx_val = float(last_line[1])
                        
                    with open(cl_file, "r") as r_file:
                        last_line = r_file.readlines()[-1].split()
                        cy_val = float(last_line[1])
                        
                    # 4. Rotamos las fuerzas trigonometricamente en Python para obtener CL y CD reales
                    cd_val = cx_val * math.cos(alpha_rad) + cy_val * math.sin(alpha_rad)
                    cl_val = -cx_val * math.sin(alpha_rad) + cy_val * math.cos(alpha_rad)
                    
                    log.write("   [OK] Leido Cx=%.4f, Cy=%.4f -> Rotado: CL = %.4f, CD = %.4f\n" % (cx_val, cy_val, cl_val, cd_val))
                    
                    with open(output_csv, "a") as f:
                        f.write("%d,%.6f,%.6f\n" % (alpha, cl_val, cd_val))
                        
                except Exception as e_read:
                    log.write("   [ERROR] Al leer archivos de reporte: " + str(e_read) + "\n")
                    
            log.write(">> Cerrando Fluent...\n")
            setup.Exit()
            
            log.write(">> Guardando proyecto...\n")
            Save()
            log.write(">> [FIN] Barrido polar completado con exito.\n")
            
        except Exception as e:
            log.write(">> [EXCEPCION] Error general en el script: " + str(e) + "\n")

run_polar_sweep()
