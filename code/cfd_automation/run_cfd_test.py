# -*- coding: utf-8 -*-
# Test script to run a 2-angle Fluent polar sweep using a single global report file
import os
import subprocess
import sys

def run_cfd_test():
    # Rutas del proyecto
    data_dir = r"C:\Users\JULIAN\JunoWorkspace\projects\DIPAS\data"
    mesh_path = r"C:\Users\JULIAN\JunoWorkspace\projects\DIPAS\archive\Simulacion_perfil_prueba_files\dp0\FFF-1\MECH\FFF-1.msh"
    
    # Comprobar que la malla existe
    if not os.path.exists(mesh_path):
        print(f"Error: No se encontro la malla exportada en: {mesh_path}")
        return
        
    journal_path = os.path.join(data_dir, "run_test_polar.jou")
    fluent_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v261\fluent\ntbin\win64\fluent.exe"
    
    # Rutas para guardar los reportes polares
    cd_polar_path = os.path.join(data_dir, "cd_polar.out")
    cl_polar_path = os.path.join(data_dir, "cl_polar.out")
    
    # Limpiamos archivos de reportes anteriores si existen
    for f_path in [cd_polar_path, cl_polar_path]:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except Exception as e:
                pass
 
    print(">> Creando journal para barrido polar de 2 angulos (Alpha = -2 y 0)...")
    
    mesh_path_fluent = mesh_path.replace("\\", "/")
    cd_polar_fluent = cd_polar_path.replace("\\", "/")
    cl_polar_fluent = cl_polar_path.replace("\\", "/")
    
    # Escribimos el Journal
    # Definimos los reportes una sola vez al principio
    # Luego cambiamos el inlet, re-inicializamos (respondiendo "ok") e iteramos para cada angulo
    journal_content = f"""/file/read-case "{mesh_path_fluent}"
/define/models/viscous/transition-sst yes
/report/reference-values/area 0.2
/report/reference-values/length 0.2
/report/reference-values/velocity 10.95
/solve/report-definitions/add cd-report drag force-vector 1 0 thread-names airfoil () quit
/solve/report-definitions/add cl-report lift force-vector 0 1 thread-names airfoil () quit
/solve/report-files/add cd-file report-defs cd-report () file-name "{cd_polar_fluent}" ()
/solve/report-files/add cl-file report-defs cl-report () file-name "{cl_polar_fluent}" ()

; --- ANGULO 0 (Alpha = -2) ---
/define/boundary-conditions/velocity-inlet inlet
yes
yes
no
10.95
no
0
no
0.999391
no
-0.034899
no
no
yes
no
1
0.1
10
/solve/initialize/hyb-initialization
/solve/iterate 350

; --- ANGULO 1 (Alpha = 0) ---
/define/boundary-conditions/velocity-inlet inlet
yes
yes
no
10.95
no
0
no
1.000000
no
0.000000
no
no
yes
no
1
0.1
10
/solve/initialize/hyb-initialization
ok
/solve/iterate 350

/exit yes
"""
    
    with open(journal_path, "w") as f:
        f.write(journal_content)
        
    print(f"Journal creado con exito en: {journal_path}")
    print(">> Ejecutando Fluent en segundo plano...")
    
    command = [
        fluent_path,
        "2ddp",
        "-g",
        "-r26.1.0",
        "-shortcut",
        f"-i{journal_path}"
    ]
    
    log_path = os.path.join(data_dir, "fluent_execution.log")
    
    try:
        with open(log_path, "w") as log_f:
            subprocess.run(command, cwd=data_dir, stdout=log_f, stderr=log_f, text=True, timeout=300)
        print(">> Ejecucion finalizada.")
    except Exception as e:
        print(f"Error al lanzar Fluent: {e}")
        return

    # Leemos y mostramos el contenido de cd_polar.out para ver la estructura
    if os.path.exists(cd_polar_path):
        print("\n" + "="*50)
        print("   CONTENIDO DE cd_polar.out (ULTIMAS 20 LINEAS)")
        print("="*50)
        with open(cd_polar_path, "r") as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.strip())
        print("="*50)
        
        # Parseamos buscando las lineas con iteracion 350
        results = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    it = int(parts[1])
                    val = float(parts[2])
                    if it == 350:
                        results.append(val)
                except ValueError:
                    continue
        print(f"Valores de arrastre (CD) encontrados para cada angulo: {results}")

if __name__ == "__main__":
    run_cfd_test()
