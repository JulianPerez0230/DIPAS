# -*- coding: utf-8 -*-
# Script to kill all hung ANSYS and Fluent processes on Windows (Updated)
import os
import subprocess

def kill_ansys():
    # Lista completa de procesos de ANSYS, SpaceClaim y resolvedores de Fluent
    processes = [
        "Ans.Cadint.exe",
        "Ans.Modeler.exe",
        "Ans.Meshing.exe",
        "runwb2.exe",
        "AnsysFWW.exe",
        "ANSYS.exe",
        "Mapdl.exe",
        "fluent.exe",
        "fl2610.exe",
        "cx2610.exe",
        "cortex.exe",
        "SpaceClaim.exe",
        "FELLOWS.exe",
        "dxlauncher.exe"
    ]
    
    print(">> Limpiando procesos de ANSYS/Fluent colgados en segundo plano...")
    for proc in processes:
        try:
            # Ejecutamos taskkill de forma silenciosa
            subprocess.run(["taskkill", "/F", "/IM", proc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("   - Intento de detener: %s" % proc)
        except Exception as e:
            pass
    print(">> [OK] Todos los procesos de ANSYS y Fluent han sido limpiados de memoria.")

if __name__ == "__main__":
    kill_ansys()
