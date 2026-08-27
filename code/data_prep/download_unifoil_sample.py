# -*- coding: utf-8 -*-
"""
Descarga los archivos de análisis de muestras de UniFoil (flag='sample')
para obtener los coeficientes aerodinámicos Cl/Cd necesarios para entrenar
el modelo sustituto (Surrogate) con transferencia de aprendizaje completa.
"""

import sys
import os
from pathlib import Path

# Asegurar importación de módulos en la raíz del proyecto
SCRIPT_DIR = Path(__file__).parent.absolute()

def main():
    print(">> Cargando el módulo GetData de UniFoil...")
    try:
        from unifoil import GetData
    except ImportError as e:
        print(f"❌ Error al importar unifoil. {e}")
        sys.exit(1)
        
    print(">> Iniciando descarga de análisis aerodinámicos de UniFoil (flag='sample')...")
    data_root = SCRIPT_DIR.parent
    gd = GetData(data_root=str(data_root))
    
    try:
        # Descarga los ZIPs de simulaciones turb y transi y los extrae en la raíz
        gd.getdata(flag="sample")
        print("\n>> Descarga e extracción de muestras aerodinámicas de UniFoil completada con éxito.")
    except Exception as e:
        print(f"\n❌ Error al descargar muestras: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
