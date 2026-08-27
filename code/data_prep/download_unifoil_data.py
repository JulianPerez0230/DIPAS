# -*- coding: utf-8 -*-
"""
Script para descargar la metadata y las geometrías obligatorias del dataset UniFoil.
Usa la interfaz oficial de unifoil instalada localmente.
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
        import unifoil
    except ImportError as e:
        print(f"❌ Error al importar unifoil. ¿Está instalado? {e}")
        sys.exit(1)
        
    print(">> Iniciando la descarga del dataset UniFoil (flag='compulsory')...")
    print("   Esto descargará las tablas de metadata (CSV) y los zips de geometría en la raíz del proyecto.")
    
    # Instanciamos GetData apuntando a la raíz del proyecto (DIPAS/)
    data_root = SCRIPT_DIR.parent
    gd = GetData(data_root=str(data_root))
    
    # Descargar archivos obligatorios (Metadata + Geometrías bases)
    try:
        gd.getdata(flag="compulsory")
        print("\n>> Descarga de archivos obligatorios completada con éxito.")
    except Exception as e:
        print(f"\n❌ Error durante la descarga de getdata: {e}")
        sys.exit(1)
        
    # Generar las carpetas de geometría descomprimidas (.dat)
    print("\n>> Generando carpetas de geometría (.dat) en la raíz...")
    # Cambiamos temporalmente al directorio raíz para que gen_ft y gen_nlf creen las carpetas allí
    os.chdir(str(data_root))
    try:
        unifoil.gen_ft()
        unifoil.gen_nlf()
        print("\n>> ¡Geometrías de UniFoil generadas con éxito!")
    except Exception as e:
        print(f"\n❌ Error al generar las geometrías de UniFoil: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
