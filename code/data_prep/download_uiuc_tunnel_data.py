# -*- coding: utf-8 -*-
"""
Descarga y extrae en bloque los datasets experimentales de tunel de viento UIUC en formato ZIP.
Incluye los Volúmenes 1, 2, 3, 6 (Williamson) y SoarTech 8 (Princeton).
Con soporte para saltear volúmenes cuya carpeta de extracción ya exista.

Salida:
    ../data/uiuc_tunnel_data/
"""

import os
import zipfile
import requests
from pathlib import Path

# URLs de los archivos ZIP de datos experimentales
ZIP_URLS = {
    "volume01": "https://m-selig.ae.illinois.edu/pd/pub/lsat/volume01.zip",
    "volume02": "https://m-selig.ae.illinois.edu/pd/pub/lsat/volume02.zip",
    "volume03": "https://m-selig.ae.illinois.edu/pd/pub/lsat/volume03.zip",
    "volume06": "https://m-selig.ae.illinois.edu/pd/pub/lsat/volume06.zip",
    "soartech8": "https://m-selig.ae.illinois.edu/uiuc_lsat/Stec8.zip"
}

SCRIPT_DIR = Path(__file__).parent.absolute()
OUT_DIR = SCRIPT_DIR.parent / "data" / "uiuc_tunnel_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research use, DIPAS project)"}

def download_and_extract_zips():
    print(">> Iniciando descarga en bloque de archivos ZIP experimentales...")
    
    for name, url in ZIP_URLS.items():
        zip_path = OUT_DIR / f"{name}.zip"
        extract_dir = OUT_DIR / name
        
        # Soporte para saltear descargas si la carpeta final ya existe
        if extract_dir.exists():
            print(f"  [OK] El volumen '{name}' ya esta descargado y extraido (carpeta existente).")
            continue
            
        # 1. Descarga del ZIP
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            print(f"Descargando {name}.zip desde {url}...")
            try:
                r = requests.get(url, headers=HEADERS, timeout=60)
                if r.status_code == 200:
                    zip_path.write_bytes(r.content)
                    print(f"  [OK] Guardado: {name}.zip ({len(r.content) / 1024 / 1024:.2f} MB)")
                else:
                    print(f"  [FALLO] HTTP {r.status_code} al descargar {name}.zip")
                    continue
            except Exception as e:
                print(f"  [ERROR] Al descargar {name}.zip: {e}")
                continue
        else:
            print(f"  [OK] {name}.zip ya existe localmente.")
            
        # 2. Extracción del ZIP
        print(f"Extrayendo {name}.zip...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"  [OK] Extraido con éxito en {extract_dir.name}/")
            
            # Limpiamos el archivo ZIP temporal para liberar espacio en disco
            os.remove(zip_path)
        except Exception as e:
            print(f"  [ERROR] Al extraer {name}.zip: {e}")

    print(f"\n>> Todos los datasets de tunel de viento (Vols 1, 2, 3, 6 y SoarTech 8) han sido extraidos con éxito en: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    download_and_extract_zips()
