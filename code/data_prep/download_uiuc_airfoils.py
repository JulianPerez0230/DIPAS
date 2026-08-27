# -*- coding: utf-8 -*-
"""
Descarga en bloque la base de datos completa de perfiles UIUC (~1.650 archivos .dat)
Fuente: https://m-selig.ae.illinois.edu/ads/coord_database.html
Con soporte para reanudación automática y reintentos con retraso en caso de bloqueo.
 
Uso:
    pip install requests beautifulsoup4
    python download_uiuc_airfoils.py
 
Salida:
    ../data/uiuc_airfoils/<nombre>.dat   (uno por perfil, formato Selig original)
    ../data/uiuc_airfoils/_download_log.csv  (registro de éxitos/fallos)
"""
 
import re
import csv
import time
import requests
import os
from pathlib import Path
from bs4 import BeautifulSoup
 
BASE_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
COORD_DIR = "https://m-selig.ae.illinois.edu/ads/coord/"

# Ajustamos la ruta de salida para guardarla de forma ordenada en data/uiuc_airfoils
SCRIPT_DIR = Path(__file__).parent.absolute()
OUT_DIR = SCRIPT_DIR.parent / "data" / "uiuc_airfoils"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
HEADERS = {"User-Agent": "Mozilla/5.0 (research use, DIPAS project)"}
 
def get_dat_filenames():
    """Parsea la página índice y extrae todos los nombres de archivo .dat listados."""
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
 
    names = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"([A-Za-z0-9_\-]+\.dat)$", href)
        if m:
            names.add(m.group(1))
 
    return sorted(names)
 
def download_all(names, delay=0.15):
    log_rows = []
    for i, fname in enumerate(names, 1):
        url = COORD_DIR + fname
        out_path = OUT_DIR / fname
        
        # Soporte para reanudación automática
        if out_path.exists() and out_path.stat().st_size > 0:
            status = "OK"
            log_rows.append({"filename": fname, "status": status})
            print(f"[{i}/{len(names)}] {fname} -> OK (Ya existente)")
            continue
            
        # Lógica de reintentos con esperas en caso de fallos de conexión (bloqueos)
        status = "PENDING"
        for attempt in range(1, 4): # Intentar hasta 3 veces
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 200 and len(r.content) > 0:
                    out_path.write_bytes(r.content)
                    status = "OK"
                    break
                else:
                    status = f"HTTP_{r.status_code}"
            except Exception as e:
                status = f"ERROR_{e.__class__.__name__}"
                
            # Si falló, esperamos un tiempo que va creciendo para dar respiro al servidor
            wait_time = attempt * 5
            print(f"  [ALERTA] Intento {attempt} fallido para {fname} ({status}). Reintentando en {wait_time}s...")
            time.sleep(wait_time)
 
        log_rows.append({"filename": fname, "status": status})
        print(f"[{i}/{len(names)}] {fname} -> {status}")
        time.sleep(delay)  # ser considerado con el servidor de la cátedra UIUC
 
    with open(OUT_DIR / "_download_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "status"])
        writer.writeheader()
        writer.writerows(log_rows)
 
    ok = sum(1 for r in log_rows if r["status"] == "OK")
    print(f"\nCompletado: {ok}/{len(names)} archivos descargados en {OUT_DIR.resolve()}")
 
if __name__ == "__main__":
    print("Obteniendo listado de perfiles desde el índice UIUC...")
    try:
        filenames = get_dat_filenames()
        print(f"Se encontraron {len(filenames)} archivos .dat listados.")
        download_all(filenames)
    except Exception as e:
        print(f"Error al conectar con la base de datos de la UIUC: {e}")
