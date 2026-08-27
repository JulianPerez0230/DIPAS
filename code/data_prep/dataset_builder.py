import os
import urllib.request
import numpy as np
import csv
import sys

# Asegurar importación de módulos hermanos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cst_generator import CSTParametrization
from xfoil_wrapper import XFoilWrapper

# URLs oficiales de la base de datos de coordenadas UIUC
SEED_URLS = {
    "s3021": "https://m-selig.ae.illinois.edu/ads/coord/s3021.dat",
    "e387": "https://m-selig.ae.illinois.edu/ads/coord/e387.dat",
    "sd7037": "https://m-selig.ae.illinois.edu/ads/coord/sd7037.dat",
    "clarky": "https://m-selig.ae.illinois.edu/ads/coord/clarky.dat",
    "naca2412": "https://m-selig.ae.illinois.edu/ads/coord/naca2412.dat",
    "naca0012": "https://m-selig.ae.illinois.edu/ads/coord/n0012.dat",
    "naca4412": "https://m-selig.ae.illinois.edu/ads/coord/naca4412.dat",
    "mh32": "https://m-selig.ae.illinois.edu/ads/coord/mh32.dat",
    "fx60126": "https://m-selig.ae.illinois.edu/ads/coord/fx60126.dat",
    "sd7062": "https://m-selig.ae.illinois.edu/ads/coord/sd7062.dat"
}

class DatasetBuilder:
    def __init__(self, seeds_dir="../data/seeds", output_file="../data/dataset.csv", xfoil_path="xfoil.exe"):
        """
        Orquestador para descargar perfiles semilla, perturbar sus coeficientes CST y simular con XFOIL.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.seeds_dir = os.path.abspath(os.path.join(script_dir, seeds_dir))
        self.output_file = os.path.abspath(os.path.join(script_dir, output_file))
        
        # Inicializamos herramientas
        self.cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
        self.xfoil = XFoilWrapper(xfoil_path=xfoil_path)
        
        # Aseguramos existencia de directorios
        os.makedirs(self.seeds_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def download_seeds(self):
        """
        Descarga los perfiles semilla desde la UIUC si no están descargados localmente.
        """
        print(">> Verificando perfiles semilla...")
        for name, url in SEED_URLS.items():
            dest_path = os.path.join(self.seeds_dir, f"{name}.dat")
            if not os.path.exists(dest_path):
                print(f"Descargando {name}.dat desde la UIUC...")
                try:
                    urllib.request.urlretrieve(url, dest_path)
                except Exception as e:
                    print(f"Error al descargar {name}: {e}")
            else:
                print(f"  Semilla '{name}' ya existe localmente.")

    def parse_uiuc_file(self, file_path, n_points=100):
        """
        Lee un archivo .dat en formato UIUC, separa extradós/intradós e interpola sobre malla estándar.
        """
        x_raw, y_raw = [], []
        with open(file_path, "r") as f:
            lines = f.readlines()
            
        # Omitimos la primera línea (cabecera con el nombre del perfil)
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 2:
                try:
                    val_x = float(parts[0])
                    val_y = float(parts[1])
                    # Las coordenadas normalizadas de un perfil siempre son <= 1.0.
                    # Esto descarta cabeceras de cantidad de puntos (ej: '61.0 61.0')
                    if val_x <= 1.0:
                        x_raw.append(val_x)
                        y_raw.append(val_y)
                except ValueError:
                    continue
                    
        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)
        
        # Detección del formato de archivo
        if x_raw[0] < 0.1:
            # Formato Lednicer (dos bloques: 0 -> 1 y 0 -> 1)
            # Buscamos el punto de corte donde x disminuye drásticamente (de 1.0 a 0.0)
            split_idx = len(x_raw) // 2 # Valor por defecto en caso de fallo
            for i in range(1, len(x_raw)):
                if x_raw[i] < x_raw[i-1]:
                    split_idx = i
                    break
                    
            x_upper = x_raw[:split_idx]
            y_upper = y_raw[:split_idx]
            x_lower = x_raw[split_idx:]
            y_lower = y_raw[split_idx:]
            
            # Crear la grilla de x normalizada (distribución de coseno)
            theta = np.linspace(0, np.pi, n_points)
            x_grid = 0.5 * (1.0 - np.cos(theta))
            
            # En Lednicer, ambas curvas ya están ordenadas de 0 a 1 (no hace falta hacer flip)
            y_upper_grid = np.interp(x_grid, x_upper, y_upper)
            y_lower_grid = np.interp(x_grid, x_lower, y_lower)
        else:
            # Formato Estándar UIUC (un solo bucle: 1 -> 0 -> 1)
            idx_min_x = np.argmin(x_raw)
            
            x_upper = x_raw[:idx_min_x+1]
            y_upper = y_raw[:idx_min_x+1]
            x_lower = x_raw[idx_min_x:]
            y_lower = y_raw[idx_min_x:]
            
            # Crear la grilla de x normalizada (distribución de coseno)
            theta = np.linspace(0, np.pi, n_points)
            x_grid = 0.5 * (1.0 - np.cos(theta))
            
            # En UIUC, el extradós va de 1 a 0 (debe invertirse para ordenar de 0 a 1)
            y_upper_grid = np.interp(x_grid, np.flip(x_upper), np.flip(y_upper))
            y_lower_grid = np.interp(x_grid, x_lower, y_lower)
        
        return x_grid, y_upper_grid, y_lower_grid

    def perturb_coefficients(self, coefs, max_delta=0.015):
        """
        Aplica una perturbación aleatoria uniforme a los coeficientes CST de la semilla.
        """
        deltas = np.random.uniform(-max_delta, max_delta, size=len(coefs))
        return coefs + deltas

    def build_dataset(self, n_variants_per_seed=5, max_delta=0.015):
        """
        Ejecuta el pipeline completo de generación de perfiles y simulación aerodinámica.
        Soporta reanudación automática si el archivo dataset.csv ya contiene datos.
        """
        self.download_seeds()
        
        # Definimos las cabeceras de nuestro archivo CSV
        headers = [
            "id", "semilla", "reynolds", "alpha", "cl", "cd", "cm",
            "au0", "au1", "au2", "au3", "au4", "au5",
            "al0", "al1", "al2", "al3", "al4", "al5"
        ]
        
        # Lógica de reanudación: leer ejecuciones previas para no repetirlas
        existing_runs = set()
        file_exists = os.path.exists(self.output_file)
        
        if file_exists:
            print(">> Detectado archivo dataset.csv previo. Analizando progreso para reanudar...")
            try:
                with open(self.output_file, "r") as csv_f:
                    reader = csv.reader(csv_f)
                    next(reader, None) # Saltar cabecera
                    for row in reader:
                        if len(row) >= 2:
                            seed_name = row[1]
                            try:
                                # El ID tiene el formato p_counter_v
                                v_idx = int(row[0].split("_")[-1])
                                existing_runs.add((seed_name, v_idx))
                            except (ValueError, IndexError):
                                continue
                print(f"  Encontradas {len(existing_runs)} variantes ya simuladas en el archivo.")
            except Exception as e:
                print(f"  Advertencia: No se pudo analizar el progreso previo ({e}). Se empezará desde cero.")
                file_exists = False
        
        # Si el archivo no existe, escribimos las cabeceras desde cero
        if not file_exists:
            with open(self.output_file, "w", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(headers)
            
        print("\n>> Iniciando generación del dataset...")
        profile_counter = 0
        success_counter = len(existing_runs)
        
        # Recorremos cada perfil semilla disponible
        for name in SEED_URLS.keys():
            seed_file = os.path.join(self.seeds_dir, f"{name}.dat")
            if not os.path.exists(seed_file):
                continue
                
            print(f"\nProcesando perfiles basados en semilla '{name}'...")
            
            # 1. Leer y alinear coordenadas
            x_grid, y_u_base, y_l_base = self.parse_uiuc_file(seed_file)
            
            # 2. Extraer coeficientes CST base de la semilla
            coefs_u_base, coefs_l_base = self.cst.fit_airfoil(x_grid, y_u_base, y_l_base)
            
            # 3. Generar variantes
            for v in range(n_variants_per_seed):
                profile_counter += 1
                
                # Si esta variante ya existe en el archivo anterior, la salteamos
                if (name, v) in existing_runs:
                    continue
                
                # Generamos coeficientes perturbados
                coefs_u_pert = self.perturb_coefficients(coefs_u_base, max_delta)
                coefs_l_pert = self.perturb_coefficients(coefs_l_base, max_delta)
                
                # Reconstruimos coordenadas del perfil hijo
                x_pert, y_u_pert, y_l_pert = self.cst.generate_coordinates(coefs_u_pert, coefs_l_pert)
                
                # Seleccionamos número de Reynolds y rango de simulación
                # Para la IA queremos variedad: barremos de forma aleatoria entre 100k y 300k
                re = int(np.random.choice([100000, 150000, 200000, 250000, 300000]))
                
                # Corremos simulación en XFOIL para este perfil
                # Definimos barrido acotado típico de operación: de -4 a 10 grados cada 1 grado
                polar = self.xfoil.run_simulation(
                    x_pert, y_u_pert, y_l_pert, 
                    reynolds=re, 
                    alpha_start=-4, alpha_end=10, alpha_step=1,
                    file_prefix=f"temp_build_{name}_{v}"
                )
                
                if polar:
                    # Guardamos todas las polares convergidas con lógica de reintento en caso de bloqueo (Excel)
                    write_success = False
                    for attempt in range(6): # Intentar hasta 6 veces (30 segundos total)
                        try:
                            with open(self.output_file, "a", newline="") as csv_f:
                                writer = csv.writer(csv_f)
                                for i in range(len(polar["alpha"])):
                                    row = [
                                        f"p_{profile_counter}_{v}",
                                        name,
                                        re,
                                        polar["alpha"][i],
                                        polar["CL"][i],
                                        polar["CD"][i],
                                        polar["CM"][i],
                                        *coefs_u_pert,
                                        *coefs_l_pert
                                    ]
                                    writer.writerow(row)
                            write_success = True
                            break
                        except PermissionError:
                            print(f"\n[ALERTA] Archivo '{self.output_file}' bloqueado (¿abierto en Excel?).")
                            print("Cierra el archivo inmediatamente. Reintentando escritura en 5 segundos...")
                            import time
                            time.sleep(5)
                            
                    if write_success:
                        success_counter += 1
                        print(f"  [OK] Variante {v+1}/{n_variants_per_seed} simulada en Re={re} ({len(polar['alpha'])} ptos).")
                    else:
                        print(f"  [ERROR] Se perdieron los datos del perfil {profile_counter} debido al bloqueo persistente de {self.output_file}.")
                else:
                    print(f"  [ERROR] Variante {v+1}/{n_variants_per_seed} no convergió en XFOIL.")
                    
        print(f"\n==================================================")
        print(f" Generación finalizada exitosamente.")
        print(f" Perfiles intentados: {profile_counter}")
        print(f" Perfiles convergidos en XFOIL: {success_counter}")
        print(f" Datos guardados en: {self.output_file}")
        print(f"==================================================")

if __name__ == "__main__":
    # Generación definitiva del dataset: 1000 variantes por perfil semilla (10 semillas total)
    builder = DatasetBuilder()
    builder.build_dataset(n_variants_per_seed=1000)
