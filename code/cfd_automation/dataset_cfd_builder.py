# -*- coding: utf-8 -*-
# Definitive CFD Dataset Builder orchestrating SpaceClaim, ANSYS Meshing, and Fluent batch study
# Expanded to support multiple Reynolds numbers (100k, 150k, 200k) with checkpointing per (profile_id, reynolds)
import os
import subprocess
import sys
import csv
import math
import time
import numpy as np

# Asegurar importación de módulos hermanos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cst_generator import CSTParametrization
from fluent_mesh_generator import generate_airfoil_mesh

# Lista de perfiles semilla locales
SEEDS = ["s3021", "e387", "sd7037", "clarky", "naca2412", "naca0012", "naca4412", "mh32", "fx60126", "sd7062"]

class CFDDatasetBuilder:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.dirname(self.script_dir)
        self.data_dir = os.path.join(self.project_dir, "data")
        self.seeds_dir = os.path.join(self.data_dir, "seeds")
        
        self.output_csv = os.path.join(self.data_dir, "dataset_cfd.csv")
        self.coords_path = os.path.join(self.data_dir, "s3021_discovery.txt")
        self.mesh_path = os.path.join(self.data_dir, "current_airfoil.msh")
        self.wb_log_path = os.path.join(self.data_dir, "workbench_script.log")
        
        self.fluent_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v261\fluent\ntbin\win64\fluent.exe"
        self.workbench_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v261\Framework\bin\Win64\runwb2.exe"
        
        # 12 coeficientes en total (6 upper, 6 lower) con espesor de borde de fuga de 0.003
        self.cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
        self.cuerda = 200.0 # 200 mm
        
        # 6 ángulos de ataque en régimen confiable (flujo adherido / pre-stall)
        self.alphas = [-2, 0, 2, 4, 6, 8]

        # Escalera de 6 números de Reynolds
        self.reynolds_list = [100000, 150000, 200000, 250000, 300000, 350000]

    def parse_uiuc_file(self, file_path, n_points=80):
        """Lee un archivo .dat de UIUC y devuelve x, y_upper, y_lower en grilla de coseno"""
        x_raw, y_raw = [], []
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 2:
                try:
                    val_x = float(parts[0])
                    val_y = float(parts[1])
                    if val_x <= 1.0:
                        x_raw.append(val_x)
                        y_raw.append(val_y)
                except ValueError:
                    continue
        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)
        
        if x_raw[0] < 0.1:
            split_idx = len(x_raw) // 2
            for i in range(1, len(x_raw)):
                if x_raw[i] < x_raw[i-1]:
                    split_idx = i
                    break
            x_upper = x_raw[:split_idx]
            y_upper = y_raw[:split_idx]
            x_lower = x_raw[split_idx:]
            y_lower = y_raw[split_idx:]
            theta = np.linspace(0, np.pi, n_points)
            x_grid = 0.5 * (1.0 - np.cos(theta))
            y_upper_grid = np.interp(x_grid, x_upper, y_upper)
            y_lower_grid = np.interp(x_grid, x_lower, y_lower)
        else:
            idx_min_x = np.argmin(x_raw)
            x_upper = x_raw[:idx_min_x+1]
            y_upper = y_raw[:idx_min_x+1]
            x_lower = x_raw[idx_min_x:]
            y_lower = y_raw[idx_min_x:]
            theta = np.linspace(0, np.pi, n_points)
            x_grid = 0.5 * (1.0 - np.cos(theta))
            y_upper_grid = np.interp(x_grid, np.flip(x_upper), np.flip(y_upper))
            y_lower_grid = np.interp(x_grid, x_lower, y_lower)
        return x_grid, y_upper_grid, y_lower_grid

    def perturb_coefficients(self, coefs, max_delta=0.015):
        """Perturba aleatoriamente los coeficientes CST"""
        deltas = np.random.uniform(-max_delta, max_delta, size=len(coefs))
        return coefs + deltas

    def write_discovery_coords(self, x, y_up, y_low):
        """Escribe las coordenadas en formato SpaceClaim (XY-plane, 200mm)"""
        # Borde de fuga manufacturable (espesor 0.6 mm en cuerda de 200 mm)
        te_half_thickness = 0.0015
        y_up_thick = y_up + (x * te_half_thickness)
        y_low_thick = y_low - (x * te_half_thickness)
        
        x_up_inv = np.flip(x)
        y_up_inv = np.flip(y_up_thick)
        x_low_rem = x[1:]
        y_low_rem = y_low_thick[1:]
        
        with open(self.coords_path, "w") as f:
            f.write("3d=true\n")
            f.write("polyline=false\n")
            f.write("fit=false\n\n")
            
            # Extradós
            for xi, yi in zip(x_up_inv, y_up_inv):
                f.write(f"0.000000   {xi*self.cuerda:.6f}   {yi*self.cuerda:.6f}\n")
            f.write("\n")
            
            # Intradós
            f.write(f"0.000000   {x_up_inv[-1]*self.cuerda:.6f}   {y_up_inv[-1]*self.cuerda:.6f}\n")
            for xi, yi in zip(x_low_rem, y_low_rem):
                f.write(f"0.000000   {xi*self.cuerda:.6f}   {yi*self.cuerda:.6f}\n")

    def build_dataset(self, total_variants=300, max_delta=0.015):
        # 1. Definimos las cabeceras del CSV
        headers = [
            "profile_id", "seed", "reynolds", "alpha", "cl", "cd",
            "au0", "au1", "au2", "au3", "au4", "au5",
            "al0", "al1", "al2", "al3", "al4", "al5"
        ]
        
        # Lógica de reanudación automática por tupla (profile_id, reynolds)
        simulated_runs = set()
        file_exists = os.path.exists(self.output_csv)
        if file_exists:
            print(">> Detectado archivo dataset_cfd.csv previo. Cargando progreso...")
            try:
                # Contamos cuántos perfiles únicos hay en total en el archivo para la reanudación
                profile_re_pairs = {}
                with open(self.output_csv, "r") as csv_f:
                    reader = csv.reader(csv_f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 4:
                            p_id = int(row[0])
                            re = int(row[2])
                            # Agrupamos por par y contamos cuántos ángulos de ataque se registraron
                            key = (p_id, re)
                            profile_re_pairs[key] = profile_re_pairs.get(key, 0) + 1
                            
                # Solo marcamos como simulado al 100% si tiene las 8 polares completas
                for (p_id, re), count in profile_re_pairs.items():
                    if count >= len(self.alphas):
                        simulated_runs.add((p_id, re))
                        
                print(f"  Encontrados {len(simulated_runs)} casos de (perfil, reynolds) ya resueltos.")
            except Exception as e:
                print(f"  Advertencia al leer progreso ({e}). Empezando de cero.")
                file_exists = False

        if not file_exists:
            with open(self.output_csv, "w", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(headers)
                
        # 2. Generación determinista de los variantes usando semillas
        variants_per_seed = int(math.ceil(total_variants / len(SEEDS)))
        all_variants = []
        variant_counter = 0
        
        np.random.seed(42) # Fijamos semilla aleatoria para repetibilidad
        
        for seed in SEEDS:
            seed_file = os.path.join(self.seeds_dir, f"{seed}.dat")
            if not os.path.exists(seed_file):
                print(f"Advertencia: No existe el archivo de semilla {seed_file}")
                continue
                
            x_grid, y_u_base, y_l_base = self.parse_uiuc_file(seed_file)
            coords = self.cst.fit_airfoil(x_grid, y_u_base, y_l_base)
            if coords is None:
                continue
            coefs_u_base, coefs_l_base = coords
            
            for v in range(variants_per_seed):
                variant_counter += 1
                if variant_counter > total_variants:
                    break
                    
                coefs_u_pert = self.perturb_coefficients(coefs_u_base, max_delta)
                coefs_l_pert = self.perturb_coefficients(coefs_l_base, max_delta)
                all_variants.append({
                    "id": variant_counter,
                    "seed": seed,
                    "coefs_u": coefs_u_pert,
                    "coefs_l": coefs_l_pert
                })
        
        print(f"\n>> Total de variantes a simular: {len(all_variants)}")
        
        # 3. Bucle principal de simulación
        for var in all_variants:
            p_id = var["id"]
            
            # Verificamos si nos falta calcular ALGUNO de los Reynolds para este perfil
            re_to_simulate = [re for re in self.reynolds_list if (p_id, re) not in simulated_runs]
            if not re_to_simulate:
                continue # Ya tiene todos los Reynolds simulados
                
            print(f"\n==================================================")
            print(f" Simulando Perfil {p_id}/{total_variants} (Semilla: {var['seed']})")
            print(f"==================================================")
            
            # A. Generamos coordenadas y escribimos s3021_discovery.txt
            x_grid, y_u, y_l = self.cst.generate_coordinates(var["coefs_u"], var["coefs_l"])
            self.write_discovery_coords(x_grid, y_u, y_l)
            
            # Construir lista de puntos ordenados (TE superior -> LE -> TE inferior) en metros
            # x_grid esta en rango [0, 1] -> multiplicar por cuerda (0.200 m)
            c_m = self.cuerda / 1000.0 # 0.200 m
            pts_upper = list(zip(x_grid[::-1] * c_m, y_u[::-1] * c_m))
            pts_lower = list(zip(x_grid[1:] * c_m, y_l[1:] * c_m))
            coords_airfoil_m = pts_upper + pts_lower
            
            # B. Generamos la malla 2D directamente con Gmsh en ~0.5 segundos
            print(f">> Generando malla 2D aerodinamica para perfil {p_id}...")
            t_mesh_0 = time.time()
            try:
                generate_airfoil_mesh(coords_airfoil_m, self.mesh_path, chord=c_m)
                t_mesh_elapsed = time.time() - t_mesh_0
                print(f">> [OK] Malla generada con exito en {t_mesh_elapsed:.2f}s: {self.mesh_path}")
            except Exception as e_mesh:
                print(f"[ERROR] Fallo la generacion de malla para perfil {p_id}: {e_mesh}")
                continue
            
            # C. Bucle por cada Reynolds faltante para este perfil
            for re_val in re_to_simulate:
                # Calculamos velocidad del inlet para este Reynolds basado en cuerda = 0.2m
                # Re = rho * V * c / mu  => V = Re * mu / (rho * c)
                # Con rho = 1.225, mu = 1.7894e-5, c = 0.20 => V = Re * 7.3036e-5
                vel_mag = re_val * (1.7894e-5 / (1.225 * 0.20))
                
                print(f"\n   -> Corriendo barrido para Reynolds: {re_val} (V = {vel_mag:.2f} m/s)")
                
                # Escribimos el Journal de Fluent para este Reynolds
                journal_path = os.path.join(self.data_dir, "run_polar.jou")
                mesh_path_fluent = self.mesh_path.replace("\\", "/")
                polar_stream_out = os.path.join(self.data_dir, "polar_stream.out").replace("\\", "/")
                
                # Eliminamos archivo stream previo si existe
                if os.path.exists(polar_stream_out.replace("/", "\\")):
                    try:
                        os.remove(polar_stream_out.replace("/", "\\"))
                    except:
                        pass
                
                # Iteraciones: 150 para el primer angulo (arranque frio), 100 para los demas (warm-start continuo)
                ITERS_FIRST = 160
                ITERS_STEP  = 100
                
                # Encabezado del Journal
                journal_lines = [
                    '/file/confirm-overwrite yes',
                    f'/file/read-case "{mesh_path_fluent}"',
                    '/define/models/viscous/transition-sst yes',
                    '/report/reference-values/area 0.2',
                    '/report/reference-values/length 0.2',
                    f'/report/reference-values/velocity {vel_mag:.6f}',
                    '/solve/report-definitions/add cd-report drag force-vector 1 0 thread-names airfoil () quit',
                    '/solve/report-definitions/add cl-report lift force-vector 0 1 thread-names airfoil () quit',
                    '/solve/report-files/add polar-file report-defs cd-report cl-report () file-name "polar_stream.out" ()',
                    '/solve/initialize/hyb-initialization'
                ]
                
                # Puntos acumulados de iteracion para extraer resultados
                iter_checkpoints = []
                current_iter = 0
                
                # Barrido continuo por cada ángulo de ataque (warm-start entre ángulos)
                for idx, alpha in enumerate(self.alphas):
                    alpha_rad = math.radians(alpha)
                    dir_x = math.cos(alpha_rad)
                    dir_y = math.sin(alpha_rad)
                    
                    n_iters = ITERS_FIRST if idx == 0 else ITERS_STEP
                    current_iter += n_iters
                    iter_checkpoints.append((idx, alpha, current_iter))
                    
                    # 17 prompts para actualizar la velocidad y dirección del inlet
                    journal_lines.extend([
                        '/define/boundary-conditions/velocity-inlet inlet',
                        'yes',
                        'yes',
                        'no',
                        f'{vel_mag:.6f}',
                        'no',
                        '0',
                        'no',
                        f'{dir_x:.6f}',
                        'no',
                        f'{dir_y:.6f}',
                        'no',
                        'no',
                        'yes',
                        'no',
                        '1',
                        '0.1',
                        '10',
                        f'/solve/iterate {n_iters}'
                    ])
                
                journal_lines.append('/exit yes')
                
                with open(journal_path, "w") as j_f:
                    j_f.write("\n".join(journal_lines) + "\n")
                
                # Lanzamos Fluent
                time.sleep(4.0) # Espera para liberacion de licencia ANSYS Student
                fluent_log = os.path.join(self.data_dir, "fluent_execution.log")
                fluent_command = [
                    self.fluent_path,
                    "2ddp",
                    "-g",
                    "-r26.1.0",
                    "-shortcut",
                    f"-i{journal_path}"
                ]
                
                with open(fluent_log, "w") as log_f:
                    subprocess.run(fluent_command, cwd=self.data_dir, stdout=log_f, stderr=log_f, text=True, timeout=2400)
                    
                time.sleep(3.0)
                
                # Procesamos resultados leyendo polar_stream.out
                results_polar = []
                all_angles_success = True
                polar_stream_disk = os.path.join(self.data_dir, "polar_stream.out")
                
                # Esperar a que Fluent libere y sincronice polar_stream.out en disco
                found_stream = False
                for _ in range(10):
                    if os.path.exists(polar_stream_disk) and os.path.getsize(polar_stream_disk) > 100:
                        found_stream = True
                        break
                    time.sleep(1.0)
                
                if found_stream:
                    try:
                        # Leer todas las lineas de datos
                        with open(polar_stream_disk, "r") as f_st:
                            stream_lines = [l.strip() for l in f_st if l.strip() and not l.startswith('"') and not l.startswith('(')]
                        
                        # Mapear iteracion -> (cd, cl)
                        iter_data = {}
                        for s_line in stream_lines:
                            parts = s_line.split()
                            if len(parts) >= 3:
                                try:
                                    it_num = int(parts[0])
                                    cd_v = float(parts[1])
                                    cl_v = float(parts[2])
                                    iter_data[it_num] = (cd_v, cl_v)
                                except:
                                    pass
                                    
                        # Extraer los datos para cada angulo segun los checkpoints
                        for idx, alpha, target_iter in iter_checkpoints:
                            alpha_rad = math.radians(alpha)
                            
                            # Buscar la iteracion mas cercana disponible a target_iter
                            available_iters = [it for it in iter_data.keys() if it <= target_iter]
                            if available_iters:
                                best_it = max(available_iters)
                                cx_val, cy_val = iter_data[best_it]
                                
                                cd_val = cx_val * math.cos(alpha_rad) + cy_val * math.sin(alpha_rad)
                                cl_val = -cx_val * math.sin(alpha_rad) + cy_val * math.cos(alpha_rad)
                                
                                results_polar.append({
                                    "alpha": alpha,
                                    "cl": cl_val,
                                    "cd": cd_val
                                })
                            else:
                                print(f"     [ERROR] No hay datos de iteracion para angulo {alpha} (target iter {target_iter})")
                                all_angles_success = False
                    except Exception as e_parse:
                        print(f"     [ERROR] Al parsear polar_stream.out: {e_parse}")
                        all_angles_success = False
                else:
                    print("     [ERROR] No se genero el archivo polar_stream.out")
                    all_angles_success = False
                
                # Guardamos en CSV
                if all_angles_success:
                    with open(self.output_csv, "a", newline="") as csv_f:
                        writer = csv.writer(csv_f)
                        for res in results_polar:
                            row = [
                                p_id,
                                var["seed"],
                                re_val,
                                res["alpha"],
                                f"{res['cl']:.6f}",
                                f"{res['cd']:.6f}",
                                *var["coefs_u"],
                                *var["coefs_l"]
                            ]
                            writer.writerow(row)
                    print(f"     [OK] Reynolds {re_val} simulado con exito.")
                else:
                    print(f"     [ERROR] Reynolds {re_val} fallo en Fluent.")

        print("\n==================================================")
        print(" ¡Proceso del Dataset Finalizado!")
        print(f" Datos guardados en: {self.output_csv}")
        print("==================================================")

if __name__ == "__main__":
    builder = CFDDatasetBuilder()
    builder.build_dataset(total_variants=300)
