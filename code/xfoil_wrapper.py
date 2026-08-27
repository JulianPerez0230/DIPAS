import subprocess
import os
import shutil
import numpy as np

class XFoilWrapper:
    def __init__(self, xfoil_path=None):
        """
        Wrapper de Python para automatizar simulaciones 2D en XFOIL.
        Soporta ejecución en Windows (xfoil.exe local) y entornos Linux / Cloud (xfoil_linux o /usr/bin/xfoil).
        """
        resolved_path = None
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_exe = os.path.join(script_dir, "xfoil.exe")
        local_linux = os.path.join(script_dir, "xfoil_linux")

        # 1. En entornos Linux / Cloud (Streamlit Community Cloud)
        if os.name != 'nt':
            if os.path.exists(local_linux):
                resolved_path = local_linux
                try:
                    os.chmod(local_linux, 0o777)
                except Exception:
                    pass
            elif shutil.which("xfoil"):
                resolved_path = shutil.which("xfoil")
            elif os.path.exists("/usr/bin/xfoil"):
                resolved_path = "/usr/bin/xfoil"
            elif os.path.exists("/usr/local/bin/xfoil"):
                resolved_path = "/usr/local/bin/xfoil"

        # 2. En entornos Windows o fallback
        if resolved_path is None:
            if xfoil_path and os.path.exists(xfoil_path) and not (os.name != 'nt' and xfoil_path.endswith('.exe')):
                resolved_path = os.path.abspath(xfoil_path)
            elif os.name == 'nt' and os.path.exists(local_exe):
                resolved_path = local_exe
            elif shutil.which("xfoil"):
                resolved_path = shutil.which("xfoil")
            else:
                resolved_path = local_exe

        self.xfoil_path = resolved_path
        
        # En Linux dar permisos de ejecución si es ruta a archivo
        if os.name != 'nt' and os.path.exists(self.xfoil_path):
            try:
                os.chmod(self.xfoil_path, 0o777)
            except Exception:
                pass

        if not os.path.exists(self.xfoil_path) and not shutil.which(str(self.xfoil_path)):
            raise FileNotFoundError(f"No se encontró el ejecutable de XFOIL en: {self.xfoil_path}")

    def write_airfoil_file(self, x, y_upper, y_lower, file_path="temp_airfoil.dat"):
        """
        Escribe las coordenadas del perfil en el formato estándar de XFOIL (.dat).
        El orden debe ir desde el borde de fuga por el extradós, dar la vuelta en el 
        borde de ataque, y volver por el intradós hasta el borde de fuga.
        """
        # Invertimos el extradós para ir desde x=1 (borde de fuga) hasta x=0 (borde de ataque)
        x_up_inv = np.flip(x)
        y_up_inv = np.flip(y_upper)
        
        # El intradós va desde x=0 (borde de ataque) hasta x=1 (borde de fuga)
        # Omitimos el primer punto (x=0) para no duplicar el borde de ataque
        x_low = x[1:]
        y_low = y_lower[1:]
        
        # Concatenamos las coordenadas en sentido horario
        x_coords = np.concatenate([x_up_inv, x_low])
        y_coords = np.concatenate([y_up_inv, y_low])
        
        # Guardamos en formato de texto plano con espaciado
        with open(file_path, "w") as f:
            f.write("TEMP_AIRFOIL\n")
            for xi, yi in zip(x_coords, y_coords):
                f.write(f" {xi:.6f}   {yi:.6f}\n")
                
        return file_path

    def _get_cmd(self):
        if os.name != 'nt' and shutil.which("xvfb-run"):
            return ["xvfb-run", "-a", "-s", "-screen 0 800x600x16", self.xfoil_path]
        return [self.xfoil_path]

    def run_simulation(self, x, y_upper, y_lower, reynolds, alpha_start, alpha_end, alpha_step, file_prefix="temp"):
        """
        Ejecuta la simulación en XFOIL y extrae la curva polar de coeficientes.
        """
        airfoil_file = self.write_airfoil_file(x, y_upper, y_lower, f"{file_prefix}_airfoil.dat")
        polar_file = f"{file_prefix}_polar.txt"
        input_file = f"{file_prefix}_input.txt"
        
        # Si existe una polar anterior, la borramos para evitar interferencias
        if os.path.exists(polar_file):
            os.remove(polar_file)
            
        # 1. Escribir la lista de comandos que le inyectaremos a XFOIL
        commands = [
            "plop",                  # Entrar al menú de opciones de graficación (Plot Options)
            "g",                     # Desactivar gráficos (toggle graphics)
            "",                      # Salir de plop
            f"load {airfoil_file}",  # Cargar el archivo de coordenadas
            "pane",                  # Repanelar superficie para convergencia óptima
            "oper",                  # Entrar al menú de operaciones
            f"visc {reynolds}",      # Definir flujo viscoso y número de Reynolds
            "iter 80",               # 80 iteraciones máximas por punto
            "pacc",                  # Abrir acumulador de polar
            polar_file,              # Nombre del archivo donde se guardará la polar
            "",                      # Enter para confirmar configuración por defecto de acumulación
            f"aseq {alpha_start} {alpha_end} {alpha_step}", # Barrido de ángulos de ataque
            "pacc",                  # Cerrar acumulador de polar
            "",                      # Enter para salir de oper
            "quit"                   # Salir de XFOIL
        ]
        
        cmd_str = "\n".join(commands) + "\n"
        
        # 2. Ejecutar XFOIL inyectando el flujo de comandos
        try:
            env = os.environ.copy()
            if os.name != 'nt' and "DISPLAY" not in env:
                env["DISPLAY"] = ":99"

            cmd = self._get_cmd()
            process = subprocess.run(
                cmd,
                input=cmd_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=35,
                env=env
            )
            print(f"[XFOIL Diagnostic] Executable: {self.xfoil_path}")
            print(f"[XFOIL Diagnostic] Command: {cmd}")
            print(f"[XFOIL Diagnostic] ReturnCode: {process.returncode}")
            if process.stderr:
                print(f"[XFOIL Diagnostic] STDERR: {process.stderr.strip()}")
            if process.stdout:
                print(f"[XFOIL Diagnostic] STDOUT Sample: {process.stdout[:300].strip()}")
        except subprocess.TimeoutExpired:
            print("Aviso: XFOIL alcanzó tiempo límite, extrayendo puntos calculados...")
        except Exception as e:
            print(f"Aviso: Error ejecutando subproceso XFOIL: {e}")
            
        # 3. Leer y parsear el archivo polar generado
        results = self._parse_polar_file(polar_file)
        print(f"[XFOIL Diagnostic] Polar parsed successfully: {results is not None} (points: {len(results['alpha']) if results else 0})")
        
        # Limpiamos los archivos temporales generados
        self._cleanup_temp_files(airfoil_file, polar_file)
        
        return results

    def _parse_polar_file(self, polar_path):
        """
        Lee el archivo de texto generado por XFOIL y extrae las columnas de datos.
        """
        if not os.path.exists(polar_path):
            # Si el archivo no existe, significa que XFOIL no pudo converger en ningún punto
            return None
            
        data = {
            "alpha": [],
            "CL": [],
            "CD": [],
            "CM": []
        }
        
        with open(polar_path, "r") as f:
            lines = f.readlines()
            
        # XFOIL escribe un encabezado de texto. Los datos numéricos empiezan 
        # después de la línea que contiene los nombres de las columnas (usualmente la línea 12)
        start_reading = False
        for line in lines:
            parts = line.split()
            if not parts:
                continue
                
            # Detectamos el encabezado de las columnas
            if "alpha" in parts and "CL" in parts:
                start_reading = True
                continue
                
            # Si ya pasamos el encabezado, leemos las filas numéricas
            if start_reading:
                # Comprobamos que sea una fila de datos (debe tener al menos 4 números)
                try:
                    # Las columnas estándar de una polar de XFOIL son:
                    # [alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr]
                    alpha_val = float(parts[0])
                    cl_val = float(parts[1])
                    cd_val = float(parts[2])
                    cm_val = float(parts[4])
                    
                    data["alpha"].append(alpha_val)
                    data["CL"].append(cl_val)
                    data["CD"].append(cd_val)
                    data["CM"].append(cm_val)
                except (ValueError, IndexError):
                    # Ignora líneas que no sean numéricas (como guiones separadores)
                    continue
                    
        # Si no se leyó ningún punto válido, retornamos None
        if not data["alpha"]:
            return None
            
        return data

    def get_cp_distribution(self, x, y_upper, y_lower, reynolds, alpha=0.0, file_prefix="temp"):
        """
        Ejecuta XFOIL para un ángulo de ataque específico y extrae la distribución de presiones Cp(x).
        """
        airfoil_file = self.write_airfoil_file(x, y_upper, y_lower, f"{file_prefix}_cp_airfoil.dat")
        cp_file = f"{file_prefix}_cp.txt"
        input_file = f"{file_prefix}_cp_input.txt"
        
        if os.path.exists(cp_file):
            os.remove(cp_file)
            
        commands = [
            "plop",
            "g",
            "",
            f"load {airfoil_file}",
            "oper",
            f"visc {reynolds}",
            "iter 100",
            f"alfa {alpha}",
            f"cpwr {cp_file}",
            "",
            "quit"
        ]
        
        cmd_str = "\n".join(commands) + "\n"
        try:
            env = os.environ.copy()
            if os.name != 'nt' and "DISPLAY" not in env:
                env["DISPLAY"] = ":99"

            subprocess.run(
                self._get_cmd(),
                input=cmd_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                env=env
            )
        except Exception:
            self._cleanup_temp_files(airfoil_file, cp_file)
            return None
            
        cp_data = self._parse_cp_file(cp_file)
        self._cleanup_temp_files(airfoil_file, cp_file)
        return cp_data

    def _parse_cp_file(self, cp_path):
        if not os.path.exists(cp_path):
            return None
        cp_res = {"x": [], "y": [], "Cp": []}
        with open(cp_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                        cp_val = float(parts[2])
                        cp_res["x"].append(x_val)
                        cp_res["y"].append(y_val)
                        cp_res["Cp"].append(cp_val)
                    except ValueError:
                        continue
        if not cp_res["x"]:
            return None
        return cp_res

    def _cleanup_temp_files(self, *files):
        """
        Elimina los archivos temporales de entrada y salida de XFOIL.
        """
        for file in files:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    # Prueba rápida del wrapper usando coordenadas del Selig S3021
    print("Probando conexión con XFOIL...")
    
    # Generamos un perfil simétrico simple para la prueba rápida
    x = np.linspace(0, 1, 100)
    y_u = 0.06 * (np.sqrt(x) - x) # Perfil delgado de prueba
    y_l = -y_u
    
    try:
        xf = XFoilWrapper(xfoil_path="xfoil.exe")
        res = xf.run_simulation(x, y_u, y_l, reynolds=150000, alpha_start=0, alpha_end=5, alpha_step=1)
        if res:
            print("\nSimulación exitosa. Resultados obtenidos:")
            for a, cl, cd, cm in zip(res["alpha"], res["CL"], res["CD"], res["CM"]):
                print(f"Alpha: {a:5.1f} | CL: {cl:6.3f} | CD: {cd:6.4f} | CM: {cm:7.4f}")
        else:
            print("\nError: XFOIL corrió pero no pudo converger en ningún punto.")
    except Exception as e:
        print(f"\nError al ejecutar la prueba: {e}")
