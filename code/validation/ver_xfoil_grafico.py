import numpy as np
import os
import sys

# Asegurar importación de módulos hermanos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xfoil_wrapper import XFoilWrapper

def ver_simulacion_grafica():
    print("==================================================")
    print("  Visualizador Gráfico de XFOIL en Vivo  ")
    print("==================================================\n")
    print("Este script abrirá la ventana de XFOIL en tu pantalla.")
    print("Presiona ENTER en la ventana de XFOIL para avanzar si se pausa.\n")

    # Generamos un perfil simple NACA 2412 aproximado
    x = np.linspace(0, 1, 100)
    # Ecuación de espesor clásica
    y_t = 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    # Camber medio para NACA 2412
    y_c = np.where(x < 0.4, (0.02 / 0.16) * (0.8 * x - x**2), (0.02 / 0.36) * (0.2 + 0.8 * x - x**2))
    
    y_u = y_c + y_t
    y_l = y_c - y_t

    # Creamos un wrapper temporal que NO tiene desactivados los gráficos
    # Escribimos los comandos directamente para la prueba gráfica
    airfoil_file = os.path.abspath("temp_ver_airfoil.dat")
    polar_file = os.path.abspath("temp_ver_polar.txt")
    input_file = os.path.abspath("temp_ver_input.txt")
    
    # Escribimos el perfil
    xf = XFoilWrapper()
    xf.write_airfoil_file(x, y_u, y_l, airfoil_file)
    
    if os.path.exists(polar_file):
        os.remove(polar_file)
        
    # Comandos con graficación activa (sin plop g)
    commands = [
        f"load {airfoil_file}",
        "oper",
        "visc 150000",
        "iter 100",
        "pacc",
        polar_file,
        "",
        "aseq 0 8 1",
        "pacc",
        "",
        "quit"
    ]
    
    with open(input_file, "w") as f:
        f.write("\n".join(commands) + "\n")
        
    print("Ejecutando XFOIL... Observa la ventana gráfica que se abrirá.")
    
    # Corremos permitiendo entrada interactiva de teclado si fuera necesario
    try:
        import subprocess
        # Nota: aquí no redirigimos stdin para que el proceso sea interactivo si se requiere
        os.system(f'"{xf.xfoil_path}" < "{input_file}"')
        print("\nSimulación finalizada. Polar guardada temporalmente.")
    except Exception as e:
        print(f"Error al correr la simulación: {e}")
        
    # Limpieza
    for f in [airfoil_file, input_file, polar_file]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    ver_simulacion_grafica()
