import numpy as np
import sys
import os

# Aseguramos que la carpeta code esté en el path de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cst_generator import CSTParametrization
except ImportError:
    print("Error: No se pudo importar cst_generator.py. Asegúrate de que esté en la misma carpeta.")
    sys.exit(1)

def run_verification():
    print("==================================================")
    print("  Verificación del Ajuste por Mínimos Cuadrados  ")
    print("==================================================\n")

    # Coordenadas reales aproximadas del perfil Selig S3021 (Bajo Reynolds)
    # Formato: 20 puntos para el extradós (upper) y 20 para el intradós (lower)
    x_real = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.20, 0.30, 0.40, 
                       0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 1.0])
    
    y_real_upper = np.array([0.0, 0.0135, 0.0195, 0.0278, 0.0385, 0.0518, 0.0605, 0.0712, 0.0768, 0.0772, 
                             0.0735, 0.0658, 0.0545, 0.0388, 0.0298, 0.0195, 0.0088, 0.0035, 0.0015])
    
    y_real_lower = np.array([0.0, -0.0082, -0.0115, -0.0152, -0.0198, -0.0238, -0.0252, -0.0248, -0.0210, -0.0155, 
                             -0.0092, -0.0035, 0.0015, 0.0048, 0.0042, 0.0028, 0.0010, 0.0002, -0.0015])

    # Inicializamos el parametrizador CST con 6 coeficientes por superficie
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)

    # 1. Ajustar el perfil real para obtener los coeficientes
    print("1. Ejecutando ajuste por Mínimos Cuadrados...")
    coefs_upper, coefs_lower = cst.fit_airfoil(x_real, y_real_upper, y_real_lower)
    
    print("\nCoeficientes obtenidos para el Extradós (Upper):")
    print(np.round(coefs_upper, 5))
    print("\nCoeficientes obtenidos para el Intradós (Lower):")
    print(np.round(coefs_lower, 5))

    # 2. Reconstruir las coordenadas con los coeficientes obtenidos
    print("\n2. Reconstruyendo geometría a partir de los coeficientes...")
    # Generamos en los mismos puntos x para poder calcular el error directamente
    c_x = cst.class_function(x_real)
    y_reconstructed_upper = c_x * cst.shape_function(x_real, coefs_upper) + x_real * (cst.te_thickness / 2.0)
    y_reconstructed_lower = c_x * cst.shape_function(x_real, coefs_lower) - x_real * (cst.te_thickness / 2.0)

    # 3. Calcular el Error Cuadrático Medio (MSE)
    mse_upper = np.mean((y_real_upper - y_reconstructed_upper)**2)
    mse_lower = np.mean((y_real_lower - y_reconstructed_lower)**2)
    
    print("\n==================================================")
    print(f"Error Cuadrático Medio (MSE) - Extradós: {mse_upper:.2e}")
    print(f"Error Cuadrático Medio (MSE) - Intradós: {mse_lower:.2e}")
    print("==================================================")

    if mse_upper < 1e-5 and mse_lower < 1e-5:
        print("\n¡ÉXITO: El ajuste es sumamente preciso (MSE < 1e-5)!")
    else:
        print("\nAdvertencia: El error de reconstrucción es mayor al esperado.")

    # Intentar graficar si matplotlib está instalado
    try:
        import matplotlib.pyplot as plt
        print("\nGenerando gráfico comparativo...")
        plt.figure(figsize=(10, 4))
        
        # Puntos reales
        plt.plot(x_real, y_real_upper, 'ro', label='Puntos Reales (Selig S3021)')
        plt.plot(x_real, y_real_lower, 'ro')
        
        # Curva CST reconstruida
        # Generamos una curva continua para que se vea suave en el gráfico
        x_smooth = np.linspace(0, 1, 200)
        x_smooth_coords, y_smooth_upper, y_smooth_lower = cst.generate_coordinates(coefs_upper, coefs_lower, n_points=200)
        
        plt.plot(x_smooth_coords, y_smooth_upper, 'b-', linewidth=2, label='Reconstrucción CST (6 coefs)')
        plt.plot(x_smooth_coords, y_smooth_lower, 'b-', linewidth=2)
        
        plt.title('Comparación: Perfil Real vs Reconstrucción CST')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        
        # Ruta para guardar la figura automáticamente en la carpeta de documentos
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'documents', 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'reconstructed_cst_fit.pdf')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"\nGráfico guardado exitosamente en: {output_path}")
        
        plt.show()
    except ImportError:
        print("\nNota: Matplotlib no está instalado. No se pudo mostrar ni guardar el gráfico visual,")
        print("pero los resultados numéricos demuestran la validez del ajuste.")

if __name__ == "__main__":
    run_verification()
