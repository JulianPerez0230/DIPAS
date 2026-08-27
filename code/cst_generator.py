import numpy as np
from scipy.optimize import lsq_linear

class CSTParametrization:
    def __init__(self, n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003):
        """
        Clase para manejar la parametrización de perfiles alares mediante CST (Class Shape Transformation).
        
        Args:
            n_coefs_upper (int): Número de coeficientes de Bernstein para el extradós (superficie superior).
            n_coefs_lower (int): Número de coeficientes de Bernstein para el intradós (superficie inferior).
            te_thickness (float): Espesor relativo del borde de fuga (t/c en x=1). Por defecto 0.003 (~0.6mm para cuerda de 200mm).
        """
        self.n_upper = n_coefs_upper
        self.n_lower = n_coefs_lower
        self.te_thickness = te_thickness

    @staticmethod
    def class_function(x, n1=0.5, n2=1.0):
        """
        Función de clase C(x) que define la topología del perfil (punta redonda, borde de fuga afilado).
        """
        # Evitar errores de dominio en x=0 o x=1
        x = np.clip(x, 0.0, 1.0)
        return (x**n1) * ((1.0 - x)**n2)

    @staticmethod
    def bernstein_polynomial(x, i, n):
        """
        Calcula el i-ésimo polinomio de Bernstein de orden n.
        """
        from scipy.special import comb
        return comb(n, i) * (x**i) * ((1.0 - x)**(n - i))

    def shape_function(self, x, coefficients):
        """
        Función de forma S(x) obtenida como combinación lineal de polinomios de Bernstein.
        """
        n = len(coefficients) - 1
        s = np.zeros_like(x)
        for i, coef in enumerate(coefficients):
            s += coef * self.bernstein_polynomial(x, i, n)
        return s

    def generate_coordinates(self, coefs_upper, coefs_lower, n_points=100):
        """
        Genera las coordenadas (x, y) de las superficies superior e inferior.
        
        Args:
            coefs_upper (list/array): Coeficientes CST para la superficie superior.
            coefs_lower (list/array): Coeficientes CST para la superficie inferior.
            n_points (int): Cantidad de puntos por superficie.
            
        Returns:
            x (array): Coordenadas x distribuidas con espaciamiento tipo coseno.
            y_upper (array): Coordenadas y de la superficie superior.
            y_lower (array): Coordenadas y de la superficie inferior.
        """
        # Distribución de coseno para mayor densidad de puntos en borde de ataque y fuga
        theta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1.0 - np.cos(theta))
        
        # Funciones de clase
        c_x = self.class_function(x)
        
        # Funciones de forma
        s_upper = self.shape_function(x, coefs_upper)
        s_lower = self.shape_function(x, coefs_lower)
        
        # Geometrías sin espesor de borde de fuga
        y_upper = c_x * s_upper + x * (self.te_thickness / 2.0)
        y_lower = c_x * s_lower - x * (self.te_thickness / 2.0)
        
        return x, y_upper, y_lower

    def fit_airfoil(self, x, y_upper, y_lower):
        """
        Ajusta coeficientes CST a partir de un perfil existente (coordenadas x, y).
        Útil para procesar la base de datos de UIUC hacia espacio de coeficientes.
        """
        c_x = self.class_function(x)
        
        # Restamos la contribución del espesor del borde de fuga
        y_u_mod = y_upper - x * (self.te_thickness / 2.0)
        y_l_mod = y_lower + x * (self.te_thickness / 2.0)
        
        # Matrices de diseño para mínimos cuadrados lineales
        A_upper = np.zeros((len(x), self.n_upper))
        A_lower = np.zeros((len(x), self.n_lower))
        
        for i in range(self.n_upper):
            A_upper[:, i] = c_x * self.bernstein_polynomial(x, i, self.n_upper - 1)
            
        for i in range(self.n_lower):
            A_lower[:, i] = c_x * self.bernstein_polynomial(x, i, self.n_lower - 1)
            
        # Resolvemos mínimos cuadrados lineales con restricciones si fuera necesario
        # En este caso por mínimos cuadrados simples:
        coefs_upper, _, _, _ = np.linalg.lstsq(A_upper, y_u_mod, rcond=None)
        coefs_lower, _, _, _ = np.linalg.lstsq(A_lower, y_l_mod, rcond=None)
        
        return coefs_upper, coefs_lower

if __name__ == "__main__":
    # Ejemplo de prueba rápida de generación y ajuste:
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    
    # Coeficientes típicos para un perfil NACA 0012 aproximado
    a_u = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    a_l = [-0.15, -0.15, -0.15, -0.15, -0.15, -0.15]
    
    x, yu, yl = cst.generate_coordinates(a_u, a_l, n_points=50)
    print("Coordenadas x superiores de prueba:")
    print(np.round(x[:5], 4))
    print("Coordenadas y superiores de prueba:")
    print(np.round(yu[:5], 4))
    
    # Probamos el ajuste inverso
    fit_u, fit_l = cst.fit_airfoil(x, yu, yl)
    print("\nCoeficientes ajustados superiores (deben coincidir con a_u si no hay error):")
    print(np.round(fit_u, 4))
