# ✈️ DIPAS: Deep Inverse Aerodynamic Synthesis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dipas-airfoils.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Physics: XFOIL & Fluent](https://img.shields.io/badge/Physics-XFOIL%20%7C%20ANSYS%20Fluent-00AFB5.svg)](https://web.mit.edu/drela/Public/web/xfoil/)

> **Diseño Inverso de Perfiles Aerodinámicos mediante Autoencoders Variacionales Condicionales (CVAE) y Validación Multi-Fidelidad (XFOIL & RANS CFD Transition SST).**
> 
> *Universidad Nacional de La Plata (UNLP) — Facultad de Ingeniería*

---

## 🎯 Descripción General

**DIPAS** es un entorno interactivo y computacional para el diseño inverso de superficies sustentantes optimizadas a bajos y medios números de Reynolds ( = 100.000 - 350.000$). Combina redes neuronales generativas profundas con simulación física multi-fidelidad para sintetizar geometrías viables en milisegundos y validarlas numéricamente con rigor experimental.

### 🌟 Características Principales:
1. **Síntesis Generativa Condicional (CVAE):**
   - Espacio latente continuo y físicamente estructurado.
   - Genera 5 arquetipos de diseño especializados:
     - **#1 Óptimo Balanceado:** Compromiso nominal de máxima eficiencia /D$.
     - **#2 Mínimo Arrastre (Min Drag):** Penalización agresiva de $ para alto planeo.
     - **#3 Mayor Espesor (Estructural):** $+20\%\ t/c$ para volumen de larguero/baterías.
     - **#4 Perfil Delgado (Alta Velocidad):** $-20\%\ t/c$ para baja resistencia de forma.
     - **#5 Bajo Momento (Estabilidad):** Minimización de $|C_m|$ para fácil trimado.
2. **Parametrización Analítica Suave (CST de Orden 4):**
   - Transforma 12 coeficientes analíticos (6 extradós, 6 intradós) en curvas continuas de clase/forma (*Class-Shape Transformation* de Kulfan).
3. **Validación Multi-Fidelidad:**
   - **Baja Fidelidad (XFOIL):** Método de paneles con formulación integral de capa límite acoplada (^N$).
   - **Alta Fidelidad (ANSYS Fluent / CFD):** RANS 2D con modelo de transición laminar-turbulenta $\gamma\text{-}\widetilde{Re}_{\theta\text{t}}$ SST para capturar la burbuja laminar de separación (LSB).
4. **Exportación Automatizada a Manufactura y CAD:**
   - Coordenadas Selig/UIUC .DAT con escalado milimétrico para corte CNC / FDM.
   - Mallas computacionales no estructuradas de alta resolución .MSH generadas vía Gmsh (^+ \approx 1$).
   - Diarios automatizados de ejecución .JOU para ANSYS Fluent.

---

## 📊 Datasets y Respaldos Experimentales

- **DIPAS Base (Elaboración Propia):** 9.931 perfiles únicos parametrizados en CST y 146.916 simulaciones numéricas viscosas en XFOIL ( \in [100k, 300k]$, $\alpha \in [-4^\circ, 10^\circ]$).
- **DIPAS CFD High-Fidelity (Elaboración Propia):** 300 perfiles de alta fidelidad con 10.806 simulaciones 2D RANS Transition SST.
- **UniFoil Geometric Dataset:** 10.000 geometrías CST para pre-entrenamiento global (*Transfer Learning*).
- **UIUC Database:** 1.620 perfiles experimentales y 6.560 puntos de ensayo en túnel de viento a bajas velocidades.

---

## 🚀 Instalación y Uso Local

`ash
# 1. Clonar el repositorio
git clone https://github.com/JulianPerez0230/DIPAS.git
cd DIPAS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar la aplicación Streamlit
streamlit run code/app.py
`

---

## 🏛️ Estructura del Proyecto

`
DIPAS/
├── code/                   # Código fuente de inferencia y simulación
│   ├── app.py             # Aplicación interactiva Streamlit
│   ├── dipas_engine.py    # Motor de inferencia y optimización latente
│   ├── cvae_model.py      # Arquitectura de la red neuronal CVAE
│   ├── surrogate_model.py # Modelo sustituto aerodinámico
│   ├── cst_generator.py   # Parametrización analítica CST Kulfan
│   ├── xfoil_wrapper.py   # Automatización desacoplada de XFOIL
│   └── cfd_automation/    # Mallas Gmsh y scripts para ANSYS Fluent
├── data/                   # Datasets compilados (.csv)
├── outputs/                # Pesos de los modelos entrenados (.pth)
├── .streamlit/             # Configuración de tema visual
├── packages.txt            # Dependencias del sistema (Linux / Cloud)
├── requirements.txt        # Dependencias de Python
└── LICENSE                 # Licencia de código abierto MIT
`

---

## 📜 Licencia y Cita

Distribuido bajo la Licencia **MIT**. Consulta el archivo LICENSE para más información.

**Citas y Referencias:**
* *Drela, M. (1989). XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils. MIT.*
* *Kulfan, B. M. (2008). Universal Parametric Geometry Representation Method. Journal of Aircraft.*
* *Selig, M. S. et al. (1995). Summary of Low-Speed Airfoil Data. UIUC.*

---
**Сделано Jota (Hecho por Jota)** • *Universidad Nacional de La Plata (UNLP)*
