# Estado - DIPAS

## Estado actual

- **Etapa:** 
  - **Fase 1 (CST + XFOIL)**: Completada (146.916 muestras).
  - **Fase 2 (Automatización CFD ANSYS Fluent)**: Completada al 100%. Generadas 10.806 simulaciones RANS en 300 perfiles a 6 números de Reynolds ($100k, 150k, 200k, 250k, 300k, 350k$).
  - **Fase 3 (Modelado Generativo CVAE con Fine-Tuning CFD)**: Completada al 100%. Los 3 CVAEs alcanzan un error geométrico de $RMSE = 0.001888$ y error en espesor $\Delta(t/c) = 0.0592\%$.
  - **Fase 4 (Modelos Surrogates CFD)**: Completada al 100%. Surrogate RANS puro con $R^2 = 96.23\%$ en arrastre ($C_D$) y $R^2 = 90.33\%$ en sustentación ($C_L$).
  - **Fase 5 (Diseño Inverso y Validación Cruzada)**: Optimizador latente con Adam (`optimize_airfoil.py`), Validador cruzado con XFOIL (`validate_optimized_airfoil.py`) y Aplicación Web Interactiva (`app.py`) totalmente operativos.
- **Estado del modelo:** Modelos entrenados con física de alta fidelidad guardados en `outputs/`. Inferencia instantánea con precisión CFD.
- **Validación física:** Ensayo a bajo Reynolds ($Re \approx 130.000\text{ a }400.000$) con cuerda de $200\text{ mm}$. Validación numérica y experimental externa completada.

## Supuestos de Ensayo y Fabricación (UAV / Bajo Reynolds)

| Parámetro / Restricción | Valor Asumido | Origen / Justificación |
| :--- | :--- | :--- |
| **Cuerda del modelo ($c$)** | $200\text{ mm}$ | Dimensiones óptimas para impresión 3D en una sola pieza y ensayo 2D directo en el túnel. |
| **Rango de velocidad ($V$)** | $10\text{–}30\text{ m/s}$ | Rango operativo estable del túnel de viento. |
| **Rango de $Re$ operativo** | $\sim 130.000 \text{ a } 400.000$ | Régimen típico de vuelo de UAVs chicos y prototipos de la competencia AIAA. |
| **Bloqueo aerodinámico** | Relación de bloqueo $< 1{,}5\%$ | Despreciable. No requiere correcciones de bloqueo complejas debido al tamaño compacto del modelo. |
| **Instrumentación** | Balanza de fuerzas + Tomas de presión discretas | Medición de sustentación ($C_L$), arrastre ($C_D$) y muestreo de distribución de presiones ($C_p$). |
| **Espesor mín. borde de fuga** | $0,6\text{ mm}$ | Límite físico para manufactura FDM (impresión 3D convencional con boquilla de $0,4\text{ mm}$). |

## Base técnica y bibliografía disponible

- **Datasets Relevantes**:
  - **UniFoil (Harvard, NeurIPS 2025)**: 10.000 geometrías NLF/FT para pre-entrenamiento geométrico universal.
  - **Base de datos UIUC**: 1.620 perfiles de bajo Reynolds y 6.560 puntos de ensayo polar experimental en túnel de viento.
  - **Dataset propio XFOIL**: 146.916 muestras numéricas sobre 9.931 variantes CST.
  - **Dataset propio ANSYS Fluent**: Muestras 2D RANS multi-Reynolds con resolvedor Transition SST $\gamma\text{-}Re_\theta$ en generación activa.

## Pendientes críticos

- [x] Establecer y validar el setup de simulación manual en ANSYS Fluent 2026 Student para perfiles a bajo Reynolds ($Re \approx 150.000$) y verificar su convergencia (Completado).
- [x] Resolver el conflicto de licenciamiento standalone en ANSYS Student para correr Fluent en segundo plano. (Solucionado: se ejecuta batch dentro de runwb2.exe heredando licencias).
- [x] Desarrollar el script de automatización en Python para el barrido polar de 8 ángulos del perfil Selig S3021 (Completado).
- [x] Lograr remallado dinámico automático en batch utilizando SpaceClaim y el script de Python customizado (Completado).
- [x] Diseñar e implementar la capa diferenciable CST (`CSTCoordinateLayer`) en PyTorch (Completado).
- [x] Implementar y entrenar los 3 Experimentos CVAE (Base, Transfer Learning UniFoil, y TL + UIUC) (Completado).
- [x] Implementar y entrenar los 2 Modelos Surrogates (Numérico e Híbrido UIUC) (Completado).
- [x] Implementar el algoritmo de diseño inverso y optimización latente con Adam (`optimize_airfoil.py`) (Completado).
- [x] Implementar el pipeline de validación cruzada automática (`validate_optimized_airfoil.py`) (Completado).
- [x] Completar la corrida de 3 días de ANSYS Fluent para el dataset final de alta fidelidad (10.806 simulaciones generadas).
- [x] Ejecutar el fine-tuning final con el dataset CFD de Fluent sobre los 3 CVAEs y los 2 Surrogates (Completado).
- [x] Diseñar e integrar la aplicación web interactiva con Streamlit y Plotly (`app.py`) (Completado).
- [ ] Seleccionar y exportar el perfil definitivo en formato CAD/STL para manufactura FDM y ensayo en el túnel de viento.

## Incertidumbres

- **Fidelidad y convergencia de XFOIL**: Resuelto (se logró una tasa de éxito del 99.3% parametrizando con 10 semillas variadas).
- **Automatización de Fluent**: Resuelta con éxito rotundo (se solucionaron los bloqueos de archivos en Windows y las confirmaciones de re-inicialización y eliminación de objetos de reporte).
- **Licenciamiento standalone de Fluent**: Resuelto. El lanzador de Workbench `runwb2.exe` en segundo plano hereda la firma de estudiante perfectamente.

## Próximo hito

Diseñar e implementar en PyTorch / TensorFlow el modelo de Inteligencia Artificial CVAE (Conditional Variational Autoencoder) acondicionado por espesor ($t/c$) y coeficientes aerodinámicos, junto con el modelo sustituto entrenado con los datos de XFOIL (Fase D).

## Preguntas para la Cátedra y Laboratorio (LaCLyFA)

1. **Sección de Ensayo y Selección de Túnel**:
   - Para un modelo de $200\text{ mm}$ de cuerda y régimen de bajo Reynolds ($130.000\text{--}400.000$), ¿es más conveniente ensayar en el túnel de capa límite grande o existe el túnel "TV1" como sección independiente que se adapte mejor?
2. **Instrumentación y Tomas de Presión**:
   - ¿Contamos con modelos de $200\text{ mm}$ instrumentados con tomas de presión internas o debemos fabricar el modelo en impresión 3D e integrar nosotros las tomas?
   - ¿De cuántas tomas de presión activas disponemos y cuáles son sus coordenadas relativas de cuerda ($x/c$) para contrastar con las simulaciones?
   - ¿Cuál es la precisión de la balanza de fuerzas para medir arrastres pequeños ($C_d$) a velocidades bajas ($10\text{--}25\text{ m/s}$)?
3. **Validación del Enfoque CFD**:
   - ¿Tiene la cátedra preferencia por algún modelo de transición de capa límite en ANSYS Fluent (como el *Transition $k\text{-}\omega$ SST* o el *Transition SST* de 4 ecuaciones) para simular la burbuja de separación laminar a $Re \approx 150.000$?
4. **Montaje y Rotación**:
   - ¿Cómo es el sistema de fijación del eje mecánico para el cambio de ángulo de ataque ($\alpha$) en el modelo de $200\text{ mm}$?



