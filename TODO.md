# Pendientes - DIPAS

## Ahora

- [x] Realizar la primera simulación manual en ANSYS Fluent con un perfil de bajo Reynolds (ej. Selig S3021 a $Re = 150.000$) y documentar el flujo de trabajo (recorrido) (completado).
- [ ] Confirmar los supuestos del túnel de viento y el modelo CFD en ANSYS (especialmente el modelo de turbulencia Transition SST) con la cátedra.
- [x] Definir el alcance técnico del proyecto y validar la arquitectura general con el usuario (completado).
- [x] Inicializar la estructura de carpetas locales y archivos de configuración (completado).
- [x] Implementar script en Python para parametrización CST con restricción de espesor mínimo en el borde de fuga (`cst_generator.py`).
- [x] Desarrollar e integrar el wrapper de XFOIL en Python para simular de forma secuencial y guardar coeficientes aerodinámicos (`xfoil_wrapper.py`).
- [x] Consolidar el dataset de pre-entrenamiento: generar y simular con éxito 9.931 variantes en lote (`dataset.csv`).

## Después

- [x] Desarrollar el script de automatización en Python para Fluent (wb_run_polar.py validado a -2, 0, 2, 4, 6, 8, 10, 12 deg).
- [x] Lograr remallado dinámico automático en batch utilizando SpaceClaim y el script de Python customizado (completado con éxito).
- [x] Generar el dataset de ajuste fino de alta fidelidad para el primer Reynolds de diseño (`data/dataset_cfd.csv`).
- [x] Expandir el dataset CFD en Fluent a múltiples Reynolds ($100.000, 150.000, 200.000, 250.000, 300.000, 350.000$), sumando 10.806 simulaciones (Completado).
- [x] Descargar e integrar el dataset UniFoil (HARVARD) y los datos experimentales UIUC para el pipeline de pre-entrenamiento y validación externa ciega (Completado).
- [x] Diseñar e implementar el modelo de red neuronal CVAE condicionado en PyTorch con capa diferenciable CST (`cvae_model.py`) (Completado con los 3 experimentos).
- [x] Desarrollar y entrenar los modelos sustitutos (*surrogate models*) numérico e híbrido (`surrogate_model.py`) (Completado).
- [x] Integrar el pipeline interactivo de diseño inverso y optimización latente con Adam (`optimize_airfoil.py`) (Completado).
- [x] Implementar y validar el pipeline de validación cruzada automática (`validate_optimized_airfoil.py`) (Completado).
- [x] Ejecutar el fine-tuning final con el dataset completo de Fluent sobre los 3 CVAEs y los 2 Surrogates (Completado).
- [ ] Seleccionar el perfil definitivo para la validación física en base al desempeño del Top 5.

## Más adelante

- [ ] Diseñar el modelo CAD del perfil extruido (cuerda de $200\text{ mm}$) con eje de anclaje estructural y conductos para sensores de presión.
- [ ] Fabricar el modelo físico mediante impresión 3D convencional FDM en una sola pieza.
- [ ] Ejecutar los ensayos experimentales en el túnel de viento de la facultad.
- [ ] Comparar los resultados de performance experimental vs. predicciones.
- [ ] Documentar el simulation gap y redactar el reporte de validación final.
