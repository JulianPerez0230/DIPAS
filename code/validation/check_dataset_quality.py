import pandas as pd
import numpy as np

csv_path = "c:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/dataset_cfd.csv"

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print("No se pudo leer dataset_cfd.csv (quizas aun no inicio la primera corrida):", e)
    exit()

print("=== REPORTE DE CALIDAD Y PROGRESO DATASET CFD (DIPAS) ===")
print("Perfiles unicos simulados:", df["profile_id"].nunique())
print("Registros totales (filas):", len(df))
print("Reynolds presentes:      ", sorted(df["reynolds"].unique().tolist()))
print("Angulos presentes:       ", sorted(df["alpha"].unique().tolist()))
print("Semillas representadas:  ", sorted(df["seed"].unique().tolist()))

# Verificar variacion en alpha=0, Re=150000
sub = df[(df["alpha"] == 0) & (df["reynolds"] == 150000)][["profile_id", "seed", "cl", "cd"]].drop_duplicates("profile_id").sort_values("profile_id")

if len(sub) > 0:
    print("\n--- Muestra: CL y CD a alpha=0°, Re=150.000 por perfil ---")
    print(sub.head(10).to_string(index=False))
    
    n_unique_cl = sub["cl"].nunique()
    n_total = len(sub)
    
    print("\n--- Verificacion de unicidad ---")
    if n_total < 2:
        print("[ESPERANDO] Solo 1 perfil completado hasta ahora.")
    elif n_unique_cl == n_total:
        print(f"[OK] 100% de perfiles ({n_total}/{n_total}) tienen CL DISTINTO. Pipeline fisico correcto.")
    else:
        print(f"[ALERTA] Hay duplicados: {n_unique_cl} CLs unicos para {n_total} perfiles.")
        
    print(f"Rango CL @ 0°: [{sub['cl'].min():.4f}, {sub['cl'].max():.4f}] (Span: {sub['cl'].max() - sub['cl'].min():.4f})")
    print(f"Rango CD @ 0°: [{sub['cd'].min():.5f}, {sub['cd'].max():.5f}]")
