# -*- coding: utf-8 -*-
"""
DIPAS - Script de Validacion de Calidad del Dataset CFD
========================================================
Compara los coeficientes aerodinamicos del dataset de ANSYS Fluent (DIPAS)
contra los datos experimentales de tunel de viento de la UIUC (Vols. 1-3).

Genera graficas de curvas polares superpuestas (CL vs alpha y CD vs alpha)
y calcula el error absoluto medio (MAE) por rango de angulo de ataque,
permitiendo detectar donde el modelo Transition-SST diverge del experimento.

Uso:
    pip install pandas matplotlib numpy
    python validate_cfd_quality.py

Salida:
    plots/validation/validation_<perfil>_Re<reynolds>.png
    data/validation_report.csv
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Configuracion
SCRIPT_DIR  = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR    = PROJECT_DIR / "data"

CFD_CSV   = DATA_DIR / "dataset_cfd.csv"
UIUC_DIRS = [
    DATA_DIR / "uiuc_tunnel_data" / "volume01",
    DATA_DIR / "uiuc_tunnel_data" / "volume02",
    DATA_DIR / "uiuc_tunnel_data" / "volume03",
]
PLOTS_DIR = PROJECT_DIR / "plots" / "validation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Perfiles a comparar: semilla DIPAS -> nombre UIUC
PROFILES_TO_COMPARE = {
    "e387":  "E387",
    "s3021": "S3021",
    "s1223": "S1223",
    "s7012": "S7012",
    "e374":  "E374",
}

TARGET_REYNOLDS    = [100_000, 150_000, 200_000]
RE_TOLERANCE       = 20_000
RELIABLE_ALPHA_MAX = 8.0

def parse_uiuc_file(filepath, coeff_name):
    rows = []
    airfoil = None
    reynolds = None
    in_data  = False

    with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"Airfoil:\s*(.+)", line)
            if m:
                airfoil = re.sub(r"\s*\([A-Z]\)", "", m.group(1).strip()).upper()
                in_data = False
                continue
            m = re.match(r"\s*(\d+)\s*$", line)
            if m and airfoil:
                c = int(m.group(1))
                if 10_000 < c < 2_000_000:
                    reynolds = c
                    in_data  = False
                    continue
            if "alpha" in line.lower() and ("cl" in line.lower() or "cd" in line.lower()):
                in_data = True
                continue
            if in_data and airfoil and reynolds:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rows.append({"airfoil": airfoil, "reynolds": reynolds,
                                     "alpha": float(parts[0]), coeff_name: float(parts[1])})
                    except ValueError:
                        in_data = False

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_uiuc_data():
    lf, df = [], []
    for d in UIUC_DIRS:
        for f in sorted(d.glob("LIFT*.TXT")):
            r = parse_uiuc_file(f, "cl")
            if not r.empty: lf.append(r)
        for f in sorted(d.glob("DRAG*.TXT")):
            r = parse_uiuc_file(f, "cd")
            if not r.empty: df.append(r)

    lift_all = pd.concat(lf, ignore_index=True) if lf else pd.DataFrame()
    drag_all = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
    if lift_all.empty or drag_all.empty:
        return {}

    uiuc = pd.merge(lift_all, drag_all, on=["airfoil","reynolds","alpha"], how="outer")
    uiuc["reynolds"] = uiuc["reynolds"].astype(int)
    return {n: uiuc[uiuc["airfoil"]==n].reset_index(drop=True) for n in uiuc["airfoil"].unique()}


def compute_mae(cfd_sub, uiuc_sub, col):
    if col not in cfd_sub.columns or col not in uiuc_sub.columns:
        return float("nan")
    if uiuc_sub.empty:
        return float("nan")
    interp = np.interp(cfd_sub["alpha"].values, uiuc_sub["alpha"].values, uiuc_sub[col].values)
    return float(np.mean(np.abs(cfd_sub[col].values - interp)))


def validate_profile(seed, uiuc_name, cfd_df, uiuc_data, report_rows):
    cfd_profile = cfd_df[cfd_df["seed"] == seed.lower()]
    if cfd_profile.empty:
        print(f"  [OMITIDO] Semilla '{seed}' no encontrada en dataset_cfd.csv")
        return

    key = uiuc_name.upper()
    if key not in uiuc_data:
        print(f"  [OMITIDO] Perfil '{uiuc_name}' no encontrado en datos UIUC")
        return

    uiuc_profile = uiuc_data[key]

    for target_re in TARGET_REYNOLDS:
        cfd_re = cfd_profile[cfd_profile["reynolds"] == target_re].sort_values("alpha")
        if cfd_re.empty:
            continue

        closest_re = min(uiuc_profile["reynolds"].unique(), key=lambda r: abs(r - target_re))
        if abs(closest_re - target_re) > RE_TOLERANCE:
            print(f"  [AVISO] {uiuc_name} Re={target_re}: UIUC mas cercano {closest_re} fuera de tolerancia")
            continue

        uiuc_re = uiuc_profile[uiuc_profile["reynolds"] == closest_re].sort_values("alpha")

        # Reporte de MAE por rango de angulo
        for alpha_max, label in [(RELIABLE_ALPHA_MAX, "confiable<=8deg"), (12.0, "completo<=12deg")]:
            cfd_sub  = cfd_re[cfd_re["alpha"] <= alpha_max]
            uiuc_sub = uiuc_re[uiuc_re["alpha"] <= alpha_max]
            report_rows.append({
                "perfil_dipas": seed, "perfil_uiuc": uiuc_name,
                "re_dipas": target_re, "re_uiuc": closest_re,
                "rango_alpha": label,
                "MAE_CL": round(compute_mae(cfd_sub, uiuc_sub, "cl"), 5),
                "MAE_CD": round(compute_mae(cfd_sub, uiuc_sub, "cd"), 6),
            })

        # Graficas comparativas
        fig = plt.figure(figsize=(16, 5))
        fig.suptitle(
            f"Validacion DIPAS vs UIUC  |  {uiuc_name}   Re_CFD={target_re:,}   Re_UIUC={closest_re:,}",
            fontsize=13, fontweight="bold"
        )
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # Panel CL vs alpha
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(cfd_re["alpha"], cfd_re["cl"], "b-o", ms=5, lw=2, label="DIPAS Fluent (RANS Transition-SST)")
        if "cl" in uiuc_re.columns:
            ax1.plot(uiuc_re["alpha"], uiuc_re["cl"], "r--s", ms=5, lw=2, label=f"UIUC Experimental (Re={closest_re:,})")
        ax1.axvline(RELIABLE_ALPHA_MAX, color="orange", ls=":", lw=2, label=f"alfa={RELIABLE_ALPHA_MAX} lim. confiable")
        ax1.set_xlabel("alfa [deg]"); ax1.set_ylabel("CL")
        ax1.set_title("Sustentacion"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

        # Panel CD vs alpha
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(cfd_re["alpha"], cfd_re["cd"], "b-o", ms=5, lw=2, label="DIPAS Fluent")
        if "cd" in uiuc_re.columns:
            ax2.plot(uiuc_re["alpha"], uiuc_re["cd"], "r--s", ms=5, lw=2, label=f"UIUC Experimental (Re={closest_re:,})")
        ax2.axvline(RELIABLE_ALPHA_MAX, color="orange", ls=":", lw=2, label=f"alfa={RELIABLE_ALPHA_MAX} lim. confiable")
        ax2.set_xlabel("alfa [deg]"); ax2.set_ylabel("CD")
        ax2.set_title("Arrastre"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

        # Polar aerodinamica CL vs CD
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(cfd_re["cd"], cfd_re["cl"], "b-o", ms=5, lw=2, label="DIPAS Fluent")
        if "cl" in uiuc_re.columns and "cd" in uiuc_re.columns:
            ax3.plot(uiuc_re["cd"], uiuc_re["cl"], "r--s", ms=5, lw=2, label="UIUC Experimental")
        ax3.set_xlabel("CD"); ax3.set_ylabel("CL")
        ax3.set_title("Polar aerodinamica"); ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"validation_{seed}_Re{target_re}.png"
        fig.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Grafica guardada: plots/validation/{fname}")


if __name__ == "__main__":
    print("=" * 60)
    print("  DIPAS - Validacion de Calidad del Dataset CFD")
    print("=" * 60)

    print("\n>> Cargando dataset CFD de DIPAS...")
    cfd_df = pd.read_csv(CFD_CSV)
    cfd_df["seed"] = cfd_df["seed"].str.lower().str.strip()
    print(f"   {len(cfd_df)} registros | {cfd_df['seed'].nunique()} semillas unicas")

    print("\n>> Cargando datos experimentales de la UIUC...")
    uiuc_data = load_uiuc_data()
    print(f"   {len(uiuc_data)} perfiles experimentales cargados")

    report_rows = []
    for seed, uiuc_name in PROFILES_TO_COMPARE.items():
        print(f"\n[{seed.upper()} vs {uiuc_name}]")
        validate_profile(seed, uiuc_name, cfd_df, uiuc_data, report_rows)

    if report_rows:
        report_path = DATA_DIR / "validation_report.csv"
        pd.DataFrame(report_rows).to_csv(report_path, index=False)
        print(f"\n>> Reporte guardado en: data/validation_report.csv")
        print("\n" + "=" * 60)
        print("  RESUMEN DE ERRORES MAE")
        print("=" * 60)
        print(pd.DataFrame(report_rows).to_string(index=False))

    print("\n>> Validacion finalizada. Revisa las graficas en plots/validation/")
