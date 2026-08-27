# -*- coding: utf-8 -*-
# Python script to plot CFD polar results for Selig S3021 airfoil
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    csv_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/s3021_polar_cfd.csv"
    output_png = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/s3021_polar_plot.png"
    
    if not os.path.exists(csv_path):
        print("Error: CSV file not found at " + csv_path)
        return
        
    df = pd.read_csv(csv_path)
    
    # Configuracion de estilo profesional
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Grafico CL y CD vs Alpha
    color = 'tab:blue'
    ax1.set_xlabel('Angle of Attack, alpha [deg]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Lift Coefficient, CL', color=color, fontsize=11, fontweight='bold')
    line1 = ax1.plot(df['alpha'], df['cl'], marker='o', linewidth=2.0, color=color, label='CL')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Aerodynamic Coefficients vs Alpha', fontsize=12, fontweight='bold', pad=10)
    
    ax1_twin = ax1.twinx()
    color = 'tab:red'
    ax1_twin.set_ylabel('Drag Coefficient, CD', color=color, fontsize=11, fontweight='bold')
    line2 = ax1_twin.plot(df['alpha'], df['cd'], marker='s', linewidth=2.0, linestyle='--', color=color, label='CD')
    ax1_twin.tick_params(axis='y', labelcolor=color)
    
    # Leyenda combinada para el primer panel
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True)
    
    # 2. Polar Aerodinamica CL vs CD
    ax2.plot(df['cd'], df['cl'], marker='o', linewidth=2.0, color='purple', label='Polar Curve')
    ax2.set_xlabel('Drag Coefficient, CD', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Lift Coefficient, CL', fontsize=11, fontweight='bold')
    ax2.set_title('Drag Polar (CL vs CD)', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print("Plot saved successfully to: " + output_png)

if __name__ == "__main__":
    plot_results()
