# -*- coding: utf-8 -*-
"""
Fase 2 y 3: Cargador de Datos (Dataset y DataLoader) de PyTorch para DIPAS.
Calcula el espesor relativo máximo (t/c) al vuelo a partir de los coeficientes CST
y lo añade como una variable de condicionamiento junto con Cl, Cd y Reynolds.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.special import comb

def calculate_tc_vectorized(cst_array, te_thickness=0.003, n_points=100):
    """
    Calcula de manera vectorizada el espesor relativo máximo (t/c) para una matriz
    de coeficientes CST de tamaño (N, 12).
    """
    theta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(theta))
    c_x = (x**0.5) * (1.0 - x)
    
    # Polinomios de Bernstein (orden 5 para 6 coeficientes)
    B = np.zeros((6, n_points))
    for i in range(6):
        B[i, :] = comb(5, i) * (x**i) * ((1.0 - x)**(5 - i))
        
    # S_u e S_l tienen dimensiones (N, n_points)
    S_u = np.dot(cst_array[:, :6], B)
    S_l = np.dot(cst_array[:, 6:], B)
    
    y_u = c_x * S_u + x * (te_thickness / 2.0)
    y_l = c_x * S_l - x * (te_thickness / 2.0)
    
    thickness = y_u - y_l
    t_c = np.max(thickness, axis=1)
    return t_c

class AirfoilCSTDataset(Dataset):
    def __init__(self, csv_path, use_conditions=False, cond_cols=["reynolds", "cl", "cd"], sample_size=None):
        """
        Dataset de PyTorch para perfiles en formato de parámetros CST.
        Calcula t/c automáticamente como cuarta condición si use_conditions=True.
        """
        self.csv_path = csv_path
        self.use_conditions = use_conditions
        self.cond_cols = cond_cols
        
        # Cargar los datos
        self.df = pd.read_csv(csv_path)
        
        # Opcional: tomar una submuestra (para depurar o entrenar con subconjuntos como 10k XFOIL)
        if sample_size is not None and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
        # Columnas de los coeficientes CST
        self.cst_cols = [
            "au0", "au1", "au2", "au3", "au4", "au5",
            "al0", "al1", "al2", "al3", "al4", "al5"
        ]
        
        # Extraer tensores de geometría
        self.cst_data = torch.tensor(self.df[self.cst_cols].values, dtype=torch.float32)
        
        # Extraer tensores de condiciones si se solicita
        if self.use_conditions:
            # Aseguramos que existan todas las columnas básicas (cl, cd, reynolds)
            for col in self.cond_cols:
                if col not in self.df.columns:
                    raise KeyError(f"La columna de condición '{col}' no existe en el dataset {csv_path}")
            
            # Copiamos condiciones y calculamos t/c al vuelo
            cond_df = self.df[self.cond_cols].copy()
            
            # Calcular espesor relativo
            cst_np = self.df[self.cst_cols].values
            t_c_vals = calculate_tc_vectorized(cst_np)
            cond_df["t_c"] = t_c_vals
            
            # Normalización min-max para las condiciones (crucial para la convergencia en PyTorch)
            self.cond_min = cond_df.min(axis=0)
            self.cond_max = cond_df.max(axis=0)
            
            denom = (self.cond_max - self.cond_min)
            denom[denom == 0] = 1.0
            
            cond_data_norm = (cond_df - self.cond_min) / denom
            self.cond_data = torch.tensor(cond_data_norm.values, dtype=torch.float32)
        else:
            self.cond_data = torch.zeros((len(self.df), 1), dtype=torch.float32)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        x = self.cst_data[idx]
        if self.use_conditions:
            c = self.cond_data[idx]
            return x, c
        else:
            return x, torch.tensor([], dtype=torch.float32)

def get_dataloader(csv_path, batch_size=32, shuffle=True, use_conditions=False, cond_cols=["reynolds", "cl", "cd"], sample_size=None):
    dataset = AirfoilCSTDataset(csv_path, use_conditions=use_conditions, cond_cols=cond_cols, sample_size=sample_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader, dataset
