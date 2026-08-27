# -*- coding: utf-8 -*-
"""
Arquitectura del Modelo Sustituto (Surrogate) en PyTorch.
Define la red MLP para predecir Cl y Cd a partir del perfil (12 CST), Alpha y Reynolds.
"""

import torch
import torch.nn as nn

class AerodynamicSurrogate(nn.Module):
    def __init__(self, input_dim=14, hidden_dims=[256, 128, 64], output_dim=2):
        super(AerodynamicSurrogate, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU()) # Activación SiLU suave para aproximación de curvas físicas
            layers.append(nn.Dropout(0.05)) # Dropout leve para evitar sobreajuste
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
