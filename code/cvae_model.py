# -*- coding: utf-8 -*-
"""
Fase 2: Arquitectura de Red Neuronal del CVAE en PyTorch con Capas de Alta Capacidad
y Pérdida Geométrica Directa (Diferenciable) usando la transformada CST en tensores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import comb

class CSTCoordinateLayer(nn.Module):
    def __init__(self, n_points=100, n_upper=6, n_lower=6, te_thickness=0.003):
        """
        Capa diferenciable que convierte coeficientes CST a coordenadas de perfil (y_up, y_low).
        Precalcula las funciones de Bernstein y de clase como buffers constantes.
        """
        super(CSTCoordinateLayer, self).__init__()
        
        self.n_points = n_points
        self.n_upper = n_upper
        self.n_lower = n_lower
        self.te_thickness = te_thickness
        
        # 1. Grilla de coseno
        theta = np.linspace(0, np.pi, n_points)
        x_grid = 0.5 * (1.0 - np.cos(theta))
        self.register_buffer('x_grid', torch.tensor(x_grid, dtype=torch.float32))
        
        # 2. Función de clase C(x) = x^0.5 * (1-x)^1.0
        c_x = (x_grid**0.5) * (1.0 - x_grid)
        self.register_buffer('c_x', torch.tensor(c_x, dtype=torch.float32))
        
        # 3. Polinomios de Bernstein precalculados como matrices (n_coefs, n_points)
        B_upper = np.zeros((n_upper, n_points))
        for i in range(n_upper):
            B_upper[i, :] = comb(n_upper - 1, i) * (x_grid**i) * ((1.0 - x_grid)**(n_upper - 1 - i))
            
        B_lower = np.zeros((n_lower, n_points))
        for i in range(n_lower):
            B_lower[i, :] = comb(n_lower - 1, i) * (x_grid**i) * ((1.0 - x_grid)**(n_lower - 1 - i))
            
        self.register_buffer('B_upper', torch.tensor(B_upper, dtype=torch.float32))
        self.register_buffer('B_lower', torch.tensor(B_lower, dtype=torch.float32))
        
    def forward(self, coefs):
        """
        Dada una matriz de coeficientes (Batch, n_upper + n_lower), retorna las
        coordenadas y_up (Batch, n_points) e y_low (Batch, n_points).
        """
        # Separar coeficientes extradós (upper) e intradós (lower)
        coefs_upper = coefs[:, :self.n_upper]
        coefs_lower = coefs[:, self.n_upper:]
        
        # Multiplicación de matrices para obtener las funciones de forma S(x)
        # (Batch, n_coefs) @ (n_coefs, n_points) -> (Batch, n_points)
        s_upper = torch.matmul(coefs_upper, self.B_upper)
        s_lower = torch.matmul(coefs_lower, self.B_lower)
        
        # Geometrías combinadas con el espesor de borde de fuga
        y_upper = self.c_x.unsqueeze(0) * s_upper + self.x_grid.unsqueeze(0) * (self.te_thickness / 2.0)
        y_lower = self.c_x.unsqueeze(0) * s_lower - self.x_grid.unsqueeze(0) * (self.te_thickness / 2.0)
        
        return y_upper, y_lower

class CVAE(nn.Module):
    def __init__(self, cst_dim=12, cond_dim=0, latent_dim=6, hidden_dims=[256, 128, 64]):
        """
        CVAE de alta capacidad para perfiles aerodinámicos.
        """
        super(CVAE, self).__init__()
        
        self.cst_dim = cst_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        
        # Capa de reconstrucción física
        self.cst_to_coord = CSTCoordinateLayer(n_points=100, n_upper=6, n_lower=6, te_thickness=0.003)
        
        # -----------------------------
        # ENCODER: q(z | x, c)
        # -----------------------------
        encoder_input_dim = cst_dim + cond_dim
        
        modules = []
        in_dim = encoder_input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim)
                )
            )
            in_dim = h_dim
            
        self.encoder_backbone = nn.Sequential(*modules)
        
        # Salidas latentes
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # -----------------------------
        # DECODER: p(x | z, c)
        # -----------------------------
        decoder_input_dim = latent_dim + cond_dim
        
        modules = []
        in_dim = decoder_input_dim
        for h_dim in reversed(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim)
                )
            )
            in_dim = h_dim
            
        self.decoder_backbone = nn.Sequential(*modules)
        self.fc_recon = nn.Linear(hidden_dims[0], cst_dim)
        
    def encode(self, x, c=None):
        if self.cond_dim > 0:
            if c is None:
                raise ValueError("Se requiere cond_tensor para modo condicional.")
            inputs = torch.cat([x, c], dim=1)
        else:
            inputs = x
            
        h = self.encoder_backbone(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, c=None):
        if self.cond_dim > 0:
            if c is None:
                raise ValueError("Se requiere cond_tensor para modo condicional.")
            inputs = torch.cat([z, c], dim=1)
        else:
            inputs = z
            
        h = self.decoder_backbone(inputs)
        recon_cst = self.fc_recon(h)
        return recon_cst
        
    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_cst = self.decode(z, c)
        return recon_cst, mu, logvar

def cvae_loss_function(recon_cst, target_cst, mu, logvar, cst_layer, beta=1.0, geom_weight=10.0):
    """
    Función de pérdida híbrida:
    Loss = MSE(CST) + geom_weight * MSE(Coordenadas_y) + beta * KLD.
    
    Al evaluar directamente las coordenadas y_up/y_low dentro de PyTorch, obligamos
    a la red a priorizar la física real de la forma del perfil.
    """
    # 1. Pérdida sobre los coeficientes CST abstractos
    cst_loss = F.mse_loss(recon_cst, target_cst, reduction='mean')
    
    # 2. Pérdida sobre las coordenadas físicas (Diferenciable)
    y_up_rec, y_low_rec = cst_layer(recon_cst)
    y_up_tgt, y_low_tgt = cst_layer(target_cst)
    
    geom_loss_up = F.mse_loss(y_up_rec, y_up_tgt, reduction='mean')
    geom_loss_low = F.mse_loss(y_low_rec, y_low_tgt, reduction='mean')
    geom_loss = geom_loss_up + geom_loss_low
    
    # 3. Regularización KL
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    # Pérdida total combinada
    total_loss = cst_loss + geom_weight * geom_loss + beta * kld_loss
    
    return total_loss, cst_loss, geom_loss, kld_loss
