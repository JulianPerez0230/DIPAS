# -*- coding: utf-8 -*-
"""
Fase 2: Script de Entrenamiento del Modelo Base (Autoencoder geométrico puro).
Entrena sobre el dataset de la UIUC (uiuc_cst_dataset.csv) utilizando
capas de alta capacidad, espacio latente z=6 y pérdida geométrica directa.
"""

import os
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import random_split

# Asegurar importación de módulos hermanos en code/
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

from cvae_model import CVAE, cvae_loss_function
from cvae_dataset import AirfoilCSTDataset, DataLoader
from cst_generator import CSTParametrization

def main():
    # Rutas del proyecto
    dataset_path = SCRIPT_DIR.parent / "data" / "uiuc_cst_dataset.csv"
    output_dir = SCRIPT_DIR.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_path = output_dir / "base_autoencoder.pth"
    loss_plot_path = output_dir / "base_ae_losses.png"
    recon_plot_path = output_dir / "base_ae_reconstruction.png"
    
    if not dataset_path.exists():
        print(f"❌ Dataset no encontrado en: {dataset_path}")
        sys.exit(1)
        
    print(f">> Iniciando entrenamiento del Autoencoder Base (Alta Capacidad)...")
    
    # 1. Configuración de hiperparámetros
    epochs = 200     # Aumentamos a 200 épocas
    batch_size = 32
    lr = 0.0005      # Tasa de aprendizaje ligeramente más baja para estabilizar la red ancha
    latent_dim = 6   # Espacio latente óptimo verificado
    beta = 0.01      # Ponderación KLD
    geom_weight = 20.0 # Peso fuerte para forzar reconstrucción de la geometría física
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Utilizando dispositivo: {device}")
    
    # 2. Cargar Dataset
    full_dataset = AirfoilCSTDataset(str(dataset_path), use_conditions=False)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"   Dataset UIUC cargado: {len(full_dataset)} muestras.")
    print(f"   - Entrenamiento: {len(train_dataset)} muestras.")
    print(f"   - Validación: {len(val_dataset)} muestras.")
    
    # 3. Inicializar Modelo, Optimizador
    model = CVAE(cst_dim=12, cond_dim=0, latent_dim=latent_dim, hidden_dims=[256, 128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Listas para guardar las pérdidas
    train_losses, train_recons, train_geoms = [], [], []
    val_losses, val_recons, val_geoms = [], [], []
    
    # 4. Bucle de Entrenamiento
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_geom = 0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            
            loss, recon_loss, geom_loss, kld_loss = cvae_loss_function(
                recon_x, batch_x, mu, logvar, model.cst_to_coord, beta=beta, geom_weight=geom_weight
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
            epoch_recon += recon_loss.item() * batch_size
            epoch_geom += geom_loss.item() * batch_size
            
        epoch_loss /= len(train_loader.dataset)
        epoch_recon /= len(train_loader.dataset)
        epoch_geom /= len(train_loader.dataset)
        
        train_losses.append(epoch_loss)
        train_recons.append(epoch_recon)
        train_geoms.append(epoch_geom)
        
        # Validación
        model.eval()
        v_loss, v_recon, v_geom = 0, 0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                recon_x, mu, logvar = model(batch_x)
                loss, recon_loss, geom_loss, kld_loss = cvae_loss_function(
                    recon_x, batch_x, mu, logvar, model.cst_to_coord, beta=beta, geom_weight=geom_weight
                )
                
                v_loss += loss.item() * len(batch_x)
                v_recon += recon_loss.item() * len(batch_x)
                v_geom += geom_loss.item() * len(batch_x)
                
        v_loss /= len(val_loader.dataset)
        v_recon /= len(val_loader.dataset)
        v_geom /= len(val_loader.dataset)
        
        val_losses.append(v_loss)
        val_recons.append(v_recon)
        val_geoms.append(v_geom)
        
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | "
                  f"Train Loss: {epoch_loss:.5f} (CST MSE: {epoch_recon:.5f}, Geom MSE: {epoch_geom:.6f}) | "
                  f"Val Loss: {v_loss:.5f} (CST MSE: {v_recon:.5f}, Geom MSE: {v_geom:.6f})")
            
    # 5. Guardar Checkpoint del Modelo
    torch.save(model.state_dict(), model_save_path)
    print(f"\n>> Modelo de alta capacidad guardado en: {model_save_path}")
    
    # 6. Graficar Curva de Pérdidas
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (Total)")
    plt.plot(val_losses, '--', label="Val Loss (Total)")
    plt.plot(train_geoms, label="Train Geom (MSE Coords)")
    plt.plot(val_geoms, ':', label="Val Geom (MSE Coords)")
    plt.yscale("log")
    plt.title("Pérdidas del Entrenamiento con Pérdida Geométrica Directa")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (Escala Log)")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(loss_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # 7. Graficar Reconstrucción de Prueba
    model.eval()
    with torch.no_grad():
        test_x, _ = val_dataset[0]
        test_x_tensor = test_x.unsqueeze(0).to(device)
        recon_x_tensor, _, _ = model(test_x_tensor)
        
        real_cst = test_x.cpu().numpy()
        recon_cst = recon_x_tensor.squeeze(0).cpu().numpy()
        
    cst = CSTParametrization(n_coefs_upper=6, n_coefs_lower=6, te_thickness=0.003)
    x_real, y_real_up, y_real_low = cst.generate_coordinates(real_cst[:6], real_cst[6:], n_points=100)
    x_rec, y_rec_up, y_rec_low = cst.generate_coordinates(recon_cst[:6], recon_cst[6:], n_points=100)
    
    plt.figure(figsize=(10, 4))
    plt.plot(x_real, y_real_up, 'k-', label="Perfil Real (CST)", linewidth=2.5)
    plt.plot(x_real, y_real_low, 'k-', linewidth=2.5)
    plt.plot(x_rec, y_rec_up, 'r--', label="Reconstrucción IA (Geom Loss)", linewidth=1.5)
    plt.plot(x_rec, y_rec_low, 'r--', linewidth=1.5)
    
    plt.axis("equal")
    plt.title("Visualización de Validación: Modelo Base Mejorado (Geom Loss)")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.grid(True)
    plt.legend()
    plt.savefig(recon_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("\n>> Re-entrenamiento del Modelo Base finalizado.")

if __name__ == "__main__":
    main()
