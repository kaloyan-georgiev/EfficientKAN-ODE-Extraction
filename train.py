import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

from models import MLP 
from models import KAN
from dataset_generation import get_dataloaders

def check_architecture_parity(mlp: nn.Module, kan: nn.Module) -> None:
    n_mlp = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    n_kan = sum(p.numel() for p in kan.parameters() if p.requires_grad)
    
    diff_percent = abs(n_mlp - n_kan) / max(n_mlp, 1)
    print(f"MLP Parameters: {n_mlp:,}")
    print(f"External KAN Parameters: {n_kan:,}")
    print(f"Mismatch %:     {diff_percent * 100:.4f}%\n")
    
    # Assert marginal threshold < 0.5% (0.005)
    assert diff_percent < 0.005, f"Iso-Parameter mapping failed: {diff_percent*100:.3f}% > 0.5%"

def train_models() -> None:
    mlp = MLP()
    
    # We adjust hidden_dim to 643 to match exactly ~50,166 parameters. 
    # efficient_kan has a 13 parameters per edge
    kan = KAN(layers_hidden=[3, 643, 3], grid_size=8, grid_range=[-3.0, 3.0])
    
    check_architecture_parity(mlp, kan)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print(f"Targeting Device: {device}")
    
    mlp.to(device)
    kan.to(device)
    
    train_loader, val_loader = get_dataloaders(batch_size=256)
    
    optimizer_mlp = optim.AdamW(mlp.parameters(), lr=1e-3)
    optimizer_kan = optim.AdamW(kan.parameters(), lr=1e-3)
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    epochs = 50
    lambda_l1 = 1e-4

    for epoch in range(1, epochs + 1):
        mlp.train()
        kan.train()
        
        train_mse_mlp = 0.0
        train_mse_kan = 0.0
        
        for X_t, X_next in train_loader:
            X_t, X_next = X_t.to(device), X_next.to(device)
            
            optimizer_mlp.zero_grad()
            pred_mlp = mlp(X_t)
            loss_mlp = criterion_mse(pred_mlp, X_next)
            loss_mlp.backward()
            optimizer_mlp.step()
            train_mse_mlp += loss_mlp.item() * X_t.size(0)
            
            optimizer_kan.zero_grad()
            pred_kan = kan(X_t)
            mse_kan = criterion_mse(pred_kan, X_next)
            
            # L1 Structural Penalty Extraction (using built-in regularization_loss from kan.py)
            l1_penalty = kan.regularization_loss(regularize_activation=1.0, regularize_entropy=0.0)
            loss_kan = mse_kan + lambda_l1 * l1_penalty
            
            loss_kan.backward()
            optimizer_kan.step()
            train_mse_kan += mse_kan.item() * X_t.size(0)
            
        train_mse_mlp /= len(train_loader.dataset)
        train_mse_kan /= len(train_loader.dataset)
        
        mlp.eval()
        kan.eval()
        val_mse_mlp, val_mae_mlp = 0.0, 0.0
        val_mse_kan, val_mae_kan = 0.0, 0.0
        
        with torch.no_grad():
            for X_t, X_next in val_loader:
                X_t, X_next = X_t.to(device), X_next.to(device)
                
                pred_mlp = mlp(X_t)
                val_mse_mlp += criterion_mse(pred_mlp, X_next).item() * X_t.size(0)
                val_mae_mlp += criterion_mae(pred_mlp, X_next).item() * X_t.size(0)
                
                pred_kan = kan(X_t)
                val_mse_kan += criterion_mse(pred_kan, X_next).item() * X_t.size(0)
                val_mae_kan += criterion_mae(pred_kan, X_next).item() * X_t.size(0)
                
        val_mse_mlp /= len(val_loader.dataset)
        val_mae_mlp /= len(val_loader.dataset)
        val_mse_kan /= len(val_loader.dataset)
        val_mae_kan /= len(val_loader.dataset)
        
        print(f"Epoch {epoch:03d}/{epochs} | \n"
              f"MLP (Train-MSE: {train_mse_mlp:.4e}, Eval-MSE: {val_mse_mlp:.4e}, Eval-MAE: {val_mae_mlp:.4e}) \n "
              f"KAN (Train-MSE: {train_mse_kan:.4e}, Eval-MSE: {val_mse_kan:.4e}, Eval-MAE: {val_mae_kan:.4e})")

    print("\nSaving trained KAN model to 'external_kan_model.pth'...")
    torch.save(kan.state_dict(), "kan_model.pth")
    print("Training complete! You can now adapt Phase II to load 'external_kan_model.pth'.")

if __name__ == "__main__":
    train_models()
