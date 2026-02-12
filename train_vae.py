import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Import our VAE model
sys.path.append('models')
from vae_model import VAE, vae_loss_function

print("=" * 70)
print("TRAINING VARIATIONAL AUTOENCODER ON TCGA DATA")
print("=" * 70)

# ===== CONFIGURATION =====
config = {
    'input_dim': 5000,
    'latent_dim': 50,
    'batch_size': 64,
    'num_epochs': 30,
    'learning_rate': 1e-3,
    'beta': 1.0,  # Weight for KL divergence
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("\n📋 Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# ===== LOAD DATA =====
print("\n[1/5] Loading preprocessed data...")
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')

print(f"✓ Training data: {X_train.shape}")
print(f"✓ Test data: {X_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor)
test_dataset = TensorDataset(X_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

print(f"✓ Created {len(train_loader)} training batches")
print(f"✓ Created {len(test_loader)} test batches")

# ===== CREATE MODEL =====
print("\n[2/5] Creating VAE model...")
device = torch.device(config['device'])
model = VAE(input_dim=config['input_dim'], latent_dim=config['latent_dim'])
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created with {total_params:,} parameters")
print(f"✓ Device: {device}")

# ===== TRAINING FUNCTIONS =====
def train_epoch(model, train_loader, optimizer, device, beta):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss_function(
            recon_batch, data, mu, logvar, beta
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
    
    # Average losses
    n_samples = len(train_loader.dataset)
    return (train_loss / n_samples, 
            train_recon_loss / n_samples, 
            train_kl_loss / n_samples)


def test_epoch(model, test_loader, device, beta):
    """Evaluate on test set"""
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data, in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_batch, data, mu, logvar, beta
            )
            
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
    
    n_samples = len(test_loader.dataset)
    return (test_loss / n_samples, 
            test_recon_loss / n_samples, 
            test_kl_loss / n_samples)


# ===== TRAINING LOOP =====
print("\n[3/5] Training VAE...")
print(f"Training for {config['num_epochs']} epochs...")
print("-" * 70)

# Track losses
history = {
    'train_loss': [],
    'train_recon': [],
    'train_kl': [],
    'test_loss': [],
    'test_recon': [],
    'test_kl': []
}

start_time = time.time()
best_test_loss = float('inf')

for epoch in range(1, config['num_epochs'] + 1):
    epoch_start = time.time()
    
    # Train
    train_loss, train_recon, train_kl = train_epoch(
        model, train_loader, optimizer, device, config['beta']
    )
    
    # Test
    test_loss, test_recon, test_kl = test_epoch(
        model, test_loader, device, config['beta']
    )
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_recon'].append(train_recon)
    history['train_kl'].append(train_kl)
    history['test_loss'].append(test_loss)
    history['test_recon'].append(test_recon)
    history['test_kl'].append(test_kl)
    
    # Print progress
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch:2d}/{config['num_epochs']} | "
          f"Train Loss: {train_loss:.2f} | "
          f"Test Loss: {test_loss:.2f} | "
          f"Time: {epoch_time:.1f}s")
    
    # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'models/vae_best.pth')
        print(f"  ✓ Best model saved (test loss: {test_loss:.2f})")

total_time = time.time() - start_time
print("-" * 70)
print(f"✅ Training complete! Total time: {total_time/60:.1f} minutes")

# ===== SAVE FINAL MODEL =====
print("\n[4/5] Saving model...")
torch.save(model.state_dict(), 'models/vae_final.pth')
torch.save({
    'config': config,
    'history': history,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'models/vae_checkpoint.pth')

print("✓ Saved models/vae_final.pth")
print("✓ Saved models/vae_checkpoint.pth")

# ===== PLOT TRAINING CURVES =====
print("\n[5/5] Creating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total loss
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['test_loss'], label='Test', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].set_title('Total VAE Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Reconstruction loss
axes[0, 1].plot(history['train_recon'], label='Train', linewidth=2, color='green')
axes[0, 1].plot(history['test_recon'], label='Test', linewidth=2, color='lightgreen')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Reconstruction Loss')
axes[0, 1].set_title('Reconstruction Loss (MSE)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# KL divergence
axes[1, 0].plot(history['train_kl'], label='Train', linewidth=2, color='red')
axes[1, 0].plot(history['test_kl'], label='Test', linewidth=2, color='salmon')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('KL Divergence')
axes[1, 0].set_title('KL Divergence')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Loss ratio
axes[1, 1].plot(np.array(history['train_recon']) / np.array(history['train_kl']), 
                label='Train Recon/KL', linewidth=2, color='purple')
axes[1, 1].plot(np.array(history['test_recon']) / np.array(history['test_kl']), 
                label='Test Recon/KL', linewidth=2, color='violet')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Ratio')
axes[1, 1].set_title('Reconstruction / KL Ratio')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved: results/training_curves.png")
plt.close()

# ===== FINAL SUMMARY =====
print("\n" + "=" * 70)
print("✅ TRAINING SUMMARY")
print("=" * 70)
print(f"Final Training Loss: {history['train_loss'][-1]:.2f}")
print(f"Final Test Loss: {history['test_loss'][-1]:.2f}")
print(f"Best Test Loss: {best_test_loss:.2f}")
print(f"Total Training Time: {total_time/60:.1f} minutes")
print("\n📁 Files created:")
print("  - models/vae_best.pth (best model)")
print("  - models/vae_final.pth (final model)")
print("  - models/vae_checkpoint.pth (full checkpoint)")
print("  - results/training_curves.png (loss plots)")
print("\n🎯 Next step: Extract latent representations and cluster!")