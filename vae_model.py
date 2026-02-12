import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder for gene expression data
    
    Architecture:
    - Encoder: 5000 → 1024 → 512 → latent_dim
    - Decoder: latent_dim → 512 → 1024 → 5000
    """
    
    def __init__(self, input_dim=5000, latent_dim=50):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ===== ENCODER =====
        # Compresses gene expression into latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(512, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(512, latent_dim)  # Log variance
        
        # ===== DECODER =====
        # Reconstructs gene expression from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, input_dim)
        )
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        Returns: mean and log-variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent representation to reconstruct input
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Full forward pass: encode → reparameterize → decode
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + KL Divergence
    
    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (beta-VAE)
    
    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# Test the model
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING VAE MODEL")
    print("=" * 60)
    
    # Create model
    model = VAE(input_dim=5000, latent_dim=50)
    print(f"\n✓ Model created")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Latent dimension: {model.latent_dim}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    dummy_input = torch.randn(32, 5000)  # Batch of 32 patients
    recon, mu, logvar = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Latent mu shape: {mu.shape}")
    print(f"  Latent logvar shape: {logvar.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    
    # Test loss
    print("\n✓ Testing loss function...")
    loss, recon_loss, kl_loss = vae_loss_function(recon, dummy_input, mu, logvar)
    print(f"  Total loss: {loss.item():.2f}")
    print(f"  Reconstruction loss: {recon_loss.item():.2f}")
    print(f"  KL divergence: {kl_loss.item():.2f}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TEST SUCCESSFUL!")
    print("=" * 60)
    print("\nModel architecture:")
    print(model)