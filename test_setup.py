import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

print("=" * 50)
print("SETUP VERIFICATION")
print("=" * 50)
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print("✓ All libraries imported successfully!")
print("=" * 50)

x = torch.randn(10, 5)
print(f"✓ Created test tensor with shape: {x.shape}")
print("\n🎉 Setup complete! Ready for Step 2.")


