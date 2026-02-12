# 🧬 AI-Powered Cancer Subtype Predictor

> Deep learning-based breast cancer stratification using Variational Autoencoders trained on TCGA genomics data
>https://github.com/AfshaCodes/AI-Powered-Cancer-Subtype-Classification/blob/main
> https://github.com/AfshaCodes/AI-Powered-Cancer-Subtype-Classification/blob/main
> https://github.com/AfshaCodes/AI-Powered-Cancer-Subtype-Classification/blob/main/3.PNG/2.PNG/1.PNG
> 




![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Results](#-results)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Web Interface](#-web-interface)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements an **end-to-end AI system** for classifying breast cancer patients into molecular subtypes based on gene expression data. Using a **Variational Autoencoder (VAE)** trained on **1,492 real patients** from The Cancer Genome Atlas (TCGA), the system:

- Compresses **5,000 genes** into a **50-dimensional latent space**
- Identifies **3 clinically distinct subtypes** with different survival outcomes
- Provides a **web interface** for real-time predictions
- Achieves **statistical significance** in clinical validation (p < 0.001)

### Why This Matters

Cancer is not one disease—it's many. Patients with the same cancer type can have vastly different outcomes. This system enables **precision medicine** by:
- Predicting **treatment response**
- Estimating **survival probability**
- Guiding **therapy selection**

---

## ✨ Key Features

### 🤖 Deep Learning Model
- **Architecture:** Variational Autoencoder (VAE)
- **Parameters:** 11.4 million trainable parameters
- **Training:** 30 epochs in 1.6 minutes on CPU
- **Input:** 5,000 gene expression values
- **Output:** 50-dimensional latent representation

### 📊 Clinical Validation
- **Dataset:** TCGA-BRCA (Breast Cancer)
- **Patients:** 1,492 samples analyzed
- **Subtypes:** 3 distinct clusters identified
- **Validation:** Kaplan-Meier survival analysis
- **Significance:** p < 0.0001 (highly significant)

### 🌐 Web Application
- **Framework:** Flask with responsive HTML/CSS
- **Visualization:** Interactive Plotly charts
- **Features:**
  - Drag-and-drop file upload
  - Real-time prediction
  - Clinical interpretation
  - Demo mode with sample data

---

## 📈 Results

### Three Cancer Subtypes Discovered

| Cluster | Patients | Mortality Rate | Risk Level | Clinical Significance |
|---------|----------|----------------|------------|----------------------|
| **0** | 412 (27.6%) | 30.4% | Intermediate Risk | Moderate outcomes |
| **1** | 1,645 (52.1%) | **25.3%** | ⭐ **Good Prognosis** | **Best survival** |
| **2** | 1,097 (34.8%) | **41.0%** | ⚠️ **Poor Prognosis** | **Needs aggressive treatment** |

### Statistical Validation

✅ **Age difference:** p = 0.0031  
✅ **Survival status:** p < 0.0001  
✅ **Kaplan-Meier (Log-rank):** p < 0.0001  

**Key Finding:** **15.7% absolute difference** in mortality between best and worst subtypes!

---

## 🛠️ Technology Stack

### Backend
- **Python 3.13**
- **PyTorch 2.10.0** - Deep learning framework
- **Flask 3.0** - Web framework
- **NumPy & Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities

### Data Science
- **UMAP** - Dimensionality reduction
- **Lifelines** - Survival analysis
- **Matplotlib & Seaborn** - Visualization

### Frontend
- **HTML5/CSS3/JavaScript**
- **Plotly.js** - Interactive charts
- **Responsive design** - Mobile-friendly

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- 16GB RAM recommended
- ~5GB disk space for data

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/cancer-stratification.git
cd cancer-stratification
```

### Step 2: Install Dependencies

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

**For GPU support:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Download TCGA Data

```bash
python download_data.py
```

This will download:
- Gene expression data (~100MB)
- Clinical data (~10MB)

---

## 💻 Usage

### Full Pipeline (From Scratch)

```bash
# 1. Preprocess data
python preprocess_data.py

# 2. Train VAE model
python train_vae.py

# 3. Perform clustering
python cluster_patients.py

# 4. Analyze clinical outcomes
python analyze_clusters.py

# 5. Generate final summary
python final_summary.py
```

### Quick Start (Web App Only)

If you already have trained models:

```bash
python app.py
```

Open browser to: **http://127.0.0.1:5000**

### Making Predictions

**Option 1: Web Interface**
1. Go to http://127.0.0.1:5000
2. Click "Run Demo Prediction" or upload CSV file
3. View results with interactive charts

**Option 2: Python API**
```python
from app import predict_cluster
import pandas as pd

# Load your gene expression data
data = pd.read_csv('your_patient_data.csv')

# Make prediction
cluster_id, latent_features, confidence = predict_cluster(data)

print(f"Predicted Cluster: {cluster_id}")
print(f"Confidence: {confidence:.2%}")
```

---

## 📁 Project Structure

```
cancer-stratification/
├── app.py                      # Flask web application
├── download_data.py            # TCGA data downloader
├── preprocess_data.py          # Data preprocessing
├── train_vae.py                # VAE model training
├── cluster_patients.py         # K-Means clustering
├── analyze_clusters.py         # Clinical analysis
├── final_summary.py            # Generate summary report
│
├── models/
│   ├── vae_model.py           # VAE architecture
│   ├── vae_best.pth           # Trained weights (download separately)
│   └── expression_scaler.pkl  # Data normalizer
│
├── data/
│   ├── selected_genes.csv     # List of 5000 genes
│   └── README.md              # Data download instructions
│
├── templates/
│   ├── index.html             # Home page
│   └── about.html             # About page
│
├── static/
│   ├── css/style.css          # Stylesheets
│   └── js/script.js           # JavaScript
│
├── results/
│   ├── training_curves.png    # Model training plots
│   ├── umap_visualization.png # Cluster visualization
│   └── clinical_analysis.png  # Clinical outcomes
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md  # Full documentation
│
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- **Source:** TCGA-BRCA cohort (1,492 patients)
- **Feature Selection:** Top 5,000 most variable genes
- **Normalization:** Z-score standardization
- **Split:** 80% train (1,193), 20% test (299)

### 2. VAE Architecture

```
Encoder: 5000 → 1024 → 512 → 50 (latent)
Decoder: 50 → 512 → 1024 → 5000 (reconstruction)

Loss = MSE(reconstruction) + β * KL_Divergence
```

### 3. Clustering
- **Method:** K-Means in latent space
- **Optimal k:** 3 (determined by Silhouette score)
- **Validation:** Clinical outcomes analysis

### 4. Clinical Validation
- **Survival Analysis:** Kaplan-Meier curves
- **Statistical Tests:** Log-rank, ANOVA, Chi-square
- **Result:** Highly significant (p < 0.0001)

---

## 🌐 Web Interface

### Features

**Home Page:**
- Beautiful gradient hero section
- Drag-and-drop file upload
- Demo prediction mode
- Cluster overview cards

**Results Page:**
- Risk classification badge
- Prediction confidence gauge
- Mortality comparison chart
- Patient distribution pie chart
- Automated clinical interpretation

**About Page:**
- Project overview and methodology
- Model architecture details
- Cluster descriptions
- Academic references


## 🔮 Future Work

### Planned Enhancements

**Short-term:**
- [ ] Add DNA methylation data (multi-omics)
- [ ] Implement β-VAE with learnable β
- [ ] Add batch prediction mode
- [ ] Generate PDF reports

**Medium-term:**
- [ ] Extend to other cancer types (lung, colon, etc.)
- [ ] Add explainability (SHAP values, gene importance)
- [ ] Develop treatment recommendation system
- [ ] Create mobile app

**Long-term:**
- [ ] Cloud deployment (AWS/GCP)
- [ ] Clinical trial integration
- [ ] FDA approval pathway
- [ ] Multi-center validation study

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution
- Add more cancer types
- Improve web interface
- Enhance visualizations
- Write tests
- Fix bugs
- Improve documentation

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Data Attribution
- TCGA data is publicly available under NCI GDC Data Use Terms
- Please cite TCGA appropriately in any publications

---



**TCGA Citation:**
```bibtex
@article{tcga2012,
  title={Comprehensive molecular portraits of human breast tumours},
  author={The Cancer Genome Atlas Network},
  journal={Nature},
  volume={490},
  number={7418},
  pages={61--70},
  year={2012}
}
```

---

---

## ⚠️ Disclaimer

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This tool is designed for bioinformatics research and education. It is **NOT approved for clinical diagnosis or treatment decisions**. All medical decisions should be made by qualified healthcare professionals based on comprehensive patient evaluation and established clinical guidelines.

---

## 🙏 Acknowledgments

- **TCGA Research Network** for providing cancer genomics data
- **PyTorch Team** for the deep learning framework
- **Flask Community** for the web framework
- **Plotly** for interactive visualization tools
- **Open-source community** for essential libraries

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

*Last Updated: February 12, 2026*
