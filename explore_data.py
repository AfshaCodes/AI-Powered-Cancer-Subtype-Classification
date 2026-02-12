import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("=" * 70)
print("EXPLORING REAL TCGA BREAST CANCER DATA")
print("=" * 70)

# Load data
print("\nLoading data...")
expression_df = pd.read_csv('data/brca_expression.csv')
clinical_df = pd.read_csv('data/brca_clinical.csv')

# ===== GENE EXPRESSION DATA =====
print("\n" + "=" * 70)
print("📊 GENE EXPRESSION DATA")
print("=" * 70)
print(f"Shape: {expression_df.shape}")
print(f"Patients: {expression_df.shape[0]:,}")
print(f"Genes: {expression_df.shape[1] - 1:,}")  # Minus patient_id column

print("\nFirst 3 patients, first 5 genes:")
print(expression_df.iloc[:3, :6])

print("\nExpression value statistics:")
# Get numeric columns only (exclude patient_id)
numeric_cols = expression_df.select_dtypes(include=[np.number]).columns
print(expression_df[numeric_cols].describe().iloc[:, :5])

# ===== CLINICAL DATA =====
print("\n" + "=" * 70)
print("📋 CLINICAL DATA")
print("=" * 70)
print(f"Shape: {clinical_df.shape}")
print(f"Patients: {clinical_df.shape[0]:,}")
print(f"Clinical features: {clinical_df.shape[1]:,}")

print("\nAvailable clinical columns:")
print(clinical_df.columns.tolist()[:20])  # Show first 20 columns
print(f"... and {len(clinical_df.columns) - 20} more columns")

print("\nFirst few patients:")
print(clinical_df.iloc[:3, :8])

# ===== KEY CLINICAL FEATURES =====
print("\n" + "=" * 70)
print("🔍 KEY CLINICAL FEATURES")
print("=" * 70)

# Check what important columns we have
important_cols = ['age_at_initial_pathologic_diagnosis', 'vital_status', 
                  'days_to_death', 'days_to_last_followup', 
                  'pathologic_stage', 'tumor_status']

available_cols = [col for col in important_cols if col in clinical_df.columns]
print(f"Found {len(available_cols)} key features: {available_cols}")

# Show statistics for available columns
if available_cols:
    print("\nClinical statistics:")
    for col in available_cols:
        if clinical_df[col].dtype == 'object':
            print(f"\n{col}:")
            print(clinical_df[col].value_counts().head())
        else:
            print(f"\n{col}: mean={clinical_df[col].mean():.1f}, median={clinical_df[col].median():.1f}")

# ===== VISUALIZATIONS =====
print("\n" + "=" * 70)
print("📈 CREATING VISUALIZATIONS")
print("=" * 70)

os.makedirs('results', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Age distribution (if available)
if 'age_at_initial_pathologic_diagnosis' in clinical_df.columns:
    age_data = pd.to_numeric(clinical_df['age_at_initial_pathologic_diagnosis'], errors='coerce')
    age_data = age_data.dropna()
    axes[0, 0].hist(age_data, bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Age at Diagnosis')
    axes[0, 0].set_ylabel('Number of Patients')
    axes[0, 0].set_title(f'Age Distribution (n={len(age_data)})')
    axes[0, 0].axvline(age_data.median(), color='red', linestyle='--', label=f'Median: {age_data.median():.0f}')
    axes[0, 0].legend()

# 2. Vital status
if 'vital_status' in clinical_df.columns:
    vital_counts = clinical_df['vital_status'].value_counts()
    axes[0, 1].bar(vital_counts.index, vital_counts.values, color=['green', 'red'])
    axes[0, 1].set_xlabel('Vital Status')
    axes[0, 1].set_ylabel('Number of Patients')
    axes[0, 1].set_title('Patient Vital Status')
    for i, v in enumerate(vital_counts.values):
        axes[0, 1].text(i, v + 10, str(v), ha='center', fontweight='bold')

# 3. Pathologic stage (if available)
if 'pathologic_stage' in clinical_df.columns:
    stage_data = clinical_df['pathologic_stage'].value_counts().head(10)
    axes[1, 0].barh(range(len(stage_data)), stage_data.values, color='coral')
    axes[1, 0].set_yticks(range(len(stage_data)))
    axes[1, 0].set_yticklabels(stage_data.index)
    axes[1, 0].set_xlabel('Number of Patients')
    axes[1, 0].set_title('Tumor Stage Distribution')
    axes[1, 0].invert_yaxis()

# 4. Gene expression distribution (sample)
sample_gene = expression_df.columns[1]  # First gene after patient_id
gene_values = expression_df[sample_gene].dropna()
axes[1, 1].hist(gene_values, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Expression Level (log2)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Sample Gene Expression\n({sample_gene})')

plt.tight_layout()
plt.savefig('results/tcga_data_exploration.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved: results/tcga_data_exploration.png")
plt.close()

# ===== DATA QUALITY CHECK =====
print("\n" + "=" * 70)
print("🔍 DATA QUALITY CHECK")
print("=" * 70)

# Check for missing values in expression data
missing_expr = expression_df.isnull().sum().sum()
total_expr = expression_df.shape[0] * expression_df.shape[1]
print(f"Expression data missing values: {missing_expr:,} / {total_expr:,} ({missing_expr/total_expr*100:.2f}%)")

# Check for missing values in clinical data
missing_clin = clinical_df.isnull().sum()
print(f"\nClinical data missing values by column (top 10):")
print(missing_clin.sort_values(ascending=False).head(10))

print("\n" + "=" * 70)
print("✅ DATA EXPLORATION COMPLETE!")
print("=" * 70)
print("\nNext step: Data cleaning and normalization")