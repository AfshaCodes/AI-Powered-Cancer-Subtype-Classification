import pandas as pd
import os
import requests

print("Downloading REAL TCGA Breast Cancer data from UCSC Xena...")
print("This may take 2-3 minutes...\n")

os.makedirs('data', exist_ok=True)

# UCSC Xena TCGA-BRCA datasets (pre-processed and ready to use)
datasets = {
    'gene_expression': {
        'url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz',
        'filename': 'data/brca_expression.tsv.gz'
    },
    'clinical': {
        'url': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix',
        'filename': 'data/brca_clinical.tsv'
    }
}

def download_file(url, filename):
    """Download a file with progress indication"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='')
        print()  # New line after download
        return True
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        return False

# Download gene expression data
print("1/2 Downloading gene expression data (~50MB)...")
if download_file(datasets['gene_expression']['url'], datasets['gene_expression']['filename']):
    print("✓ Gene expression downloaded")
    
    # Load and preview
    expression_df = pd.read_csv(datasets['gene_expression']['filename'], 
                                sep='\t', compression='gzip', index_col=0)
    print(f"   Shape: {expression_df.shape} (genes × patients)")
    
    # Transpose so rows are patients, columns are genes
    expression_df = expression_df.T
    expression_df.insert(0, 'patient_id', expression_df.index)
    expression_df.to_csv('data/brca_expression.csv', index=False)
    print(f"   Saved as CSV: {expression_df.shape} (patients × genes)")

# Download clinical data
print("\n2/2 Downloading clinical data...")
if download_file(datasets['clinical']['url'], datasets['clinical']['filename']):
    print("✓ Clinical data downloaded")
    
    # Load and preview
    clinical_df = pd.read_csv(datasets['clinical']['filename'], sep='\t')
    print(f"   Shape: {clinical_df.shape}")
    print(f"   Columns: {len(clinical_df.columns)} clinical features")
    
    # Save as CSV
    clinical_df.to_csv('data/brca_clinical.csv', index=False)
    print(f"   Saved as CSV")

print("\n" + "="*60)
print("✅ REAL TCGA DATA DOWNLOADED SUCCESSFULLY!")
print("="*60)
print(f"Location: {os.path.abspath('data')}")
print("\nFiles created:")
print("  - brca_expression.csv (gene expression matrix)")
print("  - brca_clinical.csv (patient clinical data)")