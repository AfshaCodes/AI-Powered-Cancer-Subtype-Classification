from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import plotly.utils
import sys
import os
from io import BytesIO
import base64

# Import VAE model
sys.path.append('models')
from vae_model import VAE

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ===== LOAD MODELS AND DATA =====
print("Loading models and data...")

# Load VAE model
device = torch.device('cpu')
vae_model = VAE(input_dim=5000, latent_dim=50)
vae_model.load_state_dict(torch.load('models/vae_best.pth', map_location=device))
vae_model.eval()

# Load scaler
with open('models/expression_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load selected genes
selected_genes = pd.read_csv('data/selected_genes.csv')['gene'].tolist()

# Load cluster data for statistics
cluster_stats = pd.read_csv('results/patient_clusters.csv')
clinical_data = pd.read_csv('data/processed_data.csv')
merged_stats = clinical_data.merge(cluster_stats, on='patient_id', how='inner')

# Calculate cluster statistics
cluster_info = {}
for cluster in sorted(merged_stats['cluster'].unique()):
    cluster_data = merged_stats[merged_stats['cluster'] == cluster]
    n = len(cluster_data)
    deceased = (cluster_data['vital_status'] == 'DECEASED').sum()
    mortality = (deceased / n * 100) if n > 0 else 0
    
    if mortality < 27:
        risk = "Good Prognosis"
        color = "#2ECC71"
        emoji = "⭐"
    elif mortality < 35:
        risk = "Intermediate Risk"
        color = "#F39C12"
        emoji = "⚠️"
    else:
        risk = "Poor Prognosis"
        color = "#E74C3C"
        emoji = "🔴"
    
    cluster_info[int(cluster)] = {
        'name': f'Cluster {int(cluster)}',
        'n_patients': n,
        'mortality': mortality,
        'risk_level': risk,
        'color': color,
        'emoji': emoji,
        'mean_age': cluster_data['age'].mean() if 'age' in cluster_data else None
    }

print("✓ Models and data loaded successfully!")

# ===== HELPER FUNCTIONS =====

def predict_cluster(gene_expression_data):
    """
    Predict cancer subtype cluster for given gene expression data
    
    Args:
        gene_expression_data: DataFrame with genes as columns
    
    Returns:
        cluster_id, latent_features, confidence
    """
    # Ensure we have the right genes
    missing_genes = set(selected_genes) - set(gene_expression_data.columns)
    if missing_genes:
        # Fill missing genes with zeros
        for gene in missing_genes:
            gene_expression_data[gene] = 0
    
    # Select only the genes we need, in the right order
    X = gene_expression_data[selected_genes].values
    
    # Normalize
    X_normalized = scaler.transform(X)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_normalized).to(device)
    
    # Get latent representation
    with torch.no_grad():
        mu, logvar = vae_model.encode(X_tensor)
        latent_features = mu.cpu().numpy()
    
    # Load cluster centers (K-Means)
    # For now, we'll use a simple distance-based approach
    # In production, you'd save the KMeans model
    from sklearn.cluster import KMeans
    
    # Load pre-computed latent features and clusters
    all_latent = np.load('data/latent_features.npy')
    all_clusters = cluster_stats['cluster'].values
    
    # Fit KMeans on existing data
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    kmeans.fit(all_latent)
    
    # Predict cluster
    cluster_id = kmeans.predict(latent_features)[0]
    
    # Calculate confidence (distance to cluster center)
    distances = kmeans.transform(latent_features)[0]
    min_distance = distances[cluster_id]
    confidence = 1 / (1 + min_distance)  # Simple confidence metric
    
    return int(cluster_id), latent_features[0], float(confidence)


def create_result_plots(cluster_id, latent_features, confidence):
    """Create interactive Plotly visualizations"""
    
    plots = {}
    
    # 1. Cluster assignment gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Prediction Confidence<br>{cluster_info[cluster_id]['emoji']} {cluster_info[cluster_id]['risk_level']}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': cluster_info[cluster_id]['color']},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=300, paper_bgcolor="white")
    plots['gauge'] = json.dumps(fig_gauge, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. Cluster comparison
    cluster_ids = list(cluster_info.keys())
    mortality_rates = [cluster_info[c]['mortality'] for c in cluster_ids]
    cluster_names = [cluster_info[c]['risk_level'] for c in cluster_ids]
    colors_list = [cluster_info[c]['color'] for c in cluster_ids]
    
    fig_comparison = go.Figure(data=[
        go.Bar(
            x=cluster_names,
            y=mortality_rates,
            marker_color=colors_list,
            text=[f"{m:.1f}%" for m in mortality_rates],
            textposition='auto',
        )
    ])
    fig_comparison.update_layout(
        title="Mortality Rates by Cancer Subtype",
        xaxis_title="Cancer Subtype",
        yaxis_title="Mortality Rate (%)",
        yaxis=dict(range=[0, 50]),
        height=350,
        paper_bgcolor="white"
    )
    # Highlight the predicted cluster
    fig_comparison.add_annotation(
        x=cluster_names[cluster_id],
        y=mortality_rates[cluster_id] + 3,
        text="← Your Prediction",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red"
    )
    plots['comparison'] = json.dumps(fig_comparison, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Patient distribution
    patient_counts = [cluster_info[c]['n_patients'] for c in cluster_ids]
    fig_distribution = go.Figure(data=[
        go.Pie(
            labels=cluster_names,
            values=patient_counts,
            marker=dict(colors=colors_list),
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    fig_distribution.update_layout(
        title="Patient Distribution Across Subtypes",
        height=350,
        paper_bgcolor="white"
    )
    plots['distribution'] = json.dumps(fig_distribution, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plots


# ===== ROUTES =====

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', cluster_info=cluster_info)


@app.route('/about')
def about():
    """About page"""
    stats = {
        'total_patients': len(cluster_stats),
        'n_genes': len(selected_genes),
        'n_clusters': len(cluster_info),
        'model_params': '11.4M',
        'training_time': '1.6 minutes'
    }
    return render_template('about.html', stats=stats, cluster_info=cluster_info)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file (CSV or TSV)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.tsv') or file.filename.endswith('.txt'):
            df = pd.read_csv(file, sep='\t')
        else:
            return jsonify({'error': 'File must be CSV or TSV'}), 400
        
        # Make prediction
        cluster_id, latent_features, confidence = predict_cluster(df)
        
        # Get cluster info
        info = cluster_info[cluster_id]
        
        # Create plots
        plots = create_result_plots(cluster_id, latent_features, confidence)
        
        # Prepare result
        result = {
            'cluster_id': cluster_id,
            'cluster_name': info['name'],
            'risk_level': info['risk_level'],
            'emoji': info['emoji'],
            'mortality_rate': f"{info['mortality']:.1f}%",
            'confidence': f"{confidence * 100:.1f}%",
            'n_similar_patients': info['n_patients'],
            'color': info['color'],
            'plots': plots
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/demo')
def demo():
    """Demo prediction with sample data"""
    
    try:
        # Load a random sample patient
        processed_data = pd.read_csv('data/processed_data.csv')
        sample = processed_data.sample(n=1)
        
        # Extract gene expression
        gene_data = sample[selected_genes]
        
        # Make prediction
        cluster_id, latent_features, confidence = predict_cluster(gene_data)
        
        # Get cluster info
        info = cluster_info[cluster_id]
        
        # Create plots
        plots = create_result_plots(cluster_id, latent_features, confidence)
        
        result = {
            'cluster_id': cluster_id,
            'cluster_name': info['name'],
            'risk_level': info['risk_level'],
            'emoji': info['emoji'],
            'mortality_rate': f"{info['mortality']:.1f}%",
            'confidence': f"{confidence * 100:.1f}%",
            'n_similar_patients': info['n_patients'],
            'color': info['color'],
            'plots': plots
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 CANCER SUBTYPE PREDICTION WEB APP")
    print("=" * 70)
    print("\n✓ Server starting...")
    print("✓ Open your browser and go to: http://127.0.0.1:5000")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)