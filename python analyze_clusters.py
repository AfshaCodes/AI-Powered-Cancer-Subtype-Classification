import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CLINICAL ANALYSIS OF CANCER SUBTYPES")
print("=" * 70)

# ===== LOAD DATA =====
print("\n[1/5] Loading data...")

clusters_df = pd.read_csv('results/patient_clusters.csv')
clinical_df = pd.read_csv('data/brca_clinical.csv')
processed_df = pd.read_csv('data/processed_data.csv')

print(f"✓ Clusters: {clusters_df.shape}")
print(f"✓ Clinical: {clinical_df.shape}")
print(f"✓ Processed: {processed_df.shape}")

# ===== MERGE DATA =====
print("\n[2/5] Merging cluster assignments with clinical data...")

# Merge clusters with processed data (which has clinical features)
analysis_df = processed_df.merge(clusters_df, on='patient_id', how='left')

print(f"✓ Merged dataset: {analysis_df.shape}")
print(f"✓ Patients with cluster assignment: {analysis_df['cluster'].notna().sum()}")

# ===== CLUSTER STATISTICS =====
print("\n[3/5] Analyzing clinical differences between clusters...")

# Define clinical features to analyze
clinical_features = ['age', 'vital_status', 'days_to_death', 'days_to_last_followup']
available_features = [col for col in clinical_features if col in analysis_df.columns]

print(f"\nAvailable clinical features: {available_features}")

# ===== AGE ANALYSIS =====
if 'age' in analysis_df.columns:
    print("\n" + "=" * 70)
    print("AGE ANALYSIS")
    print("=" * 70)
    
    age_by_cluster = analysis_df.groupby('cluster')['age'].agg(['mean', 'std', 'count'])
    print(age_by_cluster)
    
    # Statistical test (ANOVA)
    clusters = analysis_df['cluster'].unique()
    age_groups = [analysis_df[analysis_df['cluster'] == c]['age'].dropna() for c in clusters]
    f_stat, p_value = stats.f_oneway(*age_groups)
    print(f"\nANOVA: F={f_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("✓ Age differs significantly between clusters (p < 0.05)")
    else:
        print("✗ No significant age difference between clusters")

# ===== SURVIVAL ANALYSIS =====
if 'vital_status' in analysis_df.columns:
    print("\n" + "=" * 70)
    print("SURVIVAL ANALYSIS")
    print("=" * 70)
    
    # Count vital status by cluster
    survival_counts = pd.crosstab(analysis_df['cluster'], analysis_df['vital_status'])
    print("\nVital Status by Cluster:")
    print(survival_counts)
    
    # Calculate survival rates
    survival_rates = pd.crosstab(analysis_df['cluster'], analysis_df['vital_status'], 
                                  normalize='index') * 100
    print("\nSurvival Rates (%):")
    print(survival_rates)
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(survival_counts)
    print(f"\nChi-square test: χ²={chi2:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("✓ Survival status differs significantly between clusters (p < 0.05)")

# ===== VISUALIZATIONS =====
print("\n[4/5] Creating visualizations...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Cluster sizes
ax1 = fig.add_subplot(gs[0, 0])
cluster_counts = analysis_df['cluster'].value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax1.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
ax1.set_title('Cluster Sizes', fontsize=12, fontweight='bold')
for i, v in enumerate(cluster_counts.values):
    ax1.text(i, v + 20, str(v), ha='center', fontweight='bold')

# 2. Age distribution by cluster
if 'age' in analysis_df.columns:
    ax2 = fig.add_subplot(gs[0, 1])
    age_data = [analysis_df[analysis_df['cluster'] == i]['age'].dropna() 
                for i in sorted(analysis_df['cluster'].unique())]
    bp = ax2.boxplot(age_data, labels=[f'Cluster {i}' for i in range(len(age_data))],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Age at Diagnosis', fontsize=11, fontweight='bold')
    ax2.set_title('Age Distribution by Cluster', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

# 3. Survival status by cluster
if 'vital_status' in analysis_df.columns:
    ax3 = fig.add_subplot(gs[0, 2])
    survival_pivot = pd.crosstab(analysis_df['cluster'], analysis_df['vital_status'])
    survival_pivot.plot(kind='bar', stacked=True, ax=ax3, 
                        color=['#2ECC71', '#E74C3C'], edgecolor='black', linewidth=1)
    ax3.set_xlabel('Cluster', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
    ax3.set_title('Vital Status by Cluster', fontsize=12, fontweight='bold')
    ax3.legend(title='Status', frameon=True)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

# 4. Survival rates (percentage)
if 'vital_status' in analysis_df.columns:
    ax4 = fig.add_subplot(gs[1, 0])
    survival_pct = pd.crosstab(analysis_df['cluster'], analysis_df['vital_status'], 
                               normalize='index') * 100
    survival_pct.plot(kind='bar', ax=ax4, color=['#2ECC71', '#E74C3C'], 
                      edgecolor='black', linewidth=1)
    ax4.set_xlabel('Cluster', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Survival Rates by Cluster', fontsize=12, fontweight='bold')
    ax4.legend(title='Status', frameon=True)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.set_ylim([0, 100])

# 5. Kaplan-Meier Survival Curves
if 'days_to_death' in analysis_df.columns and 'days_to_last_followup' in analysis_df.columns:
    ax5 = fig.add_subplot(gs[1, 1:])
    
    # Prepare survival data
    analysis_df['event'] = (analysis_df['vital_status'] == 'DECEASED').astype(int)
    analysis_df['time'] = analysis_df.apply(
        lambda row: row['days_to_death'] if pd.notna(row['days_to_death']) 
        else row['days_to_last_followup'], axis=1
    )
    
    # Remove invalid times
    survival_data = analysis_df[analysis_df['time'] > 0].copy()
    
    # Plot KM curves
    kmf = KaplanMeierFitter()
    for cluster in sorted(survival_data['cluster'].unique()):
        cluster_data = survival_data[survival_data['cluster'] == cluster]
        kmf.fit(cluster_data['time'], cluster_data['event'], 
                label=f'Cluster {cluster} (n={len(cluster_data)})')
        kmf.plot_survival_function(ax=ax5, linewidth=2.5, color=colors[int(cluster)])
    
    ax5.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Survival Probability', fontsize=11, fontweight='bold')
    ax5.set_title('Kaplan-Meier Survival Curves by Cluster', fontsize=12, fontweight='bold')
    ax5.legend(frameon=True, fontsize=10)
    ax5.grid(alpha=0.3)
    
    # Log-rank test
    try:
        results = multivariate_logrank_test(
            survival_data['time'], 
            survival_data['cluster'], 
            survival_data['event']
        )
        p_val = results.p_value
        ax5.text(0.02, 0.02, f'Log-rank test: p={p_val:.4f}', 
                transform=ax5.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        print("\nKaplan-Meier Analysis:")
        print(f"  Log-rank test p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("  ✓ Survival differs significantly between clusters (p < 0.05)")
    except:
        print("\n  Note: Could not perform log-rank test")

# 6. Summary statistics table
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('tight')
ax6.axis('off')

summary_data = []
for cluster in sorted(analysis_df['cluster'].unique()):
    cluster_data = analysis_df[analysis_df['cluster'] == cluster]
    n = len(cluster_data)
    age_mean = cluster_data['age'].mean() if 'age' in cluster_data else np.nan
    deceased = (cluster_data['vital_status'] == 'DECEASED').sum() if 'vital_status' in cluster_data else 0
    alive = (cluster_data['vital_status'] == 'LIVING').sum() if 'vital_status' in cluster_data else 0
    
    summary_data.append([
        f'Cluster {int(cluster)}',
        n,
        f'{age_mean:.1f}' if not np.isnan(age_mean) else 'N/A',
        alive,
        deceased,
        f'{deceased/n*100:.1f}%' if n > 0 else 'N/A'
    ])

table = ax6.table(cellText=summary_data,
                  colLabels=['Cluster', 'N Patients', 'Mean Age', 'Alive', 'Deceased', 'Mortality'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#3498DB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(summary_data) + 1):
    for j in range(6):
        table[(i, j)].set_facecolor(colors[i-1])
        table[(i, j)].set_alpha(0.3)

ax6.set_title('Cluster Summary Statistics', fontsize=14, fontweight='bold', pad=20)

plt.savefig('results/clinical_analysis.png', dpi=200, bbox_inches='tight')
print("✓ Plot saved: results/clinical_analysis.png")
plt.close()

# ===== EXPORT DETAILED RESULTS =====
print("\n[5/5] Exporting detailed results...")

# Create summary report
with open('results/cluster_analysis_report.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("CANCER SUBTYPE ANALYSIS REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Total Patients: {len(analysis_df)}\n")
    f.write(f"Number of Clusters: {analysis_df['cluster'].nunique()}\n\n")
    
    for cluster in sorted(analysis_df['cluster'].unique()):
        cluster_data = analysis_df[analysis_df['cluster'] == cluster]
        f.write(f"\n--- Cluster {int(cluster)} ---\n")
        f.write(f"N = {len(cluster_data)} patients ({len(cluster_data)/len(analysis_df)*100:.1f}%)\n")
        
        if 'age' in cluster_data.columns:
            f.write(f"Age: {cluster_data['age'].mean():.1f} ± {cluster_data['age'].std():.1f} years\n")
        
        if 'vital_status' in cluster_data.columns:
            deceased = (cluster_data['vital_status'] == 'DECEASED').sum()
            f.write(f"Deceased: {deceased} ({deceased/len(cluster_data)*100:.1f}%)\n")

print("✓ Report saved: results/cluster_analysis_report.txt")

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("✅ CLINICAL ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\n📁 Files created:")
print("  - results/clinical_analysis.png (comprehensive visualization)")
print("  - results/cluster_analysis_report.txt (text summary)")
print("\n🎯 Key Findings:")
print(f"  - Identified {analysis_df['cluster'].nunique()} distinct subtypes")
print(f"  - Cluster sizes: {dict(cluster_counts)}")
print("\n💡 Interpretation:")
print("  These clusters represent different molecular subtypes of breast cancer")
print("  that may respond differently to treatment!")