"""
Comprehensive Exploratory Data Analysis (EDA) for Healthcare-Diabetes Dataset
All plots are saved in the 'diabetes_eda_plots' directory.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns#
import warnings
from scipy.stats import skew, kurtosis, probplot, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')
# Set up output directory for plots
PLOT_DIR = 'diabetes_eda_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Load the dataset
DATA_PATH = os.path.join('data', 'Healthcare-Diabetes.csv')
df = pd.read_csv(DATA_PATH)

# Drop the 'Id' column as it's just an identifier
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

print("\n[1] DATASET INFO")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# 2. Missing Values and Zero Analysis
print("\n[2] MISSING VALUES AND ZERO ANALYSIS")
missing = df.isnull().sum()
print("Missing values per column:\n", missing[missing > 0])
if missing.sum() == 0:
    print("No missing values found.")

# Identify columns where 0 might indicate missing data
zero_impute_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_counts = (df[zero_impute_cols] == 0).sum()
print("\nColumns with zero values (potential missing data):")
print(zero_counts[zero_counts > 0])

# Visualize distributions before potential imputation
for col in zero_impute_cols:
    if zero_counts[col] > 0:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        # Use a distinct color for original distribution
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Original Distribution of {col}')
        
        # Temporarily replace 0s with NaN for visualization
        temp_col = df[col].replace(0, np.nan)
        plt.subplot(1, 2, 2)
        # Use another color for distribution without zeros
        sns.histplot(temp_col.dropna(), kde=True, bins=30, color='lightcoral')
        plt.title(f'Distribution of {col} (Zeros Excluded)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'eda_dist_{col}_zero_analysis.png'))
        plt.close()

# 3. Descriptive Statistics
print("\n[3] DESCRIPTIVE STATISTICS")
print(df.describe().T)

# 4. Skewness and Kurtosis
print("\n[4] SKEWNESS AND KURTOSIS")
features = [col for col in df.columns if col != 'Outcome']
skewness = df[features].apply(lambda x: skew(x.dropna()))
kurt = df[features].apply(lambda x: kurtosis(x.dropna())) # Fisher's definition (normal=0)
print("Skewness:\n", skewness)
print("\nKurtosis:\n", kurt)

# 5. Target Variable Distribution
print("\n[5] TARGET VARIABLE ANALYSIS")
outcome_counts = df['Outcome'].value_counts()
print(f"Outcome Distribution:\n{outcome_counts}")
print(f"Outcome Proportions:\n{df['Outcome'].value_counts(normalize=True)}")
plt.figure(figsize=(6,4))
# Add a color palette to the countplot
sns.countplot(x='Outcome', data=df, palette='viridis')
plt.title('Target Variable Distribution (Outcome)')
plt.xlabel('Outcome (0=No Diabetes, 1=Diabetes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'eda_target_distribution.png'))
plt.close()

# 6. Feature Distributions (Histograms)
print("\n[6] FEATURE DISTRIBUTIONS")
# Use a rotating color scheme for histograms
hist_colors = plt.cm.get_cmap('tab10', len(features))
for i, col in enumerate(features):
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30, color=hist_colors(i))
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'eda_dist_{col}.png'))
    plt.close()

# 7. Feature Distributions by Outcome (Boxplots & Violin Plots)
print("\n[7] FEATURE DISTRIBUTIONS BY OUTCOME")
for col in features:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot with palette
    sns.boxplot(x='Outcome', y=col, data=df, ax=axes[0], palette='Set2')
    axes[0].set_title(f'{col} by Outcome (Boxplot)')
    
    # Violin Plot with the same palette
    sns.violinplot(x='Outcome', y=col, data=df, ax=axes[1], inner='quartile', palette='Set2')
    axes[1].set_title(f'{col} by Outcome (Violin Plot)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'eda_box_violin_{col}_by_outcome.png'))
    plt.close()

# 8. Correlation Matrix
print("\n[8] CORRELATION ANALYSIS")
corr = df.corr()
plt.figure(figsize=(10,8))
# Keep coolwarm or choose another diverging map like 'RdBu_r' or 'seismic'
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5, annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'correlation_matrix.png'))
plt.close()

# Print correlations with Outcome
print("\nCorrelation with Outcome:")
print(corr['Outcome'].sort_values(ascending=False))

# 9. Pairplot (sampled for speed)
print("\n[9] PAIRPLOT GENERATION (Sampled)")
sample_df = df.sample(min(200, len(df)), random_state=42)
# Add a palette to pairplot for hue distinction
sns.pairplot(sample_df, hue='Outcome', diag_kind='kde', palette='magma')
plt.suptitle('Pairplot of Features (Sampled)', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'eda_pairplot.png'))
plt.close()

# 10. Outlier Detection Visualization (Boxplots used earlier, this adds summary)
print("\n[10] OUTLIER SUMMARY (IQR method)")
outlier_summary = {}
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_percentage = (len(outliers) / len(df)) * 100
    outlier_summary[col] = {'count': len(outliers), 'percentage': outlier_percentage}
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers ({outlier_percentage:.2f}%) - Range [{lower_bound:.2f}, {upper_bound:.2f}]")
if all(v['count'] == 0 for v in outlier_summary.values()):
    print("No significant outliers detected using the IQR method.")

# 11. Grouped statistics by Outcome
print("\n[11] GROUPED STATISTICS BY OUTCOME")
print(df.groupby('Outcome')[features].agg(['mean', 'median', 'std', 'min', 'max']).T)

# 12. Feature Relationship Plots
print("\n[12] FEATURE RELATIONSHIP PLOTS")
# Select pairs of features with highest correlation
high_corr_pairs = []
for i, row in enumerate(corr.values):
    for j in range(i+1, len(corr.columns)):
        if abs(row[j]) > 0.5 and i != j and corr.columns[i] != 'Outcome' and corr.columns[j] != 'Outcome':
            high_corr_pairs.append((corr.columns[i], corr.columns[j], row[j]))

# Sort by absolute correlation
high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)

# Plot the top pairs
for col1, col2, corr_val in high_corr_pairs[:5]:  # Top 5 correlated pairs
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with regression line
    sns.scatterplot(x=col1, y=col2, data=df, hue='Outcome', palette='viridis', alpha=0.7)
    sns.regplot(x=col1, y=col2, data=df, scatter=False, line_kws={"color": "red", "lw": 1})
    
    plt.title(f'Relationship between {col1} and {col2}\nCorrelation: {corr_val:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'eda_relationship_{col1}_{col2}.png'))
    plt.close()

# 13. KDE Plots by Outcome
print("\n[13] KDE PLOTS BY OUTCOME")
for col in features:
    plt.figure(figsize=(9, 6))
    
    # Get data for each class
    diabetes_positive = df[df['Outcome'] == 1][col].dropna()
    diabetes_negative = df[df['Outcome'] == 0][col].dropna()
    
    # Plot KDE
    sns.kdeplot(diabetes_positive, label='Diabetes (1)', shade=True, color='darkorange')
    sns.kdeplot(diabetes_negative, label='No Diabetes (0)', shade=True, color='skyblue')
    
    plt.title(f'Distribution of {col} by Outcome')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'eda_kde_{col}_by_outcome.png'))
    plt.close()

# 14. Cumulative Distribution Functions
print("\n[14] CUMULATIVE DISTRIBUTION FUNCTIONS")
for col in features:
    plt.figure(figsize=(9, 6))
    
    # Get data for each class
    diabetes_positive = df[df['Outcome'] == 1][col].dropna()
    diabetes_negative = df[df['Outcome'] == 0][col].dropna()
    
    # Plot ECDFs
    sns.ecdfplot(data=diabetes_positive, label='Diabetes (1)', color='darkorange')
    sns.ecdfplot(data=diabetes_negative, label='No Diabetes (0)', color='skyblue')
    
    plt.title(f'ECDF of {col} by Outcome')
    plt.xlabel(col)
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'eda_ecdf_{col}_by_outcome.png'))
    plt.close()

# 15. Q-Q Plots for Normality Check
print("\n[15] Q-Q PLOTS FOR NORMALITY CHECK")
for col in features:
    plt.figure(figsize=(10, 6))
    
    probplot(df[col], plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'eda_qq_{col}.png'))
    plt.close()

# 16. Density Contour Plots for Key Pairs
print("\n[16] DENSITY CONTOUR PLOTS")
key_pairs = [
    ('Glucose', 'BMI'),
    ('Glucose', 'Age'),
    ('BMI', 'Insulin'),
    ('Age', 'BMI')
]

for col1, col2 in key_pairs:
    plt.figure(figsize=(10, 8))
    
    # Create grid with both features
    for outcome, color, label in [(0, 'skyblue', 'No Diabetes'), (1, 'darkorange', 'Diabetes')]:
        subset = df[df['Outcome'] == outcome]
        sns.kdeplot(
            data=subset,
            x=col1, y=col2,
            levels=5,
            alpha=0.6,
            fill=True,
            palette=[color],
            label=label
        )
    
    plt.title(f'Density Contour Plot: {col1} vs {col2}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'eda_contour_{col1}_{col2}.png'))
    plt.close()

# 17. Feature Engineering Sample
print("\n[17] FEATURE ENGINEERING SAMPLE")
# Create some sample derived features
fe_df = df.copy()
# BMI * Glucose interaction
fe_df['BMI_Glucose_Interaction'] = fe_df['BMI'] * fe_df['Glucose']
# Age groups
fe_df['Age_Group'] = pd.cut(fe_df['Age'], bins=[20, 30, 40, 50, 60, 100], labels=['20-30', '30-40', '40-50', '50-60', '60+'])

# Plot one of these features
plt.figure(figsize=(10, 6))
sns.boxplot(x='Age_Group', y='Glucose', hue='Outcome', data=fe_df, palette='Set3')
plt.title('Glucose by Age Group and Diabetes Status')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'eda_feature_eng_age_groups.png'))
plt.close()

# Plot BMI_Glucose interaction
plt.figure(figsize=(10, 6))
sns.histplot(data=fe_df, x='BMI_Glucose_Interaction', hue='Outcome', bins=30, kde=True, palette='Set2')
plt.title('BMI × Glucose Interaction by Outcome')
plt.tight_layout() 
plt.savefig(os.path.join(PLOT_DIR, 'eda_feature_eng_bmi_glucose.png'))
plt.close()

# 18. Anomaly Detection with Isolation Forest
print("\n[18] ANOMALY DETECTION")
# Select only numerical features
X = df[features].copy()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest for anomaly detection
clf = IsolationForest(contamination=0.05, random_state=42)
y_pred = clf.fit_predict(X_scaled)

# Convert predictions to binary anomaly indicator (1 = normal, -1 = anomaly)
df_anomaly = df.copy()
df_anomaly['anomaly'] = y_pred
df_anomaly['anomaly'] = df_anomaly['anomaly'].map({1: 0, -1: 1})  # 0 = normal, 1 = anomaly

# Plot anomalies for key features
for col in ['Glucose', 'BMI', 'Age', 'Insulin']:
    plt.figure(figsize=(10, 6))
    plt.scatter(df_anomaly.index, df_anomaly[col], c=df_anomaly['anomaly'], cmap='viridis', 
                alpha=0.7, edgecolors='k', s=50)
    plt.colorbar(label='Anomaly')
    plt.title(f'Anomaly Detection on {col}')
    plt.xlabel('Index')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'eda_anomaly_{col}.png'))
    plt.close()

# 19. Age Group Analysis
print("\n[19] AGE GROUP ANALYSIS")
# Create age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100], labels=['20-30', '30-40', '40-50', '50-60', '60+'])

# Diabetes prevalence by age group
age_group_outcome = pd.crosstab(df['Age_Group'], df['Outcome'], normalize='index') * 100
print("\nDiabetes prevalence by age group (%):")
print(age_group_outcome)

# Plot diabetes prevalence by age group
plt.figure(figsize=(10, 6))
age_group_outcome[1].plot(kind='bar', color='coral')
plt.title('Diabetes Prevalence by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Percentage with Diabetes (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'eda_diabetes_by_age.png'))
plt.close()

# Statistics by age group
age_group_stats = df.groupby('Age_Group')[features].mean()
print("\nAverage feature values by age group:")
print(age_group_stats)

# 20. PCA Visualization
print("\n[20] PCA VISUALIZATION")
# Scale the data
X = df[features].copy()
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Outcome'] = df['Outcome']

# Plot PCA results
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Outcome', data=pca_df, palette='viridis', alpha=0.8, s=80)
plt.title('PCA Projection of Diabetes Data')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend(title='Diabetes')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'eda_pca_visualization.png'))
plt.close()

# Print PCA component matrix
print("\nPCA component loadings:")
component_df = pd.DataFrame(pca.components_.T, index=features, columns=['PC1', 'PC2'])
print(component_df)

# 21. K-means Clustering
print("\n[21] K-MEANS CLUSTERING")
# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to PCA DataFrame
pca_df['Cluster'] = clusters

# Plot clusters on PCA projection
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='plasma', alpha=0.8, s=80)
plt.title('K-means Clustering on PCA Projection')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'eda_kmeans_clusters.png'))
plt.close()

# Calculate diabetes prevalence by cluster
cluster_outcome = pd.crosstab(pca_df['Cluster'], df['Outcome'], normalize='index') * 100
print("\nDiabetes prevalence by cluster (%):")
print(cluster_outcome)

print(f"\nComprehensive EDA completed. Plots saved in '{PLOT_DIR}' directory.")
