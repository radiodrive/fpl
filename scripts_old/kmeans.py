import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV
csv_path = 'fpl_players.csv'  # <-- replace with your file path
df = pd.read_csv(csv_path)

# Step 2: Select relevant features
features = [
    'Name', 'Cost (M)', 'Total Points', 'Goals Scored', 'Assists', 'Minutes',
    'Selected By (%)', 'Form', 'Points Per Game', 'ICT Index',
    'Influence', 'Creativity', 'Threat'
]
df = df[features]

# Step 3: Clean the percentage column
df['Selected By (%)'] = (
    df['Selected By (%)']
    .astype(str)
    .str.replace('%', '', regex=False)
    .pipe(pd.to_numeric, errors='coerce')
)

# Step 4: Convert all other numeric fields
numeric_columns = [c for c in df.columns if c != 'Name']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Step 5: Drop any rows with missing data
df_clean = df.dropna()

# Step 6: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[numeric_columns])

# Optional: Elbow method to identify optimal k
wcss = []
k_values = range(2, 11)
for k_ in k_values:
    km = KMeans(n_clusters=k_, random_state=42)
    km.fit(scaled_data)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_values, wcss, 'o-', color='tab:blue')
plt.title('Elbow Method: WCSS vs. Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (inertia)')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Step 7: Choose k (after inspecting elbow plot)
k = 10  # adjust based on elbow plot
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Step 8: Add cluster labels
df_clean['Cluster'] = clusters

# Step 9: Print summary statistics per cluster
cluster_means = df_clean.groupby('Cluster')[numeric_columns].mean().round(2)
print("\nAverage values per cluster:\n", cluster_means)

# Step 10: Print player names grouped by cluster
for cid in sorted(df_clean['Cluster'].unique()):
    print(f"\n🧩 Cluster {cid} Players:")
    for name in df_clean.loc[df_clean['Cluster'] == cid, 'Name']:
        print(f" - {name}")

# Step 11: Save to CSV
df_clean.to_csv('clustered_fpl_players.csv', index=False)
print("\n✅ Results saved to 'clustered_fpl_players.csv'")

# Step 12: PCA for 2D visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Step 13: Show feature importances driving PCA axes
pca_df = pd.DataFrame(pca.components_, columns=numeric_columns, index=['PC1', 'PC2'])
print("\nTop features driving PC1:")
print(pca_df.loc['PC1'].abs().sort_values(ascending=False).head(10))
print("\nTop features driving PC2:")
print(pca_df.loc['PC2'].abs().sort_values(ascending=False).head(10))

# Step 14: Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1],
            c=clusters, cmap='tab10', s=50)
plt.title('FPL Player Clusters (PCA 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()

# Top N most selected players in each cluster
top_n = 5

for cluster_id in sorted(df_clean['Cluster'].unique()):
    top_players = df_clean[df_clean['Cluster'] == cluster_id] \
        .sort_values('Selected By (%)', ascending=False) \
        .head(top_n)

    plt.figure(figsize=(8, 4))
    sns.barplot(x='Selected By (%)', y='Name', data=top_players, palette='viridis')
    plt.title(f'Top {top_n} Most Picked Players in Cluster {cluster_id}')
    plt.xlabel('% Selected By')
    plt.ylabel('Player Name')
    plt.xlim(0, df_clean['Selected By (%)'].max() + 5)
    plt.tight_layout()
    plt.show()