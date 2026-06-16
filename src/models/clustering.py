import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from typing import Tuple, Dict 

def run_pca_decomposition(scaled_df: pd.DataFrame, n_components: int = 3) -> Tuple[pd.DataFrame, PCA]:
    max_possible_components = scaled_df.shape[1]

    if n_components > max_possible_components: 
        print(f"Requested {n_components} components, but data only has {max_possible_components} columns.")
        print(f"Adjusting target components to: {max_possible_components}")
        n_components = max_possible_components
    print(f"Executing PCA Decomposition (Final Components: {n_components})")

    pca = PCA(n_components==n_components, random_state=42)
    pca_matrix = pca.fit_transform(scaled_df)

    explained_var = pca.explained_variance_ratio_
    total_var = np.sum(explained_var) * 100 
    print(f"PCA Complete. Cumulative Variance Captured: {total_var:.2f}%")
    for idx, var in enumerate(explained_var): 
        print(f"└─ Component {idx+1}: {var*100:.2f}% variance explained")
    
    col_names = [f"PC_{i+1}" for i in range(pca_matrix.shape[1])]
    pca_df = pd.DataFrame(pca_matrix, columns=col_names, index=scaled_df.index)
    return pca_df, pca

def evaluate_elbow_optimization(pca_df: pd.DataFrame, max_k: int = 10) -> Dict[int, float]:
    print(f"Running K-Means Elbow Optimization")
    inertia_scores = {}

    for k in range(1, max_k + 1): 
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(pca_df)
        inertia_scores[k] = kmeans.inertia_
    print("Elbow optimization calculations complete")
    return inertia_scores

def fit_final_regime_model(pca_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, KMeans]: 
    print(f"\nFitting Definitive K-Means Model")

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_df)
    
    labeled_df = pca_df.copy()
    labeled_df['Regime'] = cluster_labels

    print("Regime Sample Distribution")
    distribution = labeled_df['Regime'].value_counts().sort_index()

    for regime_id, count in distribution.items():
        percentage = (count / len(labeled_df)) * 100
        print(f"  └─ Regime Label {regime_id}: {count} market days allocated ({percentage:.2f}%)")
        
    return labeled_df, kmeans

def generate_regime_profiles(scaled_df: pd.DataFrame, labeled_df: pd.DataFrame) -> pd.DataFrame:
    print("\nGenerating Regime Char Profiling")

    profile_df = scaled_df.copy()
    profile_df['Regime'] = labeled_df['Regime']
    profiles = profile_df.groupby('Regime').mean()
    return profiles 
