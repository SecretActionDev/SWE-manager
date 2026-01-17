import pandas as pd
from swe_manager.features import NUMERIC_COLS, TEXT_COL
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
import hdbscan

def clean_and_embed(df: pd.DataFrame):
    ### scale numeric features
    X_num = df[NUMERIC_COLS].fillna(0)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X_text = embedder.encode(
        df[TEXT_COL].tolist(),
        batch_size=64,
        show_progress_bar=True
    )

    return X_num_scaled, X_text

def reduce_dimension(features):
    loaded_umap = joblib.load("/Users/elmiraonagh/Desktop/courses/6444/project/SWE-manger/SWE-Manger/SWE-Manager/assets/umap_model.pkl")
    features_reduced = loaded_umap.transform(features)
    return features_reduced

def Clusterer(features_reduced, df):
    loaded_hdbscan = joblib.load("/Users/elmiraonagh/Desktop/courses/6444/project/SWE-manger/SWE-Manger/SWE-Manager/assets/hdbscan_model_all.pkl")
     # Predict cluster labels
    # cluster_labels = loaded_hdbscan.predict(features_reduced)

    df['cluster'] = loaded_hdbscan.labels_
    return df

def cluster_issue(df: pd.DataFrame) -> pd.DataFrame:
    print(f" Processing {len(df)} instances for clustering..")
    X_num_scaled, X_text = clean_and_embed(df)
    TEXT_WEIGHT = 1.0
    NUM_WEIGHT = 0.5   # structural signal, not dominant

    features = np.hstack([
        TEXT_WEIGHT * X_text,
        NUM_WEIGHT * X_num_scaled
    ])
    print("Reducing dimension using UMAP...")
    features_reduced = reduce_dimension(features)
    print("Finished Reducing Dimension...")

    print("Clustering with HDBSCAN...")
    df_cluster = Clusterer(features_reduced, df)
    df_cluster.to_csv("/Users/elmiraonagh/Desktop/courses/6444/project/SWE-manger/SWE-Manger/SWE-Manager/data/cluster.csv", index = False)
    print("Finished and saved clusters")

    return df