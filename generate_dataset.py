import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_blobs

# Generate a Classification Dataset
X_classification, y_classification = make_classification(
    n_samples=500,     # Total samples
    n_features=5,      # Number of features
    n_informative=3,   # Informative features
    n_redundant=0,     # Redundant features
    n_classes=3,       # Number of classes
    random_state=42
)

# Save Classification Dataset
classification_df = pd.DataFrame(X_classification, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
classification_df['Target'] = y_classification
classification_df.to_csv('classification_dataset.csv', index=False)
print("Classification dataset saved as 'classification_dataset.csv'.")

# Generate a Clustering Dataset
X_clustering, y_clustering = make_blobs(
    n_samples=500,
    centers=3,       # Number of clusters
    n_features=5,    # Number of features
    random_state=42
)

# Save Clustering Dataset
clustering_df = pd.DataFrame(X_clustering, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
clustering_df['Cluster'] = y_clustering
clustering_df.to_csv('clustering_dataset.csv', index=False)
print("Clustering dataset saved as 'clustering_dataset.csv'.")
