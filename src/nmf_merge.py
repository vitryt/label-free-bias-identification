import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, fcluster

def collect_basis_vectors(Ws):
    """
    Ws: list of W matrices, each shape (n_samples, k)
    Returns: array of shape (n_vectors, n_samples)
    """
    basis = []
    for W in Ws:
        for j in range(W.shape[0]):
            w = W[j, :]
            norm = np.linalg.norm(w)
            if norm > 0:
                basis.append(w / norm)
    return np.array(basis)


def cluster_components(basis_vectors, similarity_threshold=0.95):
    """
    Clusters basis vectors based on cosine similarity.
    """
    # cosine distance = 1 - similarity
    sim = cosine_similarity(basis_vectors)
    dist = 1.00001 - sim

    # Hierarchical clustering
    Z = linkage(dist[np.triu_indices(len(basis_vectors), k=1)], method="average")

    # Convert similarity threshold to distance threshold
    dist_thresh = 1 - similarity_threshold
    labels = fcluster(Z, t=dist_thresh, criterion="distance")

    return labels


def merge_clusters(basis_vectors, labels, method="mean"):
    merged = []

    for cluster_id in np.unique(labels):
        members = basis_vectors[labels == cluster_id]

        if method == "mean":
            w = np.mean(members, axis=0)
        elif method == "medoid":
            sims = cosine_similarity(members)
            medoid_idx = np.argmax(np.sum(sims, axis=1))
            w = members[medoid_idx]
        else:
            raise ValueError("Unknown merge method")

        w = np.maximum(w, 0)
        w /= np.linalg.norm(w) + 1e-12
        merged.append(w)

    return np.array(merged)


def component_level_fusion(Ws, similarity_threshold=0.95):
    basis = collect_basis_vectors(Ws)
    labels = cluster_components(basis, similarity_threshold)
    W_merged = merge_clusters(basis, labels)

    return W_merged, labels