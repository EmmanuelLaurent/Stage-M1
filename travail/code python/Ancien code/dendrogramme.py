import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram
# =============================================================================
# Importation bdd
# =============================================================================

BDD = pd.read_excel('C:/Users/Admin/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/2-data_ossements/2-2/feature_table_mzwid_0.019_minfrac_0.1_no_int.xlsx')
BDD2= BDD
del BDD2['label']
y = BDD2['feature']
X = BDD2.drop('feature',axis=1)

# =============================================================================
# fonction dendrogramme
# =============================================================================

def llf(id):
    return y[id]

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, leaf_label_func=llf, **kwargs)
    
# =============================================================================
# clustering
# =============================================================================


clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete', affinity='manhattan')
fit = clustering.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(fit)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# =============================================================================
# Score de silhouette
# =============================================================================

clustering = AgglomerativeClustering(linkage='complete', affinity='manhattan').fit(X)
labels=clustering.fit_predict(X)
labels
metrics. silhouette_score(X, labels, metric = 'manhattan')
