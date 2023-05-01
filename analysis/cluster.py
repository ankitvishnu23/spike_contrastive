from sklearn.mixture import GaussianMixture
import hdbscan
import numpy as np

def GMM(train_reps, test_reps, n_clusters, random_state=0):
    gm = GaussianMixture(n_clusters, random_state=random_state).fit(train_reps)
    test_labels = gm.predict(test_reps)
    return test_labels

def HDBSCAN(test_reps, n_clusters):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    clusterer.fit(test_reps)
    return clusterer.labels_