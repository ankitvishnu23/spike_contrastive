from sklearn.mixture import GaussianMixture
import hdbscan
import numpy as np

def GMM(train_reps, test_reps, model_params, random_state=0):
    gm = GaussianMixture(model_params['n_clusters'], random_state=random_state).fit(train_reps)
    test_labels = gm.predict(test_reps)
    return test_labels

def HDBSCAN(test_reps, model_params):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=model_params['min_cluster_size'])
    clusterer.fit(test_reps)
    return clusterer.labels_