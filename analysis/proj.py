import umap.umap_ as umap
from sklearn.decomposition import PCA
import numpy as np

def learn_manifold_umap(data, umap_dim, umap_min_dist=0.2, umap_metric='euclidean', umap_neighbors=10):
    md = float(umap_min_dist)
    return umap.UMAP(random_state=0, metric=umap_metric, n_components=umap_dim, n_neighbors=umap_neighbors,
                    min_dist=md).fit_transform(data)

def pca_train(train, test, n_comps):
    pca_ = PCA(n_components=n_comps, whiten=True)
    pca_.fit(train)
    print('train done')
    test_comps = pca_.transform(test)
    print('pca test done')
    return test_comps, pca_.explained_variance_ratio_

def pca(S, n_comps):
    pca_ = PCA(n_components=n_comps, whiten=True)
    return pca_.fit_transform(S), pca_.explained_variance_ratio_, pca_

# og_pca, og_pca_var = pca(max_chan_hptp_temps, 2)
# tform_pca, tform_var = pca(tform_temps_numpy, 2)
# og_reps_pca, og_reps_var = pca(og_reps, 2)
# tform_reps_pca, tform_reps_var = pca(tform_reps, 2)

# og_reps_umap = learn_manifold_umap(og_reps, 2)
# tform_reps_umap = learn_manifold_umap(tform_reps, 2)
