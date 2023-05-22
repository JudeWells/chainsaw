import numpy as np
from sklearn.cluster import KMeans


def k_means_with_fixed_centre(
    X,
    n_var_clusters,
    *,
    init="k-means++",
    n_init=10,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
    fixed_centre=None,
):
    """A modified version of the standard K-means algorithm,
    in which a single centre isnt't updated from its initialisation.

    In practice we soft-enforce this by adding a single observation with high
    sample weight at the fixed centre. Then during the k-means centroid
    update step, whichever cluster this pseudo observation is initially assigned to will 
    place its centre at the pseudo observation because of its dominant weight.

    A further extension we might want to consider would be some way of
    controlling how strongly attracting the fixed clusters are.

    Standard k-means has no way of imposing per-cluster scales on attractiveness
    - but Gaussian mixture models achieve just this. 

    Alternatively, BP suggested we could just chuck vectors of ones onto
    the end of all observations, which would mean that one-hot vectors
    were closer to each other than they were to the zero vector.

    Refs:
        https://en.wikipedia.org/wiki/K-means_clustering
        https://stats.stackexchange.com/questions/429363/k-means-clustering-with-some-known-centers
        https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/cluster/_k_means_lloyd.pyx
    """
    n, D = X.shape
    if fixed_centre is None:
        fixed_centre = np.zeros((1, D))
    else:
        assert fixed_centre.shape == (D,)
        fixed_centre = fixed_centre[None]

    X = np.concatenate((X,fixed_centre), axis=0)

    sample_weight = np.ones((n+1,))
    sample_weight[-1] = 1e5
    est = KMeans(
        n_clusters=n_var_clusters + 1,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
    ).fit(X, sample_weight=sample_weight)

    # the 'background' cluster will be the one at. labels[-1]
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_
