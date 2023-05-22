import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.manifold._spectral_embedding import _graph_is_connected, _set_diag, spectral_embedding
from sklearn.cluster._spectral import *
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils import check_array, check_random_state, check_scalar, check_symmetric

from src.domain_assignment.kmeans import k_means_with_fixed_centre


def determine_n_clusters(eigenvalues, method="max", gap_threshold=None):
    """Determine number of clusters by looking at gap between k and k+1 th eigenvalues.

    Depending on threshold type, we compare lam k+1 - lam k to gap_threshold,
    or we compare lam k+1 / lam k / av( lam j+1/lam j), j<k to gap threshold.

    TODO: add option to find first gap exceeding threshold/last gap exceeding threshold.

    c.f. spectrum r package, von luxburg.
    self-tuning spectral clustering suggests an alternative, more complex procedure.
    """
    gaps = eigenvalues[1:] - eigenvalues[:-1]
    if method == "max":
        n = np.argmax(gaps) + 1
    elif method == "abs":
        # we might actually want to find the LAST gap above a threshold.
        assert gap_threshold is not None
        n = np.argmax(gaps > gap_threshold) + 1
    elif method == "ratio":
        assert gap_threshold is not None
        raise NotImplementedError()

    return n


def spectral_embedding_with_eigenvalues(
    adjacency,
    *,
    eigen_solver=None,
    random_state=None,
    eigen_tol=0.0,
    norm_laplacian=True,
    drop_first=True,
):
    """
    Steps:
    1. compute eigenvectors of normalized laplacian Lsym (assuming norm_laplacian is True)
    2. get eigenvectors of rw laplacian.


    Modifies scikit-learn spectral clustering to return eigenvalues and
    full set of embeddings (data points projected onto eigenvectors?), 
    instead of only the first n components.

    Modified lines are identified with # >>>
    
    https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/manifold/_spectral_embedding.py#L13

    Additionally we want to add an option to leave some points unassigned.
    This should be handled via k-means itself, or an addtional post-processing
    step

    Original docstring:

    Project the sample on the first eigenvectors of the graph Laplacian.
    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.
    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).
    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.
    Note : Laplacian Eigenmaps is the actual algorithm implemented here.
    Read more in the :ref:`User Guide <spectral_embedding>`.
    Parameters
    ----------
    adjacency : {array-like, sparse graph} of shape (n_samples, n_samples)
        The adjacency matrix of the graph to embed.
    n_components : int, default=8
        The dimension of the projection subspace.
    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities. If None, then ``'arpack'`` is
        used.
    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).
        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.
    eigen_tol : float, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.
    norm_laplacian : bool, default=True
        If True, then compute symmetric normalized Laplacian.
    drop_first : bool, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.
    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
        The reduced samples.
    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.
    References
    ----------
    * https://en.wikipedia.org/wiki/LOBPCG
    * :doi:`"Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method",
      Andrew V. Knyazev
      <10.1137/S1064827500366124>`
    """
    adjacency = check_symmetric(adjacency)

    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError as e:
        if eigen_solver == "amg":
            raise ValueError(
                "The eigen_solver was set to 'amg', but pyamg is not available."
            ) from e

    if eigen_solver is None:
        eigen_solver = "arpack"
    elif eigen_solver not in ("arpack", "lobpcg", "amg"):
        raise ValueError(
            "Unknown value for eigen_solver: '%s'."
            "Should be 'amg', 'arpack', or 'lobpcg'" % eigen_solver
        )

    random_state = check_random_state(random_state)

    n_nodes = adjacency.shape[0]
    n_components = int(n_nodes / 2)

    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected, spectral embedding may not work as expected."
        )

    laplacian, dd = csgraph_laplacian(
        adjacency, normed=norm_laplacian, return_diag=True
    )
    if (
        eigen_solver == "arpack"
        or eigen_solver != "lobpcg"
        and (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)
    ):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        # Here we'll use shift-invert mode for fast eigenvalues
        # (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        #  for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        # in standard mode.
        try:
            # We are computing the opposite of the laplacian inplace so as
            # to spare a memory allocation of a possibly very large array
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            eigenvalues, diffusion_map = eigsh(
                laplacian, k=n_components, sigma=1.0, which="LM", tol=eigen_tol, v0=v0
            )
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                # recover u = D^-1/2 x from the eigenvector output x
                embedding = embedding / dd
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails. We fallback to lobpcg
            eigen_solver = "lobpcg"
            # Revert the laplacian to its opposite to have lobpcg work
            laplacian *= -1

    elif eigen_solver == "amg":
        # Use AMG to get a preconditioner and speed up the eigenvalue
        # problem.
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        laplacian = check_array(
            laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        )
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        # The Laplacian matrix is always singular, having at least one zero
        # eigenvalue, corresponding to the trivial eigenvector, which is a
        # constant. Using a singular matrix for preconditioning may result in
        # random failures in LOBPCG and is not supported by the existing
        # theory:
        #     see https://doi.org/10.1007/s10208-015-9297-1
        # Shift the Laplacian so its diagononal is not all ones. The shift
        # does change the eigenpairs however, so we'll feed the shifted
        # matrix to the solver and afterward set it back to the original.
        diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
        laplacian += diag_shift
        ml = smoothed_aggregation_solver(check_array(laplacian, accept_sparse="csr"))
        laplacian -= diag_shift

        M = ml.aspreconditioner()
        # Create initial approximation X to eigenvectors
        X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
        X[:, 0] = dd.ravel()
        X = X.astype(laplacian.dtype)
        # >>>
        eigenvalues, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.0e-5, largest=False)
        embedding = diffusion_map.T
        if norm_laplacian:
            # recover u = D^-1/2 x from the eigenvector output x
            embedding = embedding / dd
        if embedding.shape[0] == 1:
            raise ValueError

    if eigen_solver == "lobpcg":
        laplacian = check_array(
            laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        )
        if n_nodes < 5 * n_components + 1:
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to eigh, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.toarray()
            eigenvalues, diffusion_map = eigh(laplacian, check_finite=False)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                # recover u = D^-1/2 x from the eigenvector output x
                embedding = embedding / dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension and create initial
            # approximation X to eigenvectors
            X = random_state.standard_normal(
                size=(laplacian.shape[0], n_components + 1)
            )
            X[:, 0] = dd.ravel()
            X = X.astype(laplacian.dtype)
            _, diffusion_map = lobpcg(
                laplacian, X, tol=1e-5, largest=False, maxiter=2000
            )
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                # recover u = D^-1/2 x from the eigenvector output x
                embedding = embedding / dd
            if embedding.shape[0] == 1:
                raise ValueError

    # >>>
    embedding = _deterministic_vector_sign_flip(embedding)
    return -eigenvalues[::-1], embedding[:n_components].T


def spectral_clustering(
    affinity,
    *,
    n_clusters='auto',
    eigen_solver=None,
    random_state=None,
    n_init=10,
    eigen_tol=0.0,
    assign_labels="kmeans",
    verbose=False,
    gap_threshold=0.2,
    nclust_method="max",
    add_disconnected_cluster=False,
    fix_disconnected_cluster_to=None,
    eigenvector_norm=None,
):
    """
    Modified from sklearn to add automated determination of n_clusters (n_clusters='auto'),
    and optional addition of a background cluster.

    The rows of the N x K matrix whose columns are the first K eigenvectors are
    in the ideal case one-hot assignments to the connected components, since in
    this case the eigenvectors are indicator vectors for the connected components.

    Spectral clustering acts on these rows. Disconnected nodes then have null rows
    (i.e. rows that are the zero vector in K d space).

    We therefore add an extra 'disconnected' cluster, either manually fixed at origin,
    or, as currently implemented, by increasing the number of clusters in the clustering
    step.

    In practice, the eigenvectors retrieved by a given numerical solver may not
    be exactly the indicator vectors for the connected components, and can instead be
    any? other orthonormal basis for the space spanned by the indicator vectors for
    the connected components. In this case, disconnected residues should still correspond
    to zero-norm rows in the eigenvector matrix. As an extra lever to control the distribution
    of rows in this case (to avoid connected rows close to the origin), we provide the
    option to fix the norm of the eigenvectors (i.e. columns of the eigenvector matrix),
    via the eigenvector_norm kwarg.

    Handling outliers is a challenging problem.
    1. Post-removal of outliers
        One strategy is to run some kind of linear post-processing on the
    clusters detected by spectral clustering. e.g. we use a normalized cut
    method to detect boundaries between domain and linker regions.
        Or we just look for outliers based on distance from centroid
        https://github.com/DavidBrear/sklearn-cookbook/blob/master/Chapter%203/3.8%20Using%20KMeans%20for%20Outlier%20Detection.ipynb

    2. Pre-removal of outliers:
        K-means with noise:
        https://arxiv.org/pdf/2003.02433.pdf
        https://eprints.cs.univie.ac.at/5809/1/kmn.pdf
        https://github.com/sunho/kmor-np#1

    3. Extra cluster for disconnected nodes:
    this is what is currently implemented. In a fully connected K-component graph,
    the K first eigenvectors are indicator vectors for the connected components.



    Apply clustering to a projection of the normalized Laplacian.
    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance, when clusters are
    nested circles on the 2D plane.
    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts [1]_, [2]_.
    Read more in the :ref:`User Guide <spectral_clustering>`.
    Parameters
    ----------
    affinity : {array-like, sparse matrix} of shape (n_samples, n_samples)
        The affinity matrix describing the relationship of the samples to
        embed. **Must be symmetric**.
        Possible examples:
          - adjacency matrix of a graph,
          - heat kernel of the pairwise_old distance matrix of the samples,
          - symmetric k-nearest neighbours connectivity matrix of the samples.
    n_clusters : int, default=None
        Number of clusters to extract.
    n_components : int, default=n_clusters
        Number of eigenvectors to use for the spectral embedding.
    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition method. If None then ``'arpack'`` is used.
        See [4]_ for more details regarding ``'lobpcg'``.
        Eigensolver ``'amg'`` runs ``'lobpcg'`` with optional
        Algebraic MultiGrid preconditioning and requires pyamg to be installed.
        It can be faster on very large sparse problems [6]_ and [7]_.
    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).
        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. Only used if
        ``assign_labels='kmeans'``.
    eigen_tol : float, default="auto"
        Stopping criterion for eigendecomposition of the Laplacian matrix.
        If `eigen_tol="auto"` then the passed tolerance will depend on the
        `eigen_solver`:
        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;
        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then
          `eigen_tol=None` which configures the underlying `lobpcg` solver to
          automatically resolve the value according to their heuristics. See,
          :func:`scipy.sparse.linalg.lobpcg` for details.
        Note that when using `eigen_solver="lobpcg"` or `eigen_solver="amg"`
        values of `tol<1e-5` may lead to convergence issues and should be
        avoided.
        .. versionadded:: 1.2
           Added 'auto' option.
    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, default='kmeans'
        The strategy to use to assign labels in the embedding
        space.  There are three ways to assign labels after the Laplacian
        embedding.  k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another
        approach which is less sensitive to random initialization [3]_.
        The cluster_qr method [5]_ directly extracts clusters from eigenvectors
        in spectral clustering. In contrast to k-means and discretization, cluster_qr
        has no tuning parameters and is not an iterative method, yet may outperform
        k-means and discretization in terms of both quality and speed.
        .. versionchanged:: 1.1
           Added new labeling method 'cluster_qr'.
    verbose : bool, default=False
        Verbosity mode.
        .. versionadded:: 0.24
    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.
    Notes
    -----
    The graph should contain only one connected component, elsewhere
    the results make little sense.
    This algorithm solves the normalized cut for `k=2`: it is a
    normalized spectral clustering.
    References
    ----------
    .. [1] :doi:`Normalized cuts and image segmentation, 2000
           Jianbo Shi, Jitendra Malik
           <10.1109/34.868688>`
    .. [2] :doi:`A Tutorial on Spectral Clustering, 2007
           Ulrike von Luxburg
           <10.1007/s11222-007-9033-z>`
    .. [3] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf>`_
    .. [4] :doi:`Toward the Optimal Preconditioned Eigensolver:
           Locally Optimal Block Preconditioned Conjugate Gradient Method, 2001
           A. V. Knyazev
           SIAM Journal on Scientific Computing 23, no. 2, pp. 517-541.
           <10.1137/S1064827500366124>`
    .. [5] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
           Anil Damle, Victor Minden, Lexing Ying
           <10.1093/imaiai/iay008>`
    .. [6] :doi:`Multiscale Spectral Image Segmentation Multiscale preconditioning
           for computing eigenvalues of graph Laplacians in image segmentation, 2006
           Andrew Knyazev
           <10.13140/RG.2.2.35280.02565>`
    .. [7] :doi:`Preconditioned spectral clustering for stochastic block partition
           streaming graph challenge (Preliminary version at arXiv.)
           David Zhuzhunashvili, Andrew Knyazev
           <10.1109/HPEC.2017.8091045>`
    """

    assert fix_disconnected_cluster_to in [None, "origin", "vmin"],\
        f"fix_disconnected_cluster_to must be origin or vmin but is {fix_disconnected_cluster_to}"

    if assign_labels not in ("kmeans", "discretize", "cluster_qr"):
        raise ValueError(
            "The 'assign_labels' parameter should be "
            "'kmeans' or 'discretize', or 'cluster_qr', "
            f"but {assign_labels!r} was given"
        )
    if isinstance(affinity, np.matrix):
        raise TypeError(
            "spectral_clustering does not support passing in affinity as an "
            "np.matrix. Please convert to a numpy array with np.asarray. For "
            "more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html",  # noqa
        )

    if eigen_solver == "arpack":
        check_scalar(
            self.eigen_tol,
            "eigen_tol",
            target_type=numbers.Real,
            min_val=0,
            include_boundaries="left",
        )

    random_state = check_random_state(random_state)

    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully-connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.

    # n.b. clustering requires the correct number of clusters to be returned.
    # note unlike the original function we don't allow n_components != n_clusters
    if isinstance(n_clusters, int):
        maps = spectral_embedding(
            affinity,
            n_components=n_clusters,
            eigen_solver=eigen_solver,
            random_state=random_state,
            eigen_tol=eigen_tol,
            drop_first=False,
        )
    elif n_clusters == "auto":
        eigenvals, maps = spectral_embedding_with_eigenvalues(
            affinity,
            eigen_solver=eigen_solver,
            random_state=random_state,
            eigen_tol=eigen_tol,
            drop_first=False,
        )

        # n.b. original function allows n_components (for embeddings) to differ from n_clusters (for k-means)
        # though defaults to them being the same
        n_clusters = determine_n_clusters(
            eigenvals,
            gap_threshold=gap_threshold,
            method=nclust_method,
        )
        maps = maps[:, :n_clusters]  # this is required for the clustering step.
        if verbose:
            print(f"Selected {n_clusters} clusters based on eigenvalue gaps", maps.shape)

    if verbose:
        print(f"Computing label assignment using {assign_labels}")

    if eigenvector_norm is not None:
        maps = eigenvector_norm * maps / np.sqrt((maps**2).sum(0))

    background_cluster_id = None
    if add_disconnected_cluster:
        # either fix a cluster at the origin, or allow the cluster's location to be inferred.
        if fix_disconnected_cluster_to is not None:
            if fix_disconnected_cluster_to == "origin":
                fixed_centre = np.zeros(maps.shape[-1])
            elif fix_disconnected_cluster_to == "vmin":
                fixed_centre = np.min(np.abs(maps), axis=0)
            else:
                raise ValueError()

            assert assign_labels == "kmeans", "Disconnected cluster currently only supported with kmeans clustering"
            _, labels, _ = k_means_with_fixed_centre(
                maps,
                n_clusters,
                random_state=random_state,
                n_init=n_init,
                verbose=verbose,
                fixed_centre=fixed_centre,
            )
            background_cluster_id = labels[-1]
            labels = labels[:-1]
        else:
            centres, labels, _ = k_means(
                maps, n_clusters + 1, random_state=random_state, n_init=n_init, verbose=verbose,
            )
            sq_norms = ((centres)**2).sum(-1)
            background_cluster_id = np.argmin(sq_norms)  # closest to origin
            # print(centres, norms, background_cluster_id)

    elif assign_labels == "kmeans":
        _, labels, _ = k_means(
            maps, n_clusters, random_state=random_state, n_init=n_init, verbose=verbose
        )
    elif assign_labels == "cluster_qr":
        labels = cluster_qr(maps)
    else:
        labels = discretize(maps, random_state=random_state)

    return labels, background_cluster_id
