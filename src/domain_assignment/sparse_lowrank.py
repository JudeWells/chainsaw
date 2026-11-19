import numpy as np


def mse_loss_at_residue(V_hat, Y, residue_index):
    # loss contribution of residue given current assignment
    # assumes Y is symmetric
    d = residue_index
    # avoid double-counting the diagonal
    return 2*((V_hat[d]@V_hat.T - Y[d])**2).sum() - (V_hat[d]@V_hat[d] - Y[d,d])**2


def null_mse_loss_at_residue(Y, residue_index):
    # loss contribution of residue when unassigned
    d = residue_index
    return 2*(Y[d]**2).sum() - Y[d,d]**2


def mse_loss(V_hat, Y):
    # only use to compute initial loss
    return ((V_hat@V_hat.T - Y)**2).sum()


def nll_loss(V_hat, Y):
    """We now interpret ys as probabilities.

    We want to minimise the negative log probability.
    
    NB Y must be symmetric and [0-1]
    """
    # only use to compute initial loss
    log_pij = (V_hat@V_hat.T) * np.log(Y) + (1-(V_hat@V_hat.T))*np.log(1-Y)
    # simple sum double counts off diagonals but single counts on diagonals, so need
    # to add diagonal again before dividing by 2
    return -0.5*(log_pij.sum() + np.diag(log_pij).sum())


def nll_loss_at_residue(V_hat, Y, residue_index):
    # d is residue_index
    d = residue_index  # V_hat[d]@V_hat.T is just Adj without instantiating A
    # n.b. whereas in case of mse we double-count off-diagonals, here we don't, hence the difference
    log_pdj = (V_hat[d]@V_hat.T) * np.log(Y[d]) + (1-(V_hat[d]@V_hat.T))*np.log(1-Y[d])
    return -log_pdj.sum()


def null_nll_loss_at_residue(Y, residue_index):
    return - np.log(1-Y[residue_index]).sum()


def greedy_V(Y, N_iters=3, K_init=4, cost_type="mse"):
    """Learn a binary cluster-assignment matrix V (at most one 1 per row) minimizing ||VV' - Y||_2.

    Each row of V_hat has at most one nonzero entry, so V_hat[d] @ V_hat.T for residue d placed in
    cluster k is just the indicator of cluster k's members plus d itself. Maintaining that indicator
    as a (K, D) uint8 matrix `members` scores all K candidates in one vectorised step while
    preserving the exact floating-point operations and summation order (contiguous row sums and
    NumPy-version-dependent scalar promotion) of the original per-candidate loop.
    """
    Y = (Y + Y.T) / 2  # required for consistency with model_v1
    D = Y.shape[0]
    work_dtype = np.result_type(np.uint8, Y.dtype)
    # NumPy scalar promotion: 2 * <float32 scalar> is float64 in NumPy 1.x, float32 in NumPy 2.x.
    scalar_dtype = (2 * work_dtype.type(1)).dtype

    Y_sq = Y ** 2
    if cost_type == "mse":
        loss = Y_sq.sum()  # initial loss for V_hat = 0
    elif cost_type == "nll":
        loss = nll_loss(np.zeros((D, K_init), dtype=np.uint8), Y)
    else:
        raise ValueError(cost_type)

    one = np.uint8(1)  # uint8 scalar preserves upstream NumPy promotion rules
    # Per-residue loss contributions when unassigned vs assigned (diagonal term: (1 - Y[d, d])^2).
    null = 2 * Y_sq.sum(axis=1).astype(scalar_dtype) - np.diagonal(Y).astype(scalar_dtype) ** 2
    diag_assigned = (one - np.diagonal(Y)).astype(scalar_dtype) ** 2

    K_max = K_init
    labels = np.full(D, -1, dtype=np.int64)  # cluster of each residue, -1 = unassigned
    members = np.zeros((K_max, D), dtype=np.uint8)  # members[k, j] = 1 iff residue j is in cluster k
    buf = np.empty((K_max, D), dtype=work_dtype)

    for _ in range(N_iters):
        for d in range(D):
            Yd = Y[d]
            current = labels[d]
            # Row d of V_hat @ V_hat.T if d were placed in cluster k: current members of k, plus d.
            members[:, d] = 1
            np.subtract(members, Yd, out=buf)
            np.multiply(buf, buf, out=buf)
            cost = 2 * buf.sum(axis=1).astype(scalar_dtype) - diag_assigned[d]
            members[:, d] = 0  # restore candidate column and clear d from its previous cluster before reassigning

            loss_minus_d = loss - (cost[current] if current >= 0 else null[d])
            L0 = loss_minus_d + null[d]
            L_opt = loss_minus_d + cost
            z = np.argmin(L_opt)
            if L_opt[z] < L0:
                labels[d] = z
                members[z, d] = 1
                loss = L_opt[z]
            else:
                labels[d] = -1
                loss = L0

            if z == K_max - 1:
                # Expand room for extra potential clusters (note: a cluster emptied in a later sweep is not reclaimed).
                members = np.concatenate((members, np.zeros_like(members)), 0)
                K_max = members.shape[0]
                buf = np.empty((K_max, D), dtype=work_dtype)

    # Drop unused (all-zero) cluster columns before returning.
    nonempty = members.any(axis=1)
    return np.ascontiguousarray(members[nonempty].T), loss
