"""The vectorised greedy_V must reproduce the original reference algorithm bit-for-bit."""
import numpy as np
import pytest

from src.domain_assignment.sparse_lowrank import (
    greedy_V,
    mse_loss,
    mse_loss_at_residue,
    nll_loss,
    null_mse_loss_at_residue,
)


def _greedy_V_reference(Y, N_iters=3, K_init=4, cost_type="mse"):
    """Original per-candidate greedy_V implementation kept as the test oracle."""
    Y = (Y + Y.T) / 2
    V_hat = np.zeros((Y.shape[0], K_init), dtype=np.uint8)
    loss = mse_loss(V_hat, Y) if cost_type == "mse" else nll_loss(V_hat, Y)
    K_max = K_init

    for _ in range(N_iters):
        for d in range(V_hat.shape[0]):
            loss_minus_d = loss - mse_loss_at_residue(V_hat, Y, d)
            V_hat[d] *= 0
            L0 = loss_minus_d + null_mse_loss_at_residue(Y, d)
            L_opt = np.zeros(K_max)
            for k in range(K_max):
                V_hat[d, k] = 1
                L_opt[k] = loss_minus_d + mse_loss_at_residue(V_hat, Y, d)
                V_hat[d, k] = 0

            z = np.argmin(L_opt)
            if L_opt[z] < L0:
                V_hat[d, z] = 1

            loss = loss_minus_d + mse_loss_at_residue(V_hat, Y, d)

            if z == K_max - 1:
                V_hat = np.concatenate((V_hat, np.zeros_like(V_hat)), -1)
                K_max = V_hat.shape[1]

    empty = V_hat.sum(0) == 0
    return V_hat[:, ~empty], loss


def _block_matrix(rng, n, n_domains, noise, dtype=np.float32):
    cuts = np.sort(rng.choice(np.arange(1, n), size=n_domains - 1, replace=False)) if n_domains > 1 else []
    labels = np.zeros(n, dtype=int)
    for c in cuts:
        labels[c:] += 1
    if n_domains > 2:
        labels[labels == n_domains - 1] = 0
    y = (labels[:, None] == labels[None, :]).astype(dtype)
    return np.clip(y + noise * rng.standard_normal((n, n)).astype(dtype), 1e-6, 1 - 1e-6).astype(dtype)


def _assert_matches(y, cost_type="mse"):
    V_ref, loss_ref = _greedy_V_reference(y, cost_type=cost_type)
    V_new, loss_new = greedy_V(y, cost_type=cost_type)
    assert V_new.dtype == V_ref.dtype
    np.testing.assert_array_equal(V_new, V_ref)
    assert type(loss_new) is type(loss_ref)
    assert loss_new.tobytes() == loss_ref.tobytes()


@pytest.mark.parametrize("n,k,noise", [(1, 1, 0.1), (7, 1, 0.5), (60, 3, 0.3), (300, 6, 0.25)])
def test_greedy_V_matches_reference(n, k, noise):
    _assert_matches(_block_matrix(np.random.default_rng(n), n, k, noise))


def test_greedy_V_ties_float64_and_nll():
    rng = np.random.default_rng(42)
    for y in (
        rng.random((60, 60), dtype=np.float32),
        np.full((40, 40), 0.5, dtype=np.float32),
        np.round(rng.random((60, 60), dtype=np.float32) * 4) / 4,
        _block_matrix(rng, 45, 3, 0.3, dtype=np.float64),
    ):
        _assert_matches(y)
    _assert_matches(_block_matrix(rng, 45, 3, 0.3, dtype=np.float32), cost_type="nll")
