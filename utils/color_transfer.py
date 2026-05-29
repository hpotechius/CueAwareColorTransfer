"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import cv2
import json
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from numpy.linalg import solve
import concurrent.futures

# ----------------------------------------------------------------------------------------------------------------------
# Approximates weighted KMeans:
# - use unweighted KMeans for assignment (stable)
# - compute centers and cluster masses as WEIGHTED means/sums (correct)
# ----------------------------------------------------------------------------------------------------------------------
def _kmeans_weighted_centers(X, w, K=512, seed=0):
    K = min(K, X.shape[0])
    km = KMeans(n_clusters=K, n_init=2, random_state=seed)
    labels = km.fit_predict(X)  # only for assignment
    C = np.zeros((K, 3), dtype=np.float64)
    mass = np.zeros((K,), dtype=np.float64)
    for k in range(K):
        sel = (labels == k)
        if np.any(sel):
            wk = w[sel]
            Xk = X[sel]
            mk = wk.sum()
            mass[k] = mk
            C[k] = (wk[:, None] * Xk).sum(axis=0) / mk
        else:
            # Fallback: if empty, use the KMeans center
            C[k] = km.cluster_centers_[k]
            mass[k] = 1e-12
    mass /= mass.sum()
    return C.astype(np.float32), mass

# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
def _sinkhorn(w_s, w_r, C2, eps=5e-3, n_iter=500, tol=1e-6):
    Ks, Kr = C2.shape
    K = np.exp(-C2 / eps) + 1e-300
    u = np.ones(Ks) / Ks
    v = np.ones(Kr) / Kr
    for _ in range(n_iter):
        u_prev = u
        u = w_s / (K @ v)
        v = w_r / (K.T @ u)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    Pi = np.outer(u, v) * K
    Pi = np.maximum(Pi, 0.0)
    return Pi


# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
def _fit_rbf_field(C, D, sigma=0.08, lam=1e-3):
    D2 = pairwise_distances(C, C, metric='sqeuclidean')
    Kmat = np.exp(-D2 / (2.0 * sigma * sigma)).astype(np.float64)
    Kreg = Kmat + lam * np.eye(Kmat.shape[0])
    A = np.zeros_like(D, dtype=np.float64)
    for ch in range(3):
        A[:, ch] = solve(Kreg, D[:, ch].astype(np.float64))
    return A, C.astype(np.float64), float(sigma)

# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
def _eval_rbf_field(X, A, C, sigma, batch=200_000):
    X = X.astype(np.float64)
    N = X.shape[0]
    out = np.zeros_like(X)
    for s in range(0, N, batch):
        e = min(N, s + batch)
        D2 = pairwise_distances(X[s:e], C, metric='sqeuclidean')
        Phi = np.exp(-D2 / (2.0 * sigma * sigma))
        out[s:e] = Phi @ A
    return out.astype(np.float32)

# ----------------------------------------------------------------------------------------------------------------------
# X: (N,3), w: (N,)
# Draws a weighted sample, but only from points with w>0.
# ----------------------------------------------------------------------------------------------------------------------
def _sample(X, w, max_samples=300_000, seed=0, allow_replace=False):
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    w = np.asarray(w).reshape(-1)

    # only positive weights
    pos = w > 0
    Xp = X[pos]
    wp = w[pos]

    # number of pixels with positive weight
    npos = Xp.shape[0]
    if npos == 0:
        # Fallback: unweighted random sample from all points
        m = min(max_samples, X.shape[0])
        idx = rng.choice(X.shape[0], size=m, replace=False)
        return X[idx], np.ones((m,), dtype=np.float64)

    m = min(max_samples, npos)
    idx = rng.choice(npos, size=m, replace=allow_replace)
    return Xp[idx], wp[idx]

# ----------------------------------------------------------------------------------------------------------------------
# Flattens the image and weights into 2D arrays for processing. Ensures that weights are non-negative and normalizes 
# them if necessary.
# ----------------------------------------------------------------------------------------------------------------------
def _flatten_img_and_w(img, w):
    X = img.reshape(-1, 3).astype(np.float32)
    if w is None:
        w = np.ones((X.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(w).reshape(-1).astype(np.float64)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w[:] = 1.0
    return X, w

# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
def image_stats(image, weights=None, eps=1e-6):
    r, g, b = cv2.split(image)
    r_vals = r.flatten()
    g_vals = g.flatten()
    b_vals = b.flatten()

    if weights is not None:
        weights = weights.flatten()
        def wmeanstd(vals):
            mean = np.sum(vals * weights) / (np.sum(weights) + eps)
            std = np.sqrt(np.sum(weights * (vals - mean) ** 2) / (np.sum(weights) + eps))
            return mean, std
        r_mean, r_std = wmeanstd(r_vals)
        g_mean, g_std = wmeanstd(g_vals)
        b_mean, b_std = wmeanstd(b_vals)
        return (r_mean, g_mean, b_mean), (r_std, g_std, b_std)
    else:
        return (
            (r_vals.mean(), g_vals.mean(), b_vals.mean()),
            (r_vals.std(), g_vals.std(), b_vals.std())
        )
    
# ----------------------------------------------------------------------------------------------------------------------
# Reinhard color transfer: matches mean and std of source to reference. Optionally with weights.
# ----------------------------------------------------------------------------------------------------------------------
def reinhard_color_transfer(src, ref, src_weights=None, ref_weights=None):
    eps = 1e-6

    src_mean, src_std = image_stats(src, src_weights)
    ref_mean, ref_std = image_stats(ref, ref_weights)

    r, g, b = cv2.split(src)
    r_out = (r - src_mean[0]) / (src_std[0] + eps) * ref_std[0] + ref_mean[0]
    g_out = (g - src_mean[1]) / (src_std[1] + eps) * ref_std[1] + ref_mean[1]
    b_out = (b - src_mean[2]) / (src_std[2] + eps) * ref_std[2] + ref_mean[2]

    rgb_out = cv2.merge([r_out, g_out, b_out])
    rgb_out = np.clip(rgb_out, 0, 1)

    return rgb_out

# ----------------------------------------------------------------------------------------------------------------------
# RBF-OT color transfer: uses radial basis functions and optimal transport for color transfer.
# ----------------------------------------------------------------------------------------------------------------------
def color_transfer_rbf_ot(
    src_img, ref_img,
    src_weights=None, ref_weights=None,
    Ks=512, Kr=512,
    eps=5e-3,
    sigma_rbf=0.08,
    lam=1e-3,
    max_samples=300_000,
    seed=0
):
    src_img_proc = src_img
    ref_img_proc = ref_img

    Hs, Ws, _ = src_img_proc.shape
    Hr, Wr, _ = ref_img_proc.shape

    Xs_full, ws_full = _flatten_img_and_w(src_img_proc, src_weights)
    Xr_full, wr_full = _flatten_img_and_w(ref_img_proc, ref_weights)

    # 1) (Weighted) sampling (importance sampling)
    Xs, ws = _sample(Xs_full, ws_full, max_samples=max_samples, seed=seed)
    Xr, wr = _sample(Xr_full, wr_full, max_samples=max_samples, seed=seed+1)

    # 2) Weighted clustering & masses
    Cs, ws_cl = _kmeans_weighted_centers(Xs, ws, K=Ks, seed=seed)
    Cr, wr_cl = _kmeans_weighted_centers(Xr, wr, K=Kr, seed=seed+1)

    # 3) Sinkhorn-OT with weighted marginals
    C2 = pairwise_distances(Cs, Cr, metric='sqeuclidean').astype(np.float64)
    Pi = _sinkhorn(ws_cl, wr_cl, C2, eps=eps)
  
    # 4) Barycentric targets T per source cluster
    mass_s = Pi.sum(axis=1, keepdims=True) + 1e-12
    T_ot = (Pi @ Cr) / mass_s  # (Ks,3)

    # 6) Fit RBF field (smooth, noise-free)
    D = (T_ot - Cs).astype(np.float32)
    A, C_fit, sig = _fit_rbf_field(Cs, D, sigma=sigma_rbf, lam=lam)
    
    # 7) Apply to all source pixels
    Delta = _eval_rbf_field(Xs_full, A, C_fit, sig)
    out = Xs_full + Delta
    out = np.clip(out, 0.0, 1.0).reshape(Hs, Ws, 3)

    return out

# ----------------------------------------------------------------------------------------------------------------------
# Apply semantic-aware color transfer by processing each semantic class separately and then combining the results.
# ----------------------------------------------------------------------------------------------------------------------
def semantic_color_transfer(src, ref, src_weights, ref_weights, src_semantics, ref_semantics, ct_type, parallel=True):
    result = np.zeros_like(src, dtype=np.float32)

    # Load the semantic distance matrix from the JSON file, which contains the similarity scores between semantic classes.
    with open("meta/semantic_distance_matrix.json", "r") as f:
        data = json.load(f)

    def process_class(src_key, src_array):
        src_json_key = src_key.split(" ")[0].lower()
        print(f"# Check source class: {src_key}")
        if np.all(src_array == 0):
            return None

        ref_candidates = []
        for ref_key, ref_array in ref_semantics.items():
            ref_json_key = ref_key.split(" ")[0].lower()
            prob = data[src_json_key][ref_json_key]
            ref_candidates.append((ref_key, ref_array, prob))
        ref_candidates.sort(key=lambda x: x[2], reverse=True)

        # Get the best reference class for the given source class that has non-zero pixels
        best_ref = None
        for ref_key, ref_array, prob in ref_candidates:
            if np.all(ref_array == 0):
                continue
            else:
                best_ref = (ref_key, ref_array, prob)
                break

        if best_ref is not None:
            ref_key, ref_array, prob = best_ref
            if ct_type == "reinhard":
                transferred = reinhard_color_transfer(
                    src, 
                    ref, 
                    src_weights=src_weights*src_array, 
                    ref_weights=ref_weights*ref_array
                )
            elif ct_type == "optimal":
                transferred = color_transfer_rbf_ot(
                    src, 
                    ref, 
                    src_weights=src_weights*src_array, 
                    ref_weights=ref_weights*ref_array
                )

            return transferred * src_array
        return None

    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_class, src_key, src_array) for src_key, src_array in src_semantics.items()]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res is not None:
                    result += res
    else:
        for src_key, src_array in src_semantics.items():
            res = process_class(src_key, src_array)
            if res is not None:
                result += res

    return result