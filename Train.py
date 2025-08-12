#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PCA + GMM and align GMM components to substrates using the Hungarian algorithm.

Workflow:
1) Load features X (tab-delimited, shape: n_samples x d)
2) Load binary labels Y (CSV, shape: n_samples x n_substrates; values in {0,1})
3) Fit PCA -> X_pca
4) Fit GMM with n_components == n_substrates
5) Build cluster-indicator matrix C from hard assignments (argmax over components)
6) Build cost matrix M[k, s] = FN count comparing C[:,k] (pred) vs Y[:,s] (true)
7) Hungarian assignment to get mapping component k -> substrate s
8) Save PCA, GMM, and mapping

Required:
  --n_components : number of substrates (must equal GMM components)

Example:
  python Train.py \
      --feature_path data/train_feats.txt \
      --label_path   data/train_labels.csv \
      --n_components 102 \
      --pca_dim 3 \
      --output_dir results
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment


def parse_args():
    ap = argparse.ArgumentParser(description="Train GMM and align components to substrates")
    ap.add_argument("--feature_path", type=str, required=True,
                    help="Path to feature file (tab-delimited TXT), shape n_samples x d")
    ap.add_argument("--label_path", type=str, required=True,
                    help="Path to label CSV, shape n_samples x n_substrates, values in {0,1}")
    ap.add_argument("--n_components", type=int, required=True,
                    help="Number of GMM components (equal to the number of substrates)")
    ap.add_argument("--pca_dim", type=int, default=100,
                    help="Number of PCA components (default: 100)")
    ap.add_argument("--output_dir", type=str, default="results",
                    help="Directory to save models/mapping (default: results)")
    return ap.parse_args()


def false_negatives(y_true_col: np.ndarray, y_pred_col: np.ndarray) -> int:
    """Count FN: true==1 and pred==0."""
    return int(np.sum((y_true_col == 1) & (y_pred_col == 0)))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load data
    X = np.loadtxt(args.feature_path, delimiter="\t")
    Y = pd.read_csv(args.label_path, header=None).values  # shape: n_samples x n_substrates

    n_samples = X.shape[0]
    if Y.shape[0] != n_samples:
        raise ValueError(f"Feature/label sample size mismatch: {X.shape[0]} vs {Y.shape[0]}")

    if Y.shape[1] != args.n_components:
        raise ValueError(f"labels columns ({Y.shape[1]}) must equal n_components ({args.n_components})")

    # 2) PCA
    pca = PCA(n_components=args.pca_dim)
    X_pca = pca.fit_transform(X)

    # 3) GMM
    gmm = GaussianMixture(n_components=args.n_components, covariance_type="diag", random_state=1)
    gmm.fit(X_pca)

    # 4) Hard cluster assignment -> cluster indicator matrix C
    cluster_ids = gmm.predict(X_pca)  # shape: (n_samples,)
    C = np.zeros((n_samples, args.n_components), dtype=int)
    C[np.arange(n_samples), cluster_ids] = 1  # one-hot per sample

    # 5) Build cost matrix M[k, s] = FN(C[:,k] -> Y[:,s])
    M = np.zeros((args.n_components, args.n_components), dtype=int)
    for k in range(args.n_components):
        pred_col = C[:, k]
        for s in range(args.n_components):
            true_col = Y[:, s]
            M[k, s] = false_negatives(true_col, pred_col)

    # 6) Hungarian assignment: component k -> substrate s
    row_ind, col_ind = linear_sum_assignment(M)
    # row_ind is [0..K-1], col_ind gives matched substrate index for each component
    # mapping[k] = s
    mapping = [int(col_ind[k]) for k in range(args.n_components)]

    # 7) Save models and mapping
    with open(os.path.join(args.output_dir, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(args.output_dir, "gmm_model.pkl"), "wb") as f:
        pickle.dump(gmm, f)
    with open(os.path.join(args.output_dir, "component_to_substrate.json"), "w", encoding="utf-8") as f:
        json.dump({"mapping": mapping}, f, indent=2)

    # Optional: Save the cost matrix for inspection
    np.savetxt(os.path.join(args.output_dir, "assignment_cost_matrix.csv"), M, fmt="%d", delimiter=",")

    print("✅ Training done.")
    print(f"Saved to: {args.output_dir}")
    print(f"Component→Substrate mapping (component index -> substrate index): {mapping}")


if __name__ == "__main__":
    main()
