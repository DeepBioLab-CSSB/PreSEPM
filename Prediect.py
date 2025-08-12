#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict enzyme–substrate probabilities with a trained PCA+GMM and a fixed component→substrate mapping.

Workflow:
1) Load PCA and GMM
2) Transform features by PCA
3) Predict responsibilities (probabilities) with GMM, shape: n_samples x n_components
4) Reorder columns by saved mapping so that columns align to substrate order
5) Save CSV probability matrix: rows=enzymes, cols=substrates (0..n_substrates-1)

Example:
  python Predict.py \
      --feature_path data/test_feats.txt \
      --model_dir results \
      --output_file results/prob_matrix.csv
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Predict probabilities using trained PCA+GMM and mapping")
    ap.add_argument("--feature_path", type=str, required=True,
                    help="Path to feature file (tab-delimited TXT), shape n_samples x d")
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Directory containing pca_model.pkl, gmm_model.pkl, component_to_substrate.json")
    ap.add_argument("--output_file", type=str, required=True,
                    help="Path to save probability matrix CSV")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load models
    with open(os.path.join(args.model_dir, "pca_model.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(args.model_dir, "gmm_model.pkl"), "rb") as f:
        gmm = pickle.load(f)
    with open(os.path.join(args.model_dir, "component_to_substrate.json"), "r", encoding="utf-8") as f:
        mapping = json.load(f)["mapping"]  # list of length K: mapping[k] = substrate_index

    # Load features and transform
    X = np.loadtxt(args.feature_path, delimiter="\t")
    X_pca = pca.transform(X)

    # Predict probabilities (responsibilities) per component
    probs = gmm.predict_proba(X_pca)  # shape: n_samples x K (component order)

    # Reorder columns so that col j corresponds to substrate j
    # mapping[k] = s  => component k should go to column s
    K = probs.shape[1]
    if K != len(mapping):
        raise ValueError(f"GMM components ({K}) != mapping length ({len(mapping)})")
    reordered = np.zeros_like(probs)
    for k, s in enumerate(mapping):
        reordered[:, s] = probs[:, k]

    # Save probability matrix
    df = pd.DataFrame(reordered, columns=[f"Substrate_{j}" for j in range(K)])
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    df.to_csv(args.output_file, index=False)

    print(f"✅ Probability matrix saved to: {args.output_file}")


if __name__ == "__main__":
    main()
