import argparse
import sys
import warnings
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def _extract_numeric_features(df: pd.DataFrame, label_column: Optional[str]) -> pd.DataFrame:
    """
    Extract and return only the numeric feature columns from the input DataFrame.
    If a label column is specified, it will be excluded from the returned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and, optionally, a label column.
    label_column : str or None
        Name of the label column. If None, all numeric columns are returned.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only numeric feature columns.

    Raises
    ------
    ValueError
        If the label column is specified but not found in the DataFrame.
        If non-numeric feature columns are present and have not been encoded.
    """
    if label_column is not None and label_column not in df.columns:
        raise ValueError(f"label_column='{label_column}' not found in columns: {list(df.columns)}")
    features = df.drop(columns=[label_column]) if label_column else df.copy()
    numeric_features = features.select_dtypes(include=[np.number])
    if numeric_features.shape[1] != features.shape[1]:
        non_numeric = set(features.columns) - set(numeric_features.columns)
        raise ValueError(
            f"Non-numeric columns detected: {sorted(non_numeric)}. "
            "Please encode categorical variables prior to applying SMOTE."
        )
    return numeric_features


def _adjust_k_neighbors(y: np.ndarray, k: int) -> int:
    """
    Adjust the k_neighbors parameter to ensure it is strictly less than
    the sample count of the smallest class.

    Parameters
    ----------
    y : np.ndarray
        Array of class labels.
    k : int
        Desired k_neighbors parameter.

    Returns
    -------
    int
        Adjusted k_neighbors parameter.

    Raises
    ------
    ValueError
        If any class contains fewer than two samples, as SMOTE cannot operate in such cases.
    """
    _, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()
    if min_class_count <= 1:
        raise ValueError(
            "At least one class contains only one sample; SMOTE requires a minimum of two samples per minority class."
        )
    if k >= min_class_count:
        new_k = max(1, min_class_count - 1)
        if new_k < k:
            warnings.warn(
                f"k_neighbors={k} adjusted to {new_k} to satisfy SMOTE constraints "
                f"(minority class size: {min_class_count}).",
                UserWarning,
            )
        return new_k
    return k


def augment_with_smote(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
    sampling_strategy: "str|float|dict" = "auto",
    k_neighbors: int = 5,
    random_state: Optional[int] = 42,
    generate_demo_labels: Optional[float] = None,
    scale_features: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply SMOTE to synthetically augment imbalanced datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and a label column.
    label_column : str or None
        Name of the label column. If None and generate_demo_labels is provided,
        synthetic labels will be generated for demonstration purposes.
    sampling_strategy : {"auto", float, dict}
        Strategy for sampling the minority class(es). See imbalanced-learn documentation.
    k_neighbors : int
        Number of nearest neighbors used by SMOTE. Adjusted automatically if necessary.
    random_state : int or None
        Random seed for reproducibility.
    generate_demo_labels : float or None
        If provided, generates labels with the specified minority class proportion (0 < proportion < 1).
    scale_features : bool
        If True, standardizes features prior to applying SMOTE.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - Synthetic samples only, with an additional '__is_generated__' column set to 1.
        - Full dataset (original + synthetic), with '__is_generated__' column indicating sample origin.
    """
    X_numeric = _extract_numeric_features(df, label_column)

    # Label preparation
    if label_column is None:
        if generate_demo_labels is None:
            raise ValueError(
                "label_column is None and generate_demo_labels not provided. "
                "Specify a real label column or use generate_demo_labels for demonstration."
            )
        proportion = float(generate_demo_labels)
        if not (0.0 < proportion < 1.0):
            raise ValueError("generate_demo_labels must be in (0,1), e.g., 0.2 for 20% minority class.")
        n_samples = len(df)
        n_minority = max(1, int(round(proportion * n_samples)))
        labels = np.array([0] * (n_samples - n_minority) + [1] * n_minority)
        rng = np.random.default_rng(random_state)
        rng.shuffle(labels)
        label_series = pd.Series(labels, name="__demo_label__")
        label_used = "__demo_label__"
        df = df.copy()
        df[label_used] = label_series
    else:
        label_used = label_column

    y_array = df[label_used].values
    k_neighbors = _adjust_k_neighbors(y_array, k_neighbors)

    # Optional feature scaling
    if scale_features:
        scaler = StandardScaler()
        X_for_smote = scaler.fit_transform(X_numeric.values)
    else:
        X_for_smote = X_numeric.values

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    X_resampled, y_resampled = smote.fit_resample(X_for_smote, y_array)

    # Construct resampled DataFrame
    feature_columns = X_numeric.columns.tolist()
    df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
    df_resampled[label_used] = y_resampled

    # Flag synthetic samples
    n_original = len(df)
    n_total = len(df_resampled)
    synthetic_mask = np.array([0] * n_original + [1] * (n_total - n_original))
    df_resampled["__is_generated__"] = synthetic_mask

    # Merge auxiliary columns from the original DataFrame
    auxiliary_columns = [c for c in df.columns if c not in feature_columns and c != label_used]
    for col in auxiliary_columns:
        merged_values = pd.concat(
            [df[col].reset_index(drop=True), pd.Series([np.nan] * (n_total - n_original))],
            axis=0,
            ignore_index=True,
        )
        df_resampled[col] = merged_values

    # Extract only synthetic samples
    df_synthetic_only = df_resampled[df_resampled["__is_generated__"] == 1].reset_index(drop=True)

    return df_synthetic_only, df_resampled


def main():
    parser = argparse.ArgumentParser(
        description="SMOTE-based augmentation of imbalanced datasets, exporting either synthetic samples or the full resampled dataset."
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    parser.add_argument("--label-col", default=None, help="Name of the label column in the dataset.")
    parser.add_argument(
        "--output-mode",
        choices=["generated", "resampled"],
        default="generated",
        help="'generated': export only synthetic samples; 'resampled': export both original and synthetic samples.",
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        help="SMOTE sampling_strategy parameter (e.g., 'auto', 0.5, or a Python dict).",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors for SMOTE.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--generate-demo-labels",
        type=float,
        default=None,
        help="Proportion of minority class for synthetic label generation (0 < proportion < 1).",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Standardize features prior to applying SMOTE.",
    )

    args = parser.parse_args()

    # Parse and validate sampling_strategy
    sampling_strategy = args.strategy
    try:
        if sampling_strategy not in ("auto",):
            sampling_strategy = float(sampling_strategy)
    except ValueError:
        pass

    df = pd.read_csv(args.input)
    df_synthetic, df_resampled = augment_with_smote(
        df=df,
        label_column=args.label_col,
        sampling_strategy=sampling_strategy,
        k_neighbors=args.k,
        random_state=args.random_state,
        generate_demo_labels=args.generate_demo_labels,
        scale_features=args.scale,
    )

    if args.output_mode == "generated":
        df_synthetic.to_csv(args.output, index=False)
    else:
        df_resampled.to_csv(args.output, index=False)

    print(f"Processing complete. Output saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
