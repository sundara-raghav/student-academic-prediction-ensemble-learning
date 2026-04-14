"""
preprocess.py
-------------
Standalone preprocessing pipeline used by train_model.py.

Steps applied (in order):
  1. Drop exact duplicate rows
  2. Impute missing values  (median for numerics)
  3. Remove outliers via IQR method
  4. Feature engineering   (academic_score, study_efficiency)
  5. StandardScaler normalisation
  6. SMOTE to balance classes
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ── Column sets ────────────────────────────────────────────────────────────────
BASE_FEATURES      = ['attendance', 'study_hours', 'internal_marks',
                       'assignments', 'previous_gpa']
ENGINEERED_FEATURES = ['academic_score', 'study_efficiency']
ALL_FEATURES        = BASE_FEATURES + ENGINEERED_FEATURES
TARGET              = 'result'


# ── 1. Duplicate removal ───────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after  = len(df)
    print(f"  [dedup]  Removed {before - after} duplicate rows  ->  {after} rows remain")
    return df.reset_index(drop=True)


# ── 2. Missing-value imputation ────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum().sum()
    if missing:
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        print(f"  [impute] Imputed {missing} missing cell(s) with column medians")
    else:
        print(f"  [impute] No missing values found")
    return df


# ── 3. IQR outlier removal (feature columns only) ─────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    before = len(df)
    mask   = pd.Series([True] * len(df), index=df.index)

    for col in cols:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask &= df[col].between(lower, upper)

    df    = df[mask].reset_index(drop=True)
    after = len(df)
    print(f"  [IQR]    Removed {before - after} outlier rows  ->  {after} rows remain")
    return df


# ── 4. Feature engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # academic_score: average of continuous academic performance measures
    df['academic_score'] = (df['internal_marks'] + df['assignments']) / 2.0

    # study_efficiency: study output relative to attendance
    # Clip attendance to avoid division by zero or absurd values
    df['study_efficiency'] = df['study_hours'] / df['attendance'].clip(lower=1)

    print(f"  [feat]   Created: academic_score, study_efficiency")
    return df


# ── 5. StandardScaler normalisation ───────────────────────────────────────────
def scale_features(X_train: pd.DataFrame,
                   X_test:  pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    print(f"  [scale]  StandardScaler applied  (features: {list(X_train.columns)})")
    return X_train_sc, X_test_sc, scaler


# ── 6. SMOTE oversampling ──────────────────────────────────────────────────────
def apply_smote(X_train: pd.DataFrame,
                y_train: pd.Series,
                random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    counts_before = y_train.value_counts().to_dict()
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    counts_after = pd.Series(y_res).value_counts().to_dict()
    print(f"  [SMOTE]  Before: {counts_before}  ->  After: {counts_after}")
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


# ── Master pipeline ────────────────────────────────────────────────────────────
def run_preprocessing_pipeline(df: pd.DataFrame,
                                test_size: float = 0.2,
                                random_state: int = 42):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : raw DataFrame fetched from Supabase (includes 'result' column)

    Returns
    -------
    X_train, X_test, y_train, y_test : processed arrays ready for training
    scaler                            : fitted StandardScaler (for inference)
    feature_names                     : list[str] of feature columns used
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 55)
    print("PREPROCESSING PIPELINE")
    print("=" * 55)
    print(f"  [start]  Raw dataset shape: {df.shape}")

    # Keep only relevant columns
    cols_needed = BASE_FEATURES + [TARGET]
    df = df[[c for c in cols_needed if c in df.columns]].copy()

    # Steps 1–3
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = remove_outliers_iqr(df, BASE_FEATURES)

    # Class distribution info
    counts = df[TARGET].value_counts().to_dict()
    print(f"  [class]  Distribution -> {counts}")

    # Step 4: feature engineering (before split to share stats)
    df = engineer_features(df)

    # Separate X / y
    X = df[ALL_FEATURES]
    y = df[TARGET]

    print(f"  [split]  Features used: {ALL_FEATURES}")

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  [split]  Train={len(X_train)}  Test={len(X_test)}")

    # Step 5: scale
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Step 6: SMOTE (only on training data!)
    X_train, y_train = apply_smote(X_train, y_train, random_state=random_state)

    print("=" * 55)
    print(f"  [done]   X_train={X_train.shape}  X_test={X_test.shape}\n")

    return X_train, X_test, y_train, y_test, scaler, ALL_FEATURES
