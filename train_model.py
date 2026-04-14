"""
train_model.py   (v2 — enhanced pipeline)
------------------------------------------
Full ML pipeline:
  1. Fetch ALL records from Supabase (paginated)
  2. Clean + preprocess (via preprocess.py)
  3. Hyperparameter tuning with GridSearchCV
  4. Train 5 classifiers + evaluate with accuracy / precision / recall / F1
  5. Print comparison table + identify best model
  6. Serialize models + scaler to disk
  7. Store performance metrics to Supabase
  8. Hot-reload the running Flask server
"""

import os
import pickle
import requests
import warnings
import numpy as np
import pandas as pd

from datetime import datetime
from tabulate import tabulate

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     classification_report)

from supabase_config import supabase
from preprocess      import run_preprocessing_pipeline

warnings.filterwarnings('ignore')

FLASK_URL  = "http://127.0.0.1:5000"
MODELS_DIR = "models"
CV_FOLDS   = 5        # cross-validation folds for GridSearchCV


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Fetch data from Supabase
# ══════════════════════════════════════════════════════════════════════════════
def fetch_all_data() -> pd.DataFrame:
    """Paginate through Supabase to fetch ALL rows, bypassing the 1000-row limit."""
    all_data  = []
    page_size = 1000
    offset    = 0

    while True:
        response = (
            supabase.table('students_dataset')
            .select('*')
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        all_data.extend(batch)
        print(f"  Fetched rows {offset + 1}–{offset + len(batch)}  "
              f"(running total: {len(all_data)})")
        if len(batch) < page_size:
            break
        offset += page_size

    return pd.DataFrame(all_data)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Model definitions + hyperparameter grids
# ══════════════════════════════════════════════════════════════════════════════
def get_model_grids() -> dict:
    """Return {name: (estimator, param_grid, pkl_filename)}."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42),
            {
                'C':       [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
            },
            "logistic.pkl"
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                'max_depth':        [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'criterion':        ['gini', 'entropy'],
            },
            "decision_tree.pkl"
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                'n_neighbors': [3, 5, 7, 11, 15],
                'weights':     ['uniform', 'distance'],
                'metric':      ['euclidean', 'manhattan'],
            },
            "knn.pkl"
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                'n_estimators':     [100, 200, 300],
                'max_depth':        [5, 10, None],
                'min_samples_split': [2, 5],
                'max_features':     ['sqrt', 'log2'],
            },
            "random_forest.pkl"
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                'n_estimators':  [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth':     [3, 5, 7],
                'subsample':     [0.8, 1.0],
            },
            "gradient_boost.pkl"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Save helpers
# ══════════════════════════════════════════════════════════════════════════════
def save_artifact(obj, filename: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  [save]   {filename}")


def update_performance_db(model_name: str, metrics: dict, trained_on: int):
    """Upsert model metrics into Supabase model_performance table."""
    data = {
        "model_name":   model_name,
        "accuracy":     round(float(metrics['accuracy'])  * 100, 4),
        "last_trained": datetime.now().isoformat(),
    }
    existing = (supabase.table('model_performance')
                .select('*').eq('model_name', model_name).execute())
    if existing.data:
        supabase.table('model_performance').update(data).eq('model_name', model_name).execute()
    else:
        supabase.table('model_performance').insert(data).execute()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Evaluate a fitted model
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "y_pred":    y_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  STUDENT ACADEMIC PERFORMANCE — ENHANCED TRAINING PIPELINE")
    print("=" * 60)

    # ── Fetch ──────────────────────────────────────────────────────────────
    print("\n[1/5] Fetching records from Supabase...")
    df = fetch_all_data()
    if df.empty:
        print("  [X] No data available. Run setup_database.py first.")
        return
    total_rows = len(df)
    print(f"  Total records: {total_rows}")

    # ── Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/5] Cleaning & preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_names = \
        run_preprocessing_pipeline(df, test_size=0.2, random_state=42)

    # Save the scaler so app.py can use it at inference time
    save_artifact(scaler, "scaler.pkl")
    # Save feature names list for the Flask app
    save_artifact(feature_names, "feature_names.pkl")

    # ── Train & tune ────────────────────────────────────────────────────────
    print("\n[3/5] Training & hyperparameter tuning (GridSearchCV)...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    model_grids = get_model_grids()
    results     = {}
    best_estimators = {}

    for name, (estimator, param_grid, pkl_file) in model_grids.items():
        print(f"\n  > {name}")
        grid_search = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"    Best params: {grid_search.best_params_}")
        print(f"    CV F1 score: {grid_search.best_score_:.4f}")

        # Evaluate on held-out test set
        metrics = evaluate_model(best_model, X_test, y_test)
        results[name] = metrics
        best_estimators[name] = best_model

        # Save to disk
        save_artifact(best_model, pkl_file)

    # ── Print comparison table ───────────────────────────────────────────────
    print("\n[4/5] Results on held-out test set (20%)")
    print("=" * 60)
    table_rows = []
    for name, m in results.items():
        table_rows.append([
            name,
            f"{m['accuracy']  * 100:.2f}%",
            f"{m['precision'] * 100:.2f}%",
            f"{m['recall']    * 100:.2f}%",
            f"{m['f1']        * 100:.2f}%",
        ])

    # Sort by accuracy descending
    table_rows.sort(key=lambda r: float(r[1].rstrip('%')), reverse=True)

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    print(tabulate(table_rows, headers=headers, tablefmt="grid"))

    # Best model
    best_name = max(results, key=lambda n: results[n]['f1'])
    best_m    = results[best_name]
    print(f"\n  [Best Model]  ->  {best_name}")
    print(f"     Accuracy  = {best_m['accuracy']  * 100:.2f}%")
    print(f"     Precision = {best_m['precision'] * 100:.2f}%")
    print(f"     Recall    = {best_m['recall']    * 100:.2f}%")
    print(f"     F1-Score  = {best_m['f1']        * 100:.2f}%")

    # Full classification report for best model
    print(f"\n  Classification report for {best_name}:")
    print(classification_report(y_test, best_m['y_pred'],
                                 target_names=["Fail", "Pass"]))

    # ── Update Supabase ─────────────────────────────────────────────────────
    print("\n[5/5] Pushing metrics to Supabase...")
    for name, metrics in results.items():
        update_performance_db(name, metrics, total_rows)
        print(f"  [OK] {name}: {metrics['accuracy']*100:.2f}%")

    # Save dataset watermark
    with open(os.path.join(MODELS_DIR, 'dataset_size.txt'), 'w') as f:
        f.write(str(total_rows))
    print(f"  Saved watermark: {total_rows} records")

    # ── Hot-reload Flask (if running) ───────────────────────────────────────
    try:
        r = requests.post(f"{FLASK_URL}/reload-models", timeout=5)
        if r.ok:
            print(f"\n  Flask hot-reload: {r.json().get('reloaded')}")
        else:
            print("\n  Flask not running — restart app.py to apply new models.")
    except Exception:
        print("\n  Flask not reachable — restart app.py to apply new models.")

    print("\n" + "=" * 60)
    print("  [DONE]  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
