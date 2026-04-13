import os
import pickle
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from supabase_config import supabase
from datetime import datetime

FLASK_URL = "http://127.0.0.1:5000"

def fetch_all_data():
    """Paginate through Supabase to fetch ALL rows, bypassing the 1000-row limit."""
    all_data = []
    page_size = 1000
    offset = 0

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
        print(f"  Fetched rows {offset + 1}–{offset + len(batch)} (total so far: {len(all_data)})")
        if len(batch) < page_size:
            break          # last page
        offset += page_size

    return pd.DataFrame(all_data)

def save_model(model, filename):
    if not os.path.exists('models'):
        os.makedirs('models')
    filepath = os.path.join('models', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved {filename}")

def update_performance(model_name, accuracy, trained_on):
    data = {
        "model_name":   model_name,
        "accuracy":     float(accuracy * 100),
        "last_trained": datetime.now().isoformat()
        # trained_on not stored in DB — tracked via models/dataset_size.txt
    }
    existing = supabase.table('model_performance').select('*').eq('model_name', model_name).execute()
    if existing.data:
        supabase.table('model_performance').update(data).eq('model_name', model_name).execute()
    else:
        supabase.table('model_performance').insert(data).execute()
    print(f"  {model_name}: {accuracy*100:.2f}% (trained on {trained_on} rows)")

def main():
    print("=" * 55)
    print("Fetching ALL records from Supabase (paginated)...")
    df = fetch_all_data()

    if df.empty:
        print("No data available to train.")
        return

    total_rows = len(df)
    print(f"\nTotal records fetched: {total_rows}")

    features = ['attendance', 'study_hours', 'internal_marks', 'assignments', 'previous_gpa']
    X = df[features]
    y = df['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}\n")

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), "logistic.pkl"),
        "Decision Tree":       (DecisionTreeClassifier(random_state=42), "decision_tree.pkl"),
        "KNN":                 (KNeighborsClassifier(), "knn.pkl"),
        "Random Forest":       (RandomForestClassifier(random_state=42), "random_forest.pkl"),
        "Gradient Boosting":   (GradientBoostingClassifier(random_state=42), "gradient_boost.pkl")
    }

    print("Training models...")
    for name, (model, filename) in models.items():
        print(f"  > {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        save_model(model, filename)
        update_performance(name, acc, total_rows)

    # Save dataset size watermark
    with open('models/dataset_size.txt', 'w') as f:
        f.write(str(total_rows))
    print(f"\nSaved dataset size watermark: {total_rows} records")

    # Hot-reload models in the running Flask server (no restart needed)
    try:
        r = requests.post(f"{FLASK_URL}/reload-models", timeout=5)
        if r.ok:
            print(f"Live server hot-reloaded: {r.json().get('reloaded')}")
        else:
            print("Could not hot-reload server (not running?). Restart app.py manually.")
    except Exception:
        print("Flask server not reachable for hot-reload — restart app.py to apply new models.")

    print("=" * 55)
    print("Training complete! UI will reflect live data on next refresh.")

if __name__ == "__main__":
    main()
