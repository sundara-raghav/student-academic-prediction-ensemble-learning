import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from supabase_config import supabase
from datetime import datetime

def fetch_data():
    # Fetch all records (Handling pagination if > 1000, but standard limit is 1000 so it should fetch all)
    response = supabase.table('students_dataset').select('*').limit(2000).execute()
    data = response.data
    df = pd.DataFrame(data)
    return df

def save_model(model, filename):
    if not os.path.exists('models'):
        os.makedirs('models')
    filepath = os.path.join('models', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {filename}")

def update_performance(model_name, accuracy):
    # Upsert the performance
    data = {
        "model_name": model_name,
        "accuracy": float(accuracy * 100), # Store as percentage
        "last_trained": datetime.now().isoformat()
    }
    
    # Check if exists
    existing = supabase.table('model_performance').select('*').eq('model_name', model_name).execute()
    if existing.data:
        supabase.table('model_performance').update(data).eq('model_name', model_name).execute()
    else:
        supabase.table('model_performance').insert(data).execute()
    print(f"Updated performance for {model_name} with {accuracy*100:.2f}%")

def main():
    print("Fetching data from DB...")
    df = fetch_data()
    if df.empty:
        print("No data available to train.")
        return
        
    features = ['attendance', 'study_hours', 'internal_marks', 'assignments', 'previous_gpa']
    X = df[features]
    y = df['result']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), "logistic.pkl"),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), "decision_tree.pkl"),
        "KNN": (KNeighborsClassifier(), "knn.pkl"),
        "Random Forest": (RandomForestClassifier(random_state=42), "random_forest.pkl"),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), "gradient_boost.pkl")
    }
    
    for name, (model, filename) in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        save_model(model, filename)
        update_performance(name, acc)
        
if __name__ == "__main__":
    main()
