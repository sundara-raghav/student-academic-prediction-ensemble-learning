import os
import pickle
import threading
import numpy as np
import warnings
from flask import Flask, request, jsonify, render_template
from supabase_config import supabase
from datetime import datetime

warnings.filterwarnings('ignore', message='X does not have valid feature names')

app = Flask(__name__)

MODEL_FILES = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boost.pkl"
}

FEATURE_NAMES = ['Attendance', 'Study Hours', 'Internal Marks', 'Assignments', 'Previous GPA']
BASELINE = [75.0, 5.0, 30.0, 30.0, 7.0]

loaded_models = {}

def get_model(model_name):
    return loaded_models.get(model_name)

# ─── Background Supabase logger (fire-and-forget) ────────────────────────────
def _log_prediction_async(log_data):
    """Insert prediction log in a background thread so /predict returns immediately."""
    try:
        supabase.table('prediction_logs').insert(log_data).execute()
    except Exception as e:
        print(f"  [log] background insert failed: {e}")

# ─── Rule-Based AI Academic Advisor ─────────────────────────────────────────
def generate_advice(attendance, study_hours, internal_marks, assignments, previous_gpa, prediction):
    tips = []
    urgent = []

    if attendance < 60:
        urgent.append("⚠️ Critical: Attendance below 60% — risk of being barred from exams.")
    elif attendance < 75:
        tips.append("📅 Improve attendance above 75% to meet minimum eligibility criteria.")

    if study_hours < 2:
        urgent.append("⚠️ You study less than 2 hrs/day — double your study time immediately.")
    elif study_hours < 3:
        tips.append("📚 Increase study hours to at least 3+ hours/day for steady improvement.")

    if internal_marks < 20:
        urgent.append("⚠️ Internal marks critically low (below 20/50) — seek teacher support.")
    elif internal_marks < 35:
        tips.append("📝 Target 35+ in internal exams — focus on core subjects and past papers.")

    if assignments < 20:
        urgent.append("⚠️ Assignment score very low (below 20/50) — complete all pending work.")
    elif assignments < 35:
        tips.append("📋 Submit all assignments on time to push score above 35/50.")

    if previous_gpa < 5.0:
        urgent.append("⚠️ Previous GPA below 5.0 — consider remedial classes or tutoring.")
    elif previous_gpa < 7.0:
        tips.append("🎯 Aim to raise GPA above 7.0 through consistent semester performance.")

    if prediction == "PASS" and not tips and not urgent:
        tips.append("🌟 Excellent profile! Keep maintaining this performance level.")
        tips.append("🚀 Consider participating in academic competitions or projects.")

    return {"urgent": urgent, "tips": tips}

# ─── SHAP-style local explanation ────────────────────────────────────────────
def compute_shap_like(model, features_arr):
    if not hasattr(model, 'predict_proba'):
        return [0.0] * len(FEATURE_NAMES)
    full_proba = model.predict_proba(features_arr)[0][1]
    contribs = []
    for i in range(len(FEATURE_NAMES)):
        perturbed = features_arr[0].copy()
        perturbed[i] = BASELINE[i]
        p = model.predict_proba(np.array([perturbed]))[0][1]
        contribs.append(round(float(full_proba - p) * 100, 2))
    return contribs

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    try:
        response = supabase.table('model_performance').select('*').order('accuracy', desc=True).execute()
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data           = request.json
    model_name     = data.get('model_name')
    attendance     = float(data.get('attendance', 0))
    study_hours    = float(data.get('study_hours', 0))
    internal_marks = float(data.get('internal_marks', 0))
    assignments    = float(data.get('assignments', 0))
    previous_gpa   = float(data.get('previous_gpa', 0))

    model = get_model(model_name)
    if not model:
        return jsonify({"error": "Model not found or not loaded"}), 404

    features = np.array([[attendance, study_hours, internal_marks, assignments, previous_gpa]])

    try:
        prediction  = model.predict(features)[0]
        result_text = "PASS" if prediction == 1 else "FAIL"

        if hasattr(model, 'predict_proba'):
            proba      = model.predict_proba(features)[0]
            confidence = float(max(proba) * 100)
            pass_prob  = float(proba[1] * 100)
            fail_prob  = float(proba[0] * 100)
        else:
            confidence = 100.0
            pass_prob  = 100.0 if result_text == "PASS" else 0.0
            fail_prob  = 100.0 - pass_prob

        # Compute SHAP and advice (pure Python, very fast)
        shap_contribs = compute_shap_like(model, features)
        advice        = generate_advice(attendance, study_hours, internal_marks,
                                        assignments, previous_gpa, result_text)

        # Fire-and-forget Supabase insert — does NOT block the response
        log_data = {
            "model_used": model_name, "attendance": int(attendance),
            "study_hours": study_hours, "internal_marks": int(internal_marks),
            "assignments": int(assignments), "previous_gpa": previous_gpa,
            "prediction": result_text, "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        threading.Thread(target=_log_prediction_async, args=(log_data,), daemon=True).start()

        # Return immediately — no waiting for DB
        return jsonify({
            "prediction": result_text, "confidence": round(confidence, 2),
            "pass_prob": round(pass_prob, 2), "fail_prob": round(fail_prob, 2),
            "model_used": model_name,
            "shap": {"labels": FEATURE_NAMES, "contributions": shap_contribs},
            "advice": advice
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        response = supabase.table('prediction_logs').select('*').order('timestamp', desc=True).limit(15).execute()
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history-page')
def history_page():
    return render_template('history.html')

@app.route('/history-all', methods=['GET'])
def get_history_all():
    try:
        response = supabase.table('prediction_logs').select('*').order('timestamp', desc=True).limit(1000).execute()
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    model = get_model("Random Forest")
    if model and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_.tolist()
    else:
        importances = [0.2, 0.2, 0.2, 0.2, 0.2]
    return jsonify({"labels": FEATURE_NAMES, "data": [round(i * 100, 2) for i in importances]})

@app.route('/stats', methods=['GET'])
def get_stats():
    # Read trained-on watermark
    trained_size = None
    if os.path.exists('models/dataset_size.txt'):
        with open('models/dataset_size.txt', 'r') as f:
            try: trained_size = int(f.read().strip())
            except: pass

    try:
        # 1. Get TOTAL records count directly from the table (doesn't care about columns)
        resp_total = supabase.table('students_dataset').select('id', count='exact').execute()
        total_records = resp_total.count or 0

        # 2. Get counts for Pass/Fail distribution (rows where result is defined)
        resp_pass = supabase.table('students_dataset').select('id', count='exact').eq('result', 1).execute()
        resp_fail = supabase.table('students_dataset').select('id', count='exact').eq('result', 0).execute()
        
        total_p = resp_pass.count or 0
        total_f = resp_fail.count or 0

        # If we have a watermark, the number of records added SINCE training
        # is the delta between the table's absolute size and the watermark.
        untrained_count = 0
        if trained_size is not None:
            untrained_count = max(0, total_records - trained_size)

        result = {
            "total":     total_records,
            "pass":      total_p,
            "fail":      total_f,
            "trained":   trained_size if trained_size is not None else total_records,
            "untrained": untrained_count
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reload-models', methods=['POST'])
def reload_models():
    """Hot-reload all pkl files so the live server picks up freshly trained models."""
    reloaded = []
    for name, filename in MODEL_FILES.items():
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                loaded_models[name] = pickle.load(f)
            reloaded.append(name)
            print(f"  [reload] {name} refreshed")
    return jsonify({"reloaded": reloaded, "count": len(reloaded)})


# ─── Startup: pre-load all models before serving requests ────────────────────
def _preload_models():
    for name, filename in MODEL_FILES.items():
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                loaded_models[name] = pickle.load(f)
            print(f"  [preload] {name} ready")

print("[startup] Pre-loading ML models...")
_preload_models()
print("[startup] All models loaded. Server ready.")

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)
