# Student Academic Prediction using Ensemble Learning

A full-stack web application that predicts whether a student will "Pass" or "Fail" based on their academic metrics. It utilizes 5 advanced machine learning models (Logistic Regression, Decision Tree, KNN, Random Forest, Gradient Boosting) built with scikit-learn.

## Features
- **🔮 Ensemble Learning Models**: Predict outcomes using 5 distinct trained models.
- **📈 Data Insights Hub**: Visualizes Pass/Fail distribution, model accuracy comparisons, and feature importance.
- **🎓 AI Academic Advisor**: Rule-based recommendation engine offering tips and critical actions depending on the student's metrics.
- **🔬 Explainable AI (SHAP-style)**: Visual bar chart breakdown showing how each individual feature contributed to the final prediction.
- **📱 Responsive Glassmorphism UI**: High-performance CSS dark theme designed seamlessly for desktop and mobile touch devices.
- **📜 Prediction History**: Built-in record logger synced via Supabase PostgreSQL.

## Technology Stack
- **Frontend**: HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Backend / API**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Database**: Supabase (PostgreSQL)

## Getting Started

### 1. Environment Setup
Make sure you have Python 3 installed. Rename the sample environment file:
```bash
cp .env.example .env
```
Fill in the `.env` file with your Supabase credentials.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```
*The app will automatically pre-load the serialized ML `.pkl` models and serve the application on http://127.0.0.1:5000*
