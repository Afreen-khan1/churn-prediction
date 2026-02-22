# Customer Churn Prediction – Enterprise ML Application

A production-ready, end-to-end machine learning web application that predicts customer churn from uploaded Excel files. It automatically cleans data, trains multiple models, and visualizes results through a professional dark-theme dashboard.  
Built to demonstrate enterprise-grade AI/ML engineering and full-stack deployment skills.

> **Note:** Deployment is in progress. A live demo link will be added here soon.

---

## 📌 Features

- **Automated Data Cleaning** – Missing value handling, type corrections, column removals, feature engineering.  
- **Exploratory Data Analysis (EDA)** with three key visualizations:
  - Churn distribution (bar chart)
  - Churn by contract type
  - Correlation heatmap of numerical features
- **Machine Learning Model Training** – Compares:
  - Logistic Regression  
  - Decision Tree Classifier  
  - Random Forest Classifier  
- **Performance Comparison Table** – Accuracy, Precision, Recall, F1-Score, ROC AUC  
- **Interactive Dashboard** – Plotly histogram for churn across contract types  
- **Dark Professional UI** – Custom CSS, animations, hover effects, responsive layout  
- **Downloadable Cleaned Dataset** (Excel format)  
- **Production-Ready Model Export** – Saves best model as `churn_model.pkl`

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| **Backend** | Python, Flask, Gunicorn |
| **Machine Learning** | scikit-learn, pandas, numpy, joblib |
| **Visualization** | matplotlib, seaborn, plotly |
| **Frontend** | HTML5, CSS3 (dark theme), Jinja2 templates |
| **Deployment** | Render (planned) or any Python-enabled cloud |

---

## 📂 Project Structure

```
customer-churn-project/
│
├── app.py                     # Flask application entry point
├── churn_analysis.py          # Core ML pipeline (cleaning, training, graphing)
├── churn_model.pkl            # Saved best model (created after first training)
├── requirements.txt           # Python dependencies
├── Procfile                   # For Render deployment
├── runtime.txt                # Python version (3.12.10)
│
├── templates/                 # HTML templates
│   ├── index.html             # Upload page
│   └── results.html           # Results dashboard
│
└── static/                    # CSS + images
    ├── style.css              # Dark theme design
    └── graphs/                # Auto-generated graphs at runtime
```

---

## ⚙️ Local Setup

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### **2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# OR
venv\Scripts\activate           # Windows
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Flask app**
```bash
python app.py
```

Now open:  
👉 http://127.0.0.1:5000  
Upload your Excel file to generate predictions and dashboards.

---

## 📊 Sample Data Format

Your Excel file should contain typical telecom churn columns such as:

| CustomerID | Contract | Tenure Months | Monthly Charges | Total Charges | Churn Label | … |
|------------|----------|----------------|------------------|----------------|--------------|---|
| 0001-ABCD  | Month-to-month | 2 | 65.5 | 131.0 | Yes | … |
| 0002-EFGH  | Two year | 45 | 95.2 | 4284.0 | No | … |

**Required Columns:**  
`CustomerID`, `Contract`, `Tenure Months`, `Monthly Charges`, `Total Charges`, `Churn Label`

---

##  How It Works (High-Level)

1. **Upload** → User provides an Excel file.  
2. **Clean** → Missing values imputed; irrelevant columns removed.  
3. ** Feature Engineering** → Creates new features like  
   `AvgMonthlySpend = Total Charges / Tenure Months`  
4. **Visualize** → Matplotlib generates 3 graphs saved under `/static/graphs`.  
5. **Train** → Three models are trained and evaluated.  
6. **Export** → Best model saved as `churn_model.pkl`.  
7. **Display** → Results shown in a dark-theme dashboard with interactive Plotly charts.

---

## 📈 Model Performance (Example)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|----------|
| Logistic Regression | 0.9787 | 1.0000 | 0.9250 | 0.9610 | 0.9625 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

(Your results may vary depending on dataset.)

---

## 🧪 Future Enhancements

- Add advanced ML models (XGBoost, LightGBM)  
- Hyperparameter tuning (GridSearch / Optuna)  
- REST API for real-time predictions  
- Integration with PostgreSQL / MongoDB  
- User authentication system  
- Docker deployment on Google Cloud Run  

---

## 🤝 Contributing

Contributions, feature requests, and issues are welcome.  
Feel free to check the **Issues** tab.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## Author

**Afreen Ahamed**  
GitHub: **Afreen-khan1**  
