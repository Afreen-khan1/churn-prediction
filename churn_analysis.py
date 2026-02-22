import matplotlib
matplotlib.use('Agg')  # Force Matplotlib to use a non-GUI backend
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import plotly.io as pio 
pio.renderers.default = "json"
import plotly.express as px
import os
os.makedirs("static/graphs", exist_ok=True)
def save_graph(fig, filename):
    output_path = os.path.join("static/graphs", filename)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
### MAIN FUNCTION CALLED BY FLASK
def run_analysis(file_path):
    #----1.Load the dataset----
    df =pd.read_excel(file_path)
    #Preview data
    print(df.head())
    #Check structure and missing values
    print(df.info())
    print(df.isnull().sum())
    #----2.Data Cleaning----
    #Convert Total charges to numeric(coerce errors for missing values)
    df['Total Charges']=pd.to_numeric(df['Total Charges'],errors='coerce')
    #Fill missing TotalCharges with median
    df['Total Charges'].fillna(df['Total Charges'].median(),inplace=True)
    #Drop customerID(not useful for predictions)
    df.drop('CustomerID',axis=1,inplace=True)

    #----3.EDA  + Save Graphs----
    graph_paths=[]
    #Churn count(Plot 1)
    plt.figure(figsize=(6,4))
    sns.countplot(x='Churn Label',data=df,palette={'Yes':'red','No':'blue'})
    path1 = "static/graphs/churn_count.png"
    plt.savefig(path1)
    plt.close()
    graph_paths.append(path1)
    #Churn by contract(Plot 2)
    plt.figure(figsize=(7,4))
    sns.countplot(x='Contract',hue='Churn Label',data=df)
    path2 = "static/graphs/contact_churn.png"
    plt.savefig(path2)
    plt.close()
    graph_paths.append(path2)
    #Correlation heatmap for numerical features(Plot 3)
    plt.figure(figsize=(10,6))
    corr = df[['Zip Code','Latitude','Longitude','Tenure Months','Monthly Charges', 'Total Charges','Churn Value','Churn Score','CLTV']].corr()
    sns.heatmap(corr,annot=True, fmt='.2f',cmap='coolwarm',annot_kws={"size":8})
    plt.xticks(rotation=45,ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    path3 = "static/graphs/heatmap.png"
    plt.savefig(path3)
    plt.close()
    graph_paths.append(path3)
    #----4.Feature Engineering----
    for col in df.select_dtypes(include=['object']).columns:
        if col!='Churn Label':
            df[col]=LabelEncoder().fit_transform(df[col])
    #Encode target
    df['Churn Label']=df['Churn Label'].map({'No':0,'Yes':1})
    #Create new feature:Average Monthly Spend
    df['AvgMonthlySpend']=df['Total Charges']/(df['Tenure Months'].replace(0,1))
    #-----Handling missing values in all columns-----
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    if 'Churn Label' in categorical_cols:
        categorical_cols.remove('Churn Label')
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    if df.isnull().any().any():
        print(f"Dropping {df.isnull().any(axis=1).sum()} rows with missing values")
        df.dropna(inplace=True)


    # ---- Save cleaned dataset for website download ----
    cleaned_file_path = "static/cleaned_churn_data.xlsx"
    df.to_excel(cleaned_file_path, index=False)

    #----5.Feature Scaling----
    scaler=StandardScaler()
    num_cols=['Tenure Months','Monthly Charges','Total Charges','AvgMonthlySpend']
    df[num_cols]=scaler.fit_transform(df[num_cols])

    #----6.Train-Test Split----
    X= df.drop('Churn Label',axis=1)
    y=df['Churn Label']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    #----7.Model building and comparison----
    models = {"Logistic Regression": LogisticRegression(max_iter=3000,solver ="liblinear"),"Decision Tree": DecisionTreeClassifier(),"Random Forest": RandomForestClassifier()}
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred), 4),
            "ROC AUC": round(roc_auc_score(y_test, y_pred), 4)
        }

    # save the  best trained Random Forest for deployment
    best_model = models["Random Forest"]
    joblib.dump(best_model, "churn_model.pkl")


    #----8. Cross-Validation and Hyperparameter Tuning(Eg:Random Forest)
    param_grid = {'n_estimators': [50, 100],'max_depth': [None, 10, 20],'min_samples_split': [2, 5]}

    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
        #----9. Visualization Dashboard(Save Plotly plot as HTML)
    fig = px.histogram(df, x="Contract", color="Churn Label", barmode='group')
    histogram_path = "static/plot_contract_churn.html"
    fig.write_html(histogram_path)
    
    # Get current timestamp for display
    from datetime import datetime
    now = datetime.now()
    
    # Return to flask with ALL 5 values
    return graph_paths, results, cleaned_file_path, histogram_path, now