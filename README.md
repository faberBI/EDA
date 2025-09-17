Ecco la versione tradotta in **inglese** e aggiornata con le nuove funzionalità richieste:

---

# 🔎 Exploratory Data Analysis & AutoML App

This application allows you to upload a dataset (CSV or Excel), perform **Exploratory Data Analysis (EDA)**, and then train **Machine Learning models** automatically and interactively using **Streamlit**.

---

## ✨ Features

### 🧮 Exploratory Data Analysis (EDA)

* Upload dataset (`.csv`, `.xlsx`)
* Select the **target variable (y)** from the available columns
* **Univariate analysis** (numerical and categorical distributions)
* **Bivariate analysis** (numerical vs target, categorical vs target)
* **Correlation matrix**
* **PCA Analysis** (dimensionality reduction)
* **Clustering** with KMeans and DBSCAN (with Silhouette Score)
* **Normality test (Shapiro-Wilk)**
* Download the dataset enriched with cluster labels

### 🤖 AutoML

* Automatic detection of the **problem type** (Classification or Regression)

* Customizable **Train / Validation / Test split** via slider

* Automatic **Feature Selection** with `SelectKBest`

* Train the following models (user-selectable):

  * Random Forest
  * Gradient Boosting
  * XGBoost
  * LightGBM
  * CatBoost

* **Customizable evaluation metric** selection for model training

* **Classification model calibration** (e.g., Platt scaling, isotonic regression)

* Evaluation on **Train, Validation, and Test sets** to detect overfitting

* Evaluation metrics:

  * **Classification:** Accuracy, Precision, Recall, F1, AUC, Brier Score, Expected Calibration Error (ECE)
  * **Regression:** RMSE, MAE, R²

* **Interactive visualizations**:

  * Distributions, scatter plots, heatmaps
  * Comparative chart of model metrics
  * Scatter plot to assess overfitting (Train vs Test)

### 🔮 GPT Insight

* Automatic generation of a **professional report** with **GPT** that analyzes the results, highlights overfitting, and selects the *best model*.

---

## 📦 Requirements

Install the necessary packages with:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```txt
streamlit>=1.28.0
pandas>=2.0.3
numpy>=1.25.0
matplotlib>=3.7.2
seaborn>=0.12.2
scipy>=1.11.1
scikit-learn>=1.3.0
xgboost>=1.7.6
lightgbm>=4.0.0
catboost>=1.2
joblib>=1.3.2
openpyxl>=3.1.2
openai>=1.12.0
```

---

## 🚀 Run the App

Launch the application with:

```bash
streamlit run app.py
```
