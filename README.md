# 🔎 Exploratory Data Analysis & AutoML App

Questa applicazione consente di caricare un dataset (CSV o Excel) ed eseguire un'**Analisi Esplorativa dei Dati (EDA)** e successivamente addestrare modelli di **Machine Learning** in maniera automatica e interattiva tramite **Streamlit**.

---

## ✨ Funzionalità

### 🧮 Exploratory Data Analysis (EDA)

* Caricamento dataset (`.csv`, `.xlsx`)
* Scelta della variabile **target (y)** tra le colonne disponibili
* Analisi **univariata** (distribuzioni numeriche e categoriche)
* Analisi **bivariata** (numeriche vs target, categoriche vs target)
* **Matrice di correlazione**
* **PCA Analysis** (riduzione dimensionale)
* **Clustering** con KMeans e DBSCAN (con Silhouette Score)
* **Test di normalità (Shapiro-Wilk)**
* Download del dataset arricchito con i cluster

### 🤖 AutoML

* Identificazione automatica del **tipo di problema** (Classificazione o Regressione)
* Suddivisione del dataset in **Train / Validation / Test** con percentuale personalizzabile via slider
* **Feature Selection** automatica con `SelectKBest`
* Addestramento dei seguenti modelli (selezionabili dall’utente):

  * Random Forest
  * Gradient Boosting
  * XGBoost
  * LightGBM
  * CatBoost
* Valutazione su **Train, Validation e Test set** per identificare l’overfitting
* Metriche di valutazione:

  * **Classificazione:** Accuracy, Precision, Recall, F1, AUC, Brier Score, ECE
  * **Regressione:** RMSE, MAE, R²
* **Grafici interattivi**:

  * Distribuzioni, scatter plot, heatmap
  * Grafico comparativo delle metriche dei modelli
  * Scatter plot per valutare overfitting (Train vs Test)

### 🔮 GPT Insight

* Generazione automatica di un **commento professionale** con **GPT** che analizza i risultati, evidenzia overfitting ed elegge il *best model*.

---

## 📦 Requisiti

Installa i pacchetti necessari con:

```bash
pip install -r requirements.txt
```

Contenuto di `requirements.txt`:

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

## 🚀 Avvio

Lancia l’applicazione con:

```bash
streamlit run app.py
```

