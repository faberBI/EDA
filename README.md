# EDA
Exploratory analysis for tabular dataframe

---

# 🔎 Exploratory Data Analysis App

Questa applicazione consente di caricare un dataset (CSV o Excel) ed eseguire un'**Analisi Esplorativa dei Dati (EDA)** in maniera automatica e interattiva tramite **Streamlit**.

---

## ✨ Funzionalità

- Caricamento dataset (`.csv`, `.xlsx`)
- Scelta della variabile **target (y)** tra le colonne disponibili
- Analisi **univariata** (distribuzioni numeriche e categoriche)
- Analisi **bivariata** (numeriche vs target, categoriche vs target)
- **Matrice di correlazione**
- **PCA Analysis** (riduzione dimensionale)
- **Clustering** con KMeans e DBSCAN (con Silhouette Score)
- **Test di normalità (Shapiro-Wilk)**
- Download del dataset arricchito con i cluster

---

## 📦 Requisiti

Installa i pacchetti necessari con:

```bash
pip install -r requirements.txt
