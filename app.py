import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.eda_class import EDA  # ✅ percorso aggiornato
from scipy.stats import shapiro  # ✅ serve per il test di normalità


st.set_page_config(page_title="EDA Automatica", layout="wide")

st.title("🔎 Exploratory Data Analysis App")

# Upload file
uploaded_file = st.file_uploader("Carica un dataset (.csv o .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Legge il file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Anteprima del Dataset")
    st.write(df.head())

    # Inizializza EDA
    eda = EDA(df)

    # Info dataset
    st.subheader("ℹ️ Informazioni sul Dataset")
    buffer = []
    df.info(buf=buffer.append)   # workaround per catturare il testo
    st.text("\n".join(buffer))

    st.subheader("📈 Statistiche descrittive")
    st.write(eda.numeric_df.describe())

    st.subheader("❓ Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0])

    # Scelta target
    target_column = st.selectbox("Scegli la variabile target (y)", df.columns)

    # Distribuzione target
    st.subheader(f"📌 Distribuzione della variabile target: {target_column}")
    fig, ax = plt.subplots()
    sns.histplot(df[target_column].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    # --- Analisi univariata numerica ---
    st.subheader("📊 Distribuzioni Univariate (Numeriche)")
    for col in eda.numeric_df.columns:
        fig, ax = plt.subplots()
        sns.histplot(eda.numeric_df[col], kde=True, ax=ax)
        ax.set_title(f"Distribuzione di {col}")
        st.pyplot(fig)

    # --- Analisi univariata categorica ---
    if not eda.categorical_df.empty:
        st.subheader("📊 Distribuzioni Univariate (Categoriche)")
        for col in eda.categorical_df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x=eda.categorical_df[col], ax=ax)
            ax.set_title(f"Distribuzione di {col}")
            plt.xticks(rotation=90)
            st.pyplot(fig)

    # --- Analisi bivariata numerica ---
    st.subheader("🔗 Analisi Bivariata (Numeriche vs Target)")
    for col in eda.numeric_df.columns:
        if col != target_column:
            fig, ax = plt.subplots()
            sns.scatterplot(x=eda.numeric_df[col], y=df[target_column], ax=ax)
            ax.set_title(f"{col} vs {target_column}")
            st.pyplot(fig)

    # --- Analisi bivariata categorica ---
    if not eda.categorical_df.empty:
        st.subheader("🔗 Analisi Bivariata (Categoriche vs Target)")
        for col in eda.categorical_df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=eda.categorical_df[col], y=df[target_column], ax=ax)
            ax.set_title(f"{col} vs {target_column}")
            plt.xticks(rotation=90)
            st.pyplot(fig)

    # --- Correlazione ---
    st.subheader("📌 Matrice di Correlazione")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(eda.numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --- PCA ---
    st.subheader("📊 PCA Analysis")
    fig = eda.pca_analysis(return_fig=True)
    st.pyplot(fig)

    # --- Clustering ---
    st.subheader("🤖 Clustering Analysis")
    figs = eda.clustering_analysis(return_fig=True)
    for f in figs:
        st.pyplot(f)

    # --- Normalità ---
    st.subheader("📏 Test di Normalità (Shapiro-Wilk)")
    for col in eda.numeric_df.columns:
        stat, p = shapiro(eda.numeric_df[col].dropna())
        st.write(f"**{col}** → Stat={stat:.4f}, p-value={p:.4f}")

    # --- Download dataset finale ---
    st.subheader("💾 Scarica Dataset Elaborato")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica CSV", csv, "dataset_elaborato.csv", "text/csv")

