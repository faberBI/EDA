import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.eda_utils import EDA  # ✅ percorso aggiornato
from scipy.stats import shapiro
import io
import numpy as np

# ML librerie
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.model_selection import RandomizedSearchCV

# GPT libreria
from openai import OpenAI
import time

api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="EDA + ML Automatica", layout="wide")

st.title("🔎 Exploratory Data Analysis + ML App")
target_column = None 

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

    # 🔹 Selezione variabili da considerare
    st.markdown("### 🔎 Seleziona le variabili da includere nell'analisi")
    selected_columns = st.multiselect(
        "Scegli le colonne (se non selezioni nulla, verranno usate tutte):",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )

    # Applica filtro
    df = df[selected_columns]
    # Inizializza EDA
    eda = EDA(df)

    # Info dataset
    st.subheader("ℹ️ Informazioni sul Dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("📈 Statistiche descrittive")
    st.write(eda.numeric_df.describe())

    st.subheader("❓ Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0])


    # --- Flag per imputazione ---
    missing_strategy = None  # "mean", "median", "mode", "rows", "cols", "missforest"

    if missing.sum() > 0:
        st.markdown("### 🛠️ Gestione dei Missing Values")

    option = st.radio(
        "Come vuoi gestire i valori mancanti?",
        ["Nessuna azione", "Rimuovi righe", "Rimuovi colonne", 
         "Imputazione semplice (Media/Mediana/Moda)", "Imputazione avanzata (MissForest)"]
    )

    if option == "Rimuovi righe":
        missing_strategy = "rows"
    elif option == "Rimuovi colonne":
        missing_strategy = "cols"
    elif option == "Imputazione semplice (Media/Mediana/Moda)":
        strategy = st.selectbox(
            "Scegli la strategia di imputazione",
            ["Media (solo numeriche)", "Mediana (solo numeriche)", "Moda (tutte le colonne)"]
        )
        if "Media" in strategy:
            missing_strategy = "mean"
        elif "Mediana" in strategy:
            missing_strategy = "median"
        else:
            missing_strategy = "mode"
    elif option == "Imputazione avanzata (MissForest)":
        missing_strategy = "missforest"
        st.info("ℹ️ Verrà usato MissForest dopo lo split (solo su X, non su y).")

    # Scelta target
    target_column = st.selectbox("Scegli la variabile target (y)", df.columns)

    # Distribuzione target
    st.subheader(f"📌 Distribuzione della variabile target: {target_column}")
    fig, ax = plt.subplots()
    sns.histplot(df[target_column].dropna(), kde=True, ax=ax)
    st.pyplot(fig, use_container_width=False)

    # --- Analisi univariata numerica ---
    st.subheader("📊 Distribuzioni Univariate (Numeriche)")
    for col in eda.numeric_df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(eda.numeric_df[col], kde=True, ax=ax)
        ax.set_title(f"Distribuzione di {col}")
        st.pyplot(fig, use_container_width=False)

    # --- Analisi univariata categorica ---
    if not eda.categorical_df.empty:
        st.subheader("📊 Distribuzioni Univariate (Categoriche)")
        for col in eda.categorical_df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(x=eda.categorical_df[col], ax=ax)
            ax.set_title(f"Distribuzione di {col}")
            plt.xticks(rotation=90)
            st.pyplot(fig, use_container_width=False)

    # --- Analisi bivariata numerica ---
    st.subheader("🔗 Analisi Bivariata (Numeriche vs Target)")
    for col in eda.numeric_df.columns:
        if col != target_column:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(x=eda.numeric_df[col], y=df[target_column], ax=ax)
            ax.set_title(f"{col} vs {target_column}")
            st.pyplot(fig)

    # --- Analisi bivariata categorica ---
    if not eda.categorical_df.empty:
        st.subheader("🔗 Analisi Bivariata (Categoriche vs Target)")
        for col in eda.categorical_df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x=eda.categorical_df[col], y=df[target_column], ax=ax)
            ax.set_title(f"{col} vs {target_column}")
            plt.xticks(rotation=90)
            st.pyplot(fig)

    # --- Correlazione ---
    st.subheader("📌 Matrice di Correlazione")
    fig, ax = plt.subplots(figsize=(6,4))
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

# ============================================================
# 🚀 SEZIONE MACHINE LEARNING
# ============================================================
st.header("⚡ Machine Learning Automatica")

if target_column:
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]

    # Encoding variabili categoriche
    X = pd.get_dummies(X, drop_first=True)

    # Encoding target se categorico
    if y.dtype == "object" or y.nunique() < 20:
        problem_type = "classification"
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        problem_type = "regression"

    st.write(f"🔍 Rilevato problema di **{problem_type}**")

    # --- Train-validation-test split ---
    st.markdown("### 📂 Train / Validation / Test Split")
    test_size = st.slider("Percentuale Test Set (%)", 10, 40, 20) / 100
    val_size = st.slider("Percentuale Validation Set (%)", 10, 40, 20) / 100

    # Primo split: train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42
    )
    # Secondo split: validation vs test
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val_size), random_state=42
    )

    st.write(f"📊 Train: {len(X_train)} ({len(X_train)/len(X):.1%})")
    st.write(f"📊 Validation: {len(X_val)} ({len(X_val)/len(X):.1%})")
    st.write(f"📊 Test: {len(X_test)} ({len(X_test)/len(X):.1%})")

    # ------------------------------------------------------------
    # 🔧 Gestione valori mancanti
    # ------------------------------------------------------------
    if missing_strategy:
        if missing_strategy == "drop":
            X_train = pd.DataFrame(X_train).dropna()
            X_val   = pd.DataFrame(X_val).dropna()
            X_test  = pd.DataFrame(X_test).dropna()
        elif missing_strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            X_val   = imputer.transform(X_val)
            X_test  = imputer.transform(X_test)
        elif missing_strategy == "median":
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_val   = imputer.transform(X_val)
            X_test  = imputer.transform(X_test)
        elif missing_strategy == "most_frequent":
            imputer = SimpleImputer(strategy="most_frequent")
            X_train = imputer.fit_transform(X_train)
            X_val   = imputer.transform(X_val)
            X_test  = imputer.transform(X_test)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Feature selection
    st.markdown("### ✨ Feature Selection")
    k = st.slider("Numero di features da selezionare", 
                  5, min(X.shape[1], X_train.shape[1]), 
                  min(20, X_train.shape[1]))
    if problem_type == "classification":
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)

    st.subheader("🔍 Debug X_train prima del Feature Selection")

    st.write("NaN in X_train:", np.isnan(X_train).sum().sum())
    st.write("Inf in X_train:", np.isinf(X_train).sum().sum())

    if isinstance(X_train, pd.DataFrame):
        st.write("Colonne con NaN:", X_train.columns[X_train.isna().any()].tolist())
        st.write("Colonne con Inf:", X_train.columns[np.isinf(X_train).any()].tolist())
    
    X_train = selector.fit_transform(X_train, y_train)
    X_val   = selector.transform(X_val)
    X_test  = selector.transform(X_test)

    # ------------------------------------------------------------
    # 🔘 Scelta modelli
    # ------------------------------------------------------------
    st.markdown("### ⚙️ Scegli i modelli da allenare")
    models = {}
    if problem_type == "classification":
        if st.checkbox("Logistic Regression"):
            models["Logistic Regression"] = LogisticRegression(max_iter=1000)
        if st.checkbox("Random Forest"):
            models["Random Forest"] = RandomForestClassifier()
        if st.checkbox("XGBoost"):
            models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
        if st.checkbox("LightGBM"):
            models["LightGBM"] = LGBMClassifier()
        if st.checkbox("CatBoost"):
            models["CatBoost"] = CatBoostClassifier(verbose=0)
    else:
        if st.checkbox("Linear Regression"):
            models["Linear Regression"] = LinearRegression()
        if st.checkbox("Random Forest"):
            models["Random Forest"] = RandomForestRegressor()
        if st.checkbox("XGBoost"):
            models["XGBoost"] = XGBRegressor()
        if st.checkbox("LightGBM"):
            models["LightGBM"] = LGBMRegressor()
        if st.checkbox("CatBoost"):
            models["CatBoost"] = CatBoostRegressor(verbose=0)

    # ------------------------------------------------------------
    # 🚀 Avvio Training
    # ------------------------------------------------------------
    # --- Avvio Training ---
if st.button("🚀 Avvia training"):
    if len(models) == 0:
        st.warning("⚠️ Seleziona almeno un modello per avviare il training.")
    else:
        results = {}
        best_model = None
        best_score = None

        st.subheader("🔍 Debug dataset prima del training")
        st.write(f"X_train shape: {X_train.shape} | NaN: {np.isnan(X_train).sum()}")
        st.write(f"y_train shape: {y_train.shape} | classi uniche: {np.unique(y_train)}")
        st.write(f"X_test shape: {X_test.shape} | NaN: {np.isnan(X_test).sum()}")
        st.write(f"y_test shape: {y_test.shape} | classi uniche: {np.unique(y_test)}")

        # --- Hyperparameter grids ---
        param_grids = {
            "Random Forest": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10, 20]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1]
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200, 300],
                "num_leaves": [30, 50, 100],
                "learning_rate": [0.01, 0.05, 0.1]
            },
            "CatBoost": {
                "iterations": [50, 100, 200, 300],
                "depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1]
            },
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            },
            "Linear Regression": {}  # niente da ottimizzare
        }

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_models = len(models)
        completed = 0

        for name, model in models.items():
            completed += 1
            status_text.text(f"⏳ Allenamento + tuning: {name} ({completed}/{total_models})...")

            try:
                # Se il modello ha param_grid -> tuning
                if name in param_grids and len(param_grids[name]) > 0:
                    search = RandomizedSearchCV(
                        model,
                        param_distributions=param_grids[name],
                        n_iter=10,              # numero combinazioni testate
                        cv=3,                   # cross-validation
                        scoring="f1_weighted" if problem_type == "classification" else "r2",
                        n_jobs=-1,
                        random_state=42
                    )
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    st.info(f"🔧 Best params {name}: {search.best_params_}")
                else:
                    # Nessun tuning disponibile
                    model.fit(X_train, y_train)

                # --- Predizioni
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                if problem_type == "classification":
                    metrics = {
                        "Train Accuracy": accuracy_score(y_train, y_pred_train),
                        "Test Accuracy": accuracy_score(y_test, y_pred_test),
                        "Train F1": f1_score(y_train, y_pred_train, average="weighted"),
                        "Test F1": f1_score(y_test, y_pred_test, average="weighted"),
                    }
                    score = metrics["Test F1"]

                else:
                    metrics = {
                        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                        "Train MAE": mean_absolute_error(y_train, y_pred_train),
                        "Test MAE": mean_absolute_error(y_test, y_pred_test),
                        "Train R2": r2_score(y_train, y_pred_train),
                        "Test R2": r2_score(y_test, y_pred_test),
                    }
                    score = metrics["Test R2"]

                results[name] = metrics
                st.write(f"📊 Risultati parziali - {name}", metrics)

                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    best_model = model

            except Exception as e:
                import traceback
                st.error(f"❌ Errore durante tuning/training di {name}: {type(e).__name__} - {e}")
                st.text(traceback.format_exc())

            progress_bar.progress(completed / total_models)

        status_text.text("✅ Training + tuning completato!")

        if len(results) == 0:
            st.error("❌ Nessun modello è stato allenato correttamente.")
        else:
            st.success(f"🏆 Miglior modello: {best_model.__class__.__name__}")
            st.write("### 📊 Risultati complessivi")
            results_df = pd.DataFrame(results).T
            st.write(results_df)

    # --- Grafici comparativi ---
    st.subheader("📉 Confronto modelli")
    if problem_type == "classification":
        fig, ax = plt.subplots(figsize=(8,5))
        results_df[["Train Accuracy","Test Accuracy","Train F1","Test F1"]].plot(kind="bar", ax=ax)
        plt.title("Accuracy & F1 - Train vs Test")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        if "Brier Score" in results_df.columns:
            fig, ax = plt.subplots(figsize=(8,5))
            results_df["Brier Score"].dropna().plot(kind="bar", ax=ax, color="orange")
            plt.title("Brier Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        if "ECE" in results_df.columns:
            fig, ax = plt.subplots(figsize=(8,5))
            results_df["ECE"].dropna().plot(kind="bar", ax=ax, color="purple")
            plt.title("Expected Calibration Error (ECE)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(8,5))
        results_df[["Train RMSE","Test RMSE","Train MAE","Test MAE"]].plot(kind="bar", ax=ax)
        plt.title("Errori - Train vs Test")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8,5))
        results_df[["Train R2","Test R2"]].plot(kind="bar", ax=ax, color=["green","blue"])
        plt.title("R² - Train vs Test")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Scatter Plot y_true vs y_pred (solo test) ---
    st.subheader("📌 Scatter Plot Predizioni vs Valori Reali (Test)")
    y_pred_test = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_test, y_pred_test, alpha=0.6)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set_xlabel("Valore Reale")
    ax.set_ylabel("Predizione")
    ax.set_title("Scatter Predizioni vs Valori Reali (Miglior modello)")
    st.pyplot(fig)
    
    client = OpenAI(api_key=api_key)

    # Creiamo un dataframe riassuntivo dei risultati
    summary_df = results_df.reset_index().rename(columns={"index": "Model"})

    prompt = f"""
    Ho allenato diversi modelli di Machine Learning con i seguenti risultati su Train e Test:

    {summary_df.to_string(index=False)}

    Obiettivi del commento:
    - Confronta le performance Train vs Test per evidenziare eventuale overfitting.
    - Sottolinea quale modello ha le performance migliori sul Test set.
    - Commenta robustezza, generalizzazione e eventuali rischi.
    - Suggerisci il modello più adatto come "best model".
    """

    if st.button("💬 Genera commento GPT"):
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                    {"role": "system", "content": "Sei un esperto di machine learning che commenta i risultati dei modelli."},
                    {"role": "user", "content": prompt}
                    ]
                )
                st.subheader("📑 Commento di GPT")
                st.write(response.choices[0].message.content)
                break
            except Exception as e:
                st.warning(f"Errore OpenAI ({e}), riprovo tra 5 secondi...")
                time.sleep(5)

    # --- Download modello migliore ---
    st.subheader("💾 Scarica il miglior modello")
    model_bytes = io.BytesIO()
    joblib.dump(best_model, model_bytes)
    st.download_button("Scarica modello", model_bytes, "best_model.pkl")
















