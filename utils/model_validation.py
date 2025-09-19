def plot_feature_importance(model, X_train, y_train, feature_names):


    # Recupera importance se disponibile
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = np.mean(importances, axis=0)
    else:
        st.warning("⚠️ Questo modello non fornisce direttamente le feature importance.")
        return None, None

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fi_df.head(15).plot(kind="barh", x="Feature", y="Importance", ax=ax, legend=False)
    ax.set_title("Top Feature Importance")
    plt.tight_layout()

    return fi_df, fig

def plot_learning_curve(estimator, X, y, scoring, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Genera e ritorna un grafico delle learning curves senza salvare su file.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        shuffle=True,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", label="Train score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)

    ax.plot(train_sizes, val_mean, "o-", label="Validation score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

    ax.set_title("Learning Curve")
    ax.set_xlabel("Numero di esempi di training")
    ax.set_ylabel(scoring)
    ax.legend(loc="best")
    ax.grid(True)

    return fig, (train_sizes, train_mean, val_mean)

def custom_lime_explanation(model, X_train, instance, num_features=10, num_samples=5000):
    """
    Implementazione semplificata di LIME custom senza pacchetto esterno.
    Ritorna un grafico matplotlib pronto per Streamlit.
    """
    np.random.seed(42)

    # Campiona intorno all'istanza con rumore
    X_sample = np.repeat([instance], num_samples, axis=0)
    noise = np.random.normal(0, 0.01, X_sample.shape)
    X_sample = X_sample + noise

    # Predizioni del modello
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_sample)[:, 1]  # se classificazione binaria
    else:
        y_pred = model.predict(X_sample)  # regression

    # Calcolo similarità con kernel RBF
    distances = np.linalg.norm(X_sample - instance, axis=1)
    kernel_width = np.sqrt(X_train.shape[1]) * 0.75
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    # Fit modello interpretabile (regressione lineare pesata)
    from sklearn.linear_model import Ridge
    interpretable_model = Ridge(alpha=1.0)
    interpretable_model.fit(X_sample, y_pred, sample_weight=weights)

    coefs = interpretable_model.coef_
    feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"f{i}" for i in range(X_train.shape[1])]
    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Weight": coefs
    }).sort_values(by="Weight", key=abs, ascending=False).head(num_features)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    explanation.plot(kind="barh", x="Feature", y="Weight", ax=ax, legend=False, color="skyblue")
    ax.set_title("LIME Explanation (custom)")
    plt.tight_layout()

    return explanation, fig
