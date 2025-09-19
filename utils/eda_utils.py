import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import shapiro
import ppscore as pps
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
import os
from sklearn.inspection import permutation_importance

class EDA:
    def __init__(self, df):
        self.df = df
        self.numeric_df = self.df.select_dtypes(include=np.number)
        self.categorical_df = self.df.select_dtypes(include='object')

    def save_plot(self, plot_func, filename):
        """Utility to save plots to a specified file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plot_func()
        plt.savefig(filename)
        plt.close()

    def save_data(self, filename):
        """Save the DataFrame to a file (CSV, Parquet, or Excel)."""
        ext = os.path.splitext(filename)[-1].lower()
        if ext == '.csv':
            self.df.to_csv(filename, index=False)
        elif ext == '.parquet':
            self.df.to_parquet(filename, index=False)
        elif ext == '.xlsx':
            self.df.to_excel(filename, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv, .parquet, or .xlsx.")
        print(f"DataFrame saved to {filename}")

    def feature_summary(self):
        """Displays a summary of the dataset's features.\n"""
        print(self.df.info())

    def descriptive_statistics(self):
        """Calculates and displays descriptive statistics for the dataset.\n"""
        print(self.numeric_df.describe())

    def missing_values(self):
        """Shows the number of missing values for each column."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        print("Missing values by column:\n", missing)

    def handle_missing_values(self, strategy='mean', fill_value=None, n_neighbors=5):
        """
        Handles missing values using one of the following strategies:
        - 'mean': fills with the mean
        - 'median': fills with the median
        - 'mode': fills with the mode
        - 'knn': uses KNN for imputation
        """
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            self.numeric_df.loc[:, :] = imputer.fit_transform(self.numeric_df)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'knn':
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            self.numeric_df.loc[:, :] = knn_imputer.fit_transform(self.numeric_df)
        else:
            raise ValueError("Unsupported strategy. Choose from 'mean', 'median', 'mode', 'knn'.")
        print(f"Missing values handled using strategy: {strategy}")
        # Verify that there are no remaining missing values
        if self.df.isnull().sum().sum() == 0:
            print("All missing values have been handled.\n")
        else:
            print("Warning: There are still missing values in the dataset.\n")

    def remove_duplicates(self):
        """Removes duplicate rows from the dataset."""
        initial_shape = self.df.shape
        self.df = self.df.drop_duplicates()
        print(f"Duplicate rows removed. Initial dimensions: {initial_shape}, final dimensions: {self.df.shape}\n")

    def outlier_analysis(self, k = 1.5):
        """Analyzes outliers using the IQR technique."""
        Q1 = self.numeric_df.quantile(0.25)
        Q3 = self.numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        outliers = ((self.numeric_df < (Q1 - k * IQR)) | (self.numeric_df > (Q3 + k * IQR)))

        for column in outliers.columns:
            outlier_count = outliers[column].sum()
            print(f"Column '{column}': {outlier_count} outliers detected.")

        print("\n")

    def remove_outliers(self):
        """Removes outliers based on the IQR technique."""
        Q1 = self.numeric_df.quantile(0.25)
        Q3 = self.numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        # Define a mask for data without outliers
        mask = ~((self.numeric_df < (Q1 - 1.5 * IQR)) | (self.numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
        self.df = self.df[mask]
        self.numeric_df = self.df.select_dtypes(include=np.number)
        print(f"Outliers removed. Final dimensions: {self.df.shape}\n")

    def univariate_distribution_numeric(self):
        """Displays the distribution of numerical variables."""
        for col in self.numeric_df.columns:
            plt.figure()
            plt.xticks(rotation=90)
            sns.histplot(self.numeric_df[col], kde=True)
            plt.title(f'Distribution of variable {col}')
            self.save_plot(plt.gca().get_figure().show, f'eda_images/univariate/{col}_distribution.png')

    def univariate_distribution_categorical(self):
        """Displays the distribution of categorical variables."""
        for col in self.categorical_df.columns:
            plt.figure()
            plt.xticks(rotation=90)
            sns.countplot(x=self.categorical_df[col])
            plt.title(f'Distribution of variable {col}')
            self.save_plot(plt.gca().get_figure().show, f'eda_images/univariate/{col}_distribution.png')

    def bivariate_analysis_numeric(self, target_column):
        """Bivariate analysis between numerical variables and the target column."""
        if target_column not in self.df.columns:
            raise KeyError(f"The target column '{target_column}' does not exist in the DataFrame.")

        for col in self.numeric_df.columns:
            if col != target_column:
                plt.figure()
                sns.scatterplot(x=self.numeric_df[col], y=self.df[target_column])
                plt.title(f'Bivariate Analysis: {col} vs {target_column}')
                self.save_plot(plt.gca().get_figure().show, f'eda_images/bivariate/{col}_vs_{target_column}.png')

    def bivariate_analysis_categorical(self, target_column):
        """Bivariate analysis between categorical variables and target column."""
        for col in self.categorical_df.columns:
            plt.figure()
            plt.xticks(rotation=90)
            sns.boxplot(x=self.categorical_df[col], y=self.df[target_column])
            plt.title(f'Bivariate Analysis: {col} vs {target_column}')
            self.save_plot(plt.gca().get_figure().show, f'eda_images/bivariate/{col}_vs_{target_column}.png')

    def correlation_matrix(self):
        """Displays the correlation matrix for numerical variables."""
        corr_matrix = self.numeric_df.corr()
        plt.figure(figsize=(10, 6))
        plt.xticks(rotation=90)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        self.save_plot(plt.gca().get_figure().show, 'eda_images/correlation_matrix.png')

    def pca_analysis(self, n_components=2, return_fig=False):
        df_scaled = StandardScaler().fit_transform(self.numeric_df.dropna())
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(df_scaled)
        fig, ax = plt.subplots()
        ax.scatter(components[:, 0], components[:, 1])
        ax.set_title("PCA Analysis")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        if return_fig:
            return fig
        else:
            self.save_plot(lambda: plt.scatter(components[:, 0], components[:, 1]), 'eda_images/pca_analysis.png')



    def predict_score_index(self, target_column):
        """Calculates and visualizes the Predictive Power Score (PPS) for variables."""
        def plot():
            pps_matrix = pps.matrix(self.df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
            sns.heatmap(pps_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Predictive Power Score Matrix')
            plt.xticks(rotation=90)
        self.save_plot(plot, 'eda_images/multivariate_analysis/predictive_power_score.png')

    def multivariate_visualization(self):
        """Displays multivariate plots for exploratory data analysis."""
        sns.pairplot(self.df)
        plt.title('Multivariate Visualization')
        plt.xticks(rotation=90)
        self.save_plot(plt.gca().get_figure().show, 'eda_images/multivariate_visualization.png')

    def normality_assessment(self):
        """Assesses the normality of numerical variables using the Shapiro-Wilk test."""
        for col in self.numeric_df.columns:
            stat, p_value = shapiro(self.numeric_df[col].dropna())
            print(f'Shapiro-Wilk Test for {col}: Statistic={stat}, p-value={p_value}')

    def target_distribution_analysis(self, target_column, log_scale=False):
        """Analyzes the distribution of the target variable with optional logarithmic scale."""
        plt.figure(figsize=(10, 6))
        plt.xticks(rotation=90)
        sns.histplot(self.df[target_column].dropna(), kde=True)
        if log_scale:
            plt.xscale('log')  # Use logarithmic scale
            plt.title(f'Distribution of target variable: {target_column} (Log Scale)')
        else:
            plt.title(f'Distribution of target variable: {target_column}')
        self.save_plot(plt.gca().get_figure().show, f'eda_images/target_distribution_{target_column}_{"log" if log_scale else "linear"}.png')

    def clustering_analysis(self, n_clusters=3, return_fig=False):
        """
        Performs clustering analysis using KMeans and DBSCAN, 
        calculates silhouette score and plots results.

        Parameters
        ----------
        n_clusters : int
            Number of clusters for KMeans.
        return_fig : bool
            If True, returns matplotlib figures instead of saving them.
        """

    # Usa solo righe senza NaN per il clustering
        df_no_na = self.numeric_df.dropna()
        df_scaled = StandardScaler().fit_transform(df_no_na)

        figs = []  # per Streamlit

    # --- KMeans ---
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_clusters = kmeans.fit_predict(df_scaled)

    # 🔧 Allinea ai soli indici senza NaN
        self.df.loc[df_no_na.index, "KMeans_Cluster"] = kmeans_clusters

        silhouette_avg_kmeans = silhouette_score(df_scaled, kmeans_clusters)
        print(f"Silhouette Score for KMeans: {silhouette_avg_kmeans}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=df_scaled[:, 0], 
            y=df_scaled[:, 1], 
            hue=kmeans_clusters, 
            palette="viridis", 
            ax=ax
        )
        ax.set_title("KMeans Clustering")
        figs.append(fig)

        if not return_fig:
            self.save_plot(
                lambda: sns.scatterplot(
                    x=df_scaled[:, 0], 
                    y=df_scaled[:, 1], 
                    hue=kmeans_clusters, 
                    palette="viridis"
                ),
                "eda_images/kmeans_clustering.png"
            )

    # --- DBSCAN ---
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_clusters = dbscan.fit_predict(df_scaled)

        self.df.loc[df_no_na.index, "DBSCAN_Cluster"] = dbscan_clusters

        if len(set(dbscan_clusters)) > 1:
            silhouette_avg_dbscan = silhouette_score(df_scaled, dbscan_clusters)
            print(f"Silhouette Score for DBSCAN: {silhouette_avg_dbscan}")
        else:
            print("Silhouette score not calculable for DBSCAN (number of clusters < 2).")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=df_scaled[:, 0], 
            y=df_scaled[:, 1], 
            hue=dbscan_clusters, 
            palette="viridis", 
            ax=ax
        )
        ax.set_title("DBSCAN Clustering")
        figs.append(fig)

        if not return_fig:
            self.save_plot(
                lambda: sns.scatterplot(
                    x=df_scaled[:, 0], 
                    y=df_scaled[:, 1], 
                    hue=dbscan_clusters, 
                    palette="viridis"
                ),
                "eda_images/dbscan_clustering.png"
            )

        if return_fig:
            return figs



    def save_final_db(self, filename):
        """Saves the final DataFrame with clustering columns to a file (CSV, Parquet, or Excel)."""
        self.save_data(filename)


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
