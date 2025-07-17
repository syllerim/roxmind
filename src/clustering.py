import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np

from collections import defaultdict
from mlflow.models.signature import infer_signature

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
def plot_pca_clusters(X_scaled, cluster_labels, title="Clusters (PCA)"):
    """
    Plots the clusters in a 2D space using PCA for dimensionality reduction.

    Args:
        X_scaled : array-like or DataFrame, shape (n_samples, n_features)
            The input data after scaling (e.g. MinMaxScaler or StandardScaler).
        cluster_labels : array-like, shape (n_samples,) 
            Cluster assignments for each sample (e.g. output of KMeans).
        title : str, optional (default="Clusters (PCA)")
            Title of the plot.
    """
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------
def log_model_with_mlflow(model, model_name: str, X_train, y_train, X_test, y_test):
    """
    Train and log a regression model to MLflow with input example, signature, metrics, and model type.
    
    Args:
        model (regressor): sklearn or LightGBM model instance (e.g., LinearRegression() or LGBMRegressor())
        model_name (str): A name for the model to tag the run (e.g., "LinearRegression" or "LightGBM")
        X_train, y_train: Training data
        X_test, y_test: Test data
    """
    with mlflow.start_run(run_name=f"{model_name}_Hyrox"):
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Log model
        input_example = X_test.iloc[:1] if hasattr(X_test, "iloc") else X_test[:1]
        signature = infer_signature(X_test, y_pred)
        
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=input_example)
        mlflow.log_metric("rmse_minutes", rmse)
        mlflow.log_param("model_type", model_name)

        print(f"✅ Logged {model_name} with RMSE: {rmse:.2f} minutes and model signature.")

    mlflow.end_run()


# -------------------------------------------------------------------------------------------------------------------
def get_station_gaps_vs_cluster_zscore(row, cluster_profiles, cluster_col: str, stations_zscore_cols, threshold=0.05):
    """
    Identifies underperforming stations for a participant based on their Z-score
    compared to their cluster's average.

    Args:
        row: A row from the main DataFrame (a participant).
        cluster_profiles: DataFrame with mean Z-scores per cluster.
        cluster_col: The name of the cluster column.
        stations_zscore_cols: List of columns with Z-scores for each station.
        threshold: How far above the cluster average (in Z-score) an participant can go before being considered underperforming.

    Returns:
        List of underperforming station names.
    Example:
        Let's say participant belongs to cluster 2, the average z-score for 'run_3_zscore' in cluster 2 is -0.2,
        then if the participant value is below -0.5 (i.e., slower), and threshold is -0.3,
        the station 'run_3' will be included in the result.

        Example output: ['run_3', 'work_2', 'roxzone_6']
    """
    cluster_id = row[cluster_col]
    if cluster_id not in cluster_profiles.index:
        return []
    
    
    cluster_avg = cluster_profiles.loc[cluster_id]
    underperforming = []

    for col in stations_zscore_cols:
        if col in row and col in cluster_avg:
            # higher z-score than cluster average (by threshold) = slower than expected
            if row[col] > cluster_avg[col] + threshold:
                underperforming.append(col.replace('_zscore', ''))

    return underperforming


# -----------------------------------------------------------------------------------------------------
def get_cluster_station_gaps_only(row, cluster_profiles_perf, stations_zscore_cols, threshold=0.3):
    """
    Identifies the stations where the participant underperformed compared to the average of the performance-only cluster.

    Returns:
        List of station names (without '_zscore') where underperformance was detected.
    """
    return get_station_gaps_vs_cluster_zscore(
        row,
        cluster_profiles=cluster_profiles_perf,
        cluster_col='cluster_perf_only',
        stations_zscore_cols=stations_zscore_cols,
        threshold=threshold
    )

# ------------------------------------------------------------------------------------------------------
def get_cluster_station_gaps_context(row, cluster_profiles_context, stations_zscore_cols, threshold=0.3):
    """
    Identifies the stations where the participant underperformed compared to the average of the context-based cluster.

    Returns:
        List of station names (without '_zscore') where underperformance was detected.
    """
    return get_station_gaps_vs_cluster_zscore(
        row,
        cluster_profiles=cluster_profiles_context,
        cluster_col='cluster_perf_context',
        stations_zscore_cols=stations_zscore_cols,
        threshold=threshold
    )

# ------------------------------------------------------------------------------------------------------
def interpret_participant_separated(row, cluster_profiles_perf, cluster_profiles_context, stations_zscore_cols, low_threshold, high_threshold):
    """
    Same as interpret_participant_dual but splits feedback cleanly into separate perf_only and perf_context sections.
    Useful for more flexible and structured downstream usage.
    """
    res = row['residual']
    perf_only = get_cluster_station_gaps_only(row, cluster_profiles_perf, stations_zscore_cols)
    perf_context = get_cluster_station_gaps_context(row, cluster_profiles_context, stations_zscore_cols)

    # residual feedback
    if res > high_threshold:
        base = "You performed slower than expected! There is room for improvement."
    elif res < low_threshold:
        base = "You performed better than expected based on your overall predicted time."
    else:
        base = "You performed close to expected levels."

    # structure output
    perf_text = f"Compared to all athletes, you could improve: {', '.join(perf_only)}." if perf_only else ""
    context_text = f"Compared to athletes of similar age and gender, your weak areas were: {', '.join(perf_context)}." if perf_context else ""

    return f"{base} {perf_text} {context_text}".strip()

# ------------------------------------------------------------------------------------------------------