import pandas as pd
import json
import os

from clustering import interpret_participant_separated, get_cluster_station_gaps_context, get_cluster_station_gaps_only
from fine_tuning import add_instruction_tuning_columns
from src.data_utils import (
    create_user_identification,
    find_age_range, encode_gender,
    compute_residuals,
    normalize_dataframe,
    build_prompt
)
from src.models import HyroxModelRequest

# define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_REFERENCE_CSV_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_full_data_instructed_feedback.csv")
MIN_MAX_SCALER_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_minmax_scaler_stations.pkl")
MIN_MAX_SCALER_AGG_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_minmax_scaler_stations_agg.pkl")
STANDARD_SCALER_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_zscore_scaler_stations.pkl")
STANDARD_SCALER_AGG_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_zscore_scaler_stations_agg.pkl")
STANDARD_SCALER_CONTEXT_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_zscore_scaler_stations_context.pkl")
CLUSTER_PERF_ONLY = os.path.join(ROOT_DIR, "data", "processed", "hyrox_cluster_profiles_perf_only.csv")
CLUSTER_PERF_CONTEXT = os.path.join(ROOT_DIR, "data", "processed", "hyrox_cluster_profiles_perf_context.csv")
THRESHOLDS = os.path.join(ROOT_DIR, "data", "processed", "hyrox_thresholds.json")


def process_hyrox_request(request: HyroxModelRequest) -> pd.DataFrame:
    """
    End-to-end processing pipeline that transforms a HyroxModelRequest into a preprocessed DataFrame
    ready for inference, with prediction and residuals computed.

    Parameters:
    request : HyroxModelRequest
        Request containing participant input data.
    df_reference : pd.DataFrame
        Reference DataFrame with gender and age_min/age_max to infer age range.
    station_cols : list
        List of station-level feature column names for normalization.
    aggregated_cols : list
        List of aggregated feature column names for separate normalization if needed.
    context_cols : list
        List of context feature column names for normalization.

    Returns:
    pd.DataFrame
        Processed DataFrame with predicted_total_time and residual.
    """


    df_reference = pd.read_csv(DF_REFERENCE_CSV_PATH)
    id = create_user_identification(request.race_id, request.name, request.nationality) 
    gender = encode_gender(request.gender)
    age_range = find_age_range(df_reference, gender, request.age)
    if age_range is None:
        raise ValueError("No age range found for the given gender and age")
    age_min, age_max = age_range

    # build DataFrame with required structure
    data = {
        "id": [id],
        "event_name": ["-"],
        "gender": [gender],
        "age_min": [age_min],
        "age_max": [age_max],
        "total_time": [
            request.run_1 + request.run_2 + request.run_3 + request.run_4 + request.run_5 + request.run_6 + request.run_7 + request.run_8 +
            request.work_1 + request.work_2 + request.work_3 + request.work_4 + request.work_5 + request.work_6 + request.work_7 + request.work_8 +
            request.roxzone_1 + request.roxzone_2 + request.roxzone_3 + request.roxzone_4 + request.roxzone_5 + request.roxzone_6 + request.roxzone_7
        ],
        "work_time": [
            request.work_1 + request.work_2 + request.work_3 + request.work_4 + request.work_5 + request.work_6 + request.work_7 + request.work_8
        ],
        "roxzone_time": [
            request.roxzone_1 + request.roxzone_2 + request.roxzone_3 + request.roxzone_4 + request.roxzone_5 + request.roxzone_6 + request.roxzone_7
        ],
        "run_time": [
            request.run_1 + request.run_2 + request.run_3 + request.run_4 + request.run_5 + request.run_6 + request.run_7 + request.run_8
        ],
        "run_1": [request.run_1],
        "1000m Ski": [request.work_1],
        "roxzone_1": [request.roxzone_1],
        "run_2": [request.run_2],
        "50m Sled Push": [request.work_2],
        "roxzone_2": [request.roxzone_2],
        "run_3": [request.run_3],
        "50m Sled Pull": [request.work_3],
        "roxzone_3": [request.roxzone_3],
        "run_4": [request.run_4],
        "80m Burpee Broad Jump": [request.work_4],
        "roxzone_4": [request.roxzone_4],
        "run_5": [request.run_5],
        "1000m Row": [request.work_5],
        "roxzone_5": [request.roxzone_5],
        "run_6": [request.run_6],
        "200m Farmer Carry": [request.work_6],
        "roxzone_6": [request.roxzone_6],
        "run_7": [request.run_7],
        "100m Sandbag Lunges": [request.work_7],
        "roxzone_7": [request.roxzone_7],
        "run_8": [request.run_8],
        "100 Wall Balls": [request.work_8]
    }

    df = pd.DataFrame(data)

    # apply z-score normalization 
    df = normalize_dataframe(df, create_station_cols(), suffix="_zscore", PATH=STANDARD_SCALER_PATH)
    df = normalize_dataframe(df, create_station_agg_cols(), suffix="_zscore_agg", PATH=STANDARD_SCALER_AGG_PATH)
    df = normalize_dataframe(df, create_station_context_cols(), suffix="_zscore_context", PATH=STANDARD_SCALER_CONTEXT_PATH)

    # apply MinMax normalization
    df = normalize_dataframe(df, suffix="_minmax", PATH=MIN_MAX_SCALER_PATH)
    df = normalize_dataframe(df, suffix="_minmax_agg", PATH=MIN_MAX_SCALER_AGG_PATH)

    # compute predicted total time and residuals
    df = compute_residuals(df)

    # get cluster profiles
    df_clusters_perf_only = pd.read_csv(CLUSTER_PERF_ONLY)
    df_clusters_perf_context = pd.read_csv(CLUSTER_PERF_CONTEXT)

    with open(THRESHOLDS, "r") as f:
        loaded_thresholds = json.load(f)

    low_thresh = loaded_thresholds["low_thresh"]
    high_thresh = loaded_thresholds["high_thresh"]

    # get z-score columns for stations
    stations_zscore_cols = df_clusters_perf_context.columns[1:24].tolist()

    # create columns for suggestions and performance feedback
    df["suggestions_perf_only"] = df.apply(
        lambda row: get_cluster_station_gaps_only(row, df_clusters_perf_only, stations_zscore_cols),
        axis=1
    )
    df["suggestions_perf_context"] = df.apply(
        lambda row: get_cluster_station_gaps_context(row, df_clusters_perf_context, stations_zscore_cols),
        axis=1
    )
    df["performance_feedback"] = df.apply(
        lambda row: interpret_participant_separated(row, df_clusters_perf_only, df_clusters_perf_context, stations_zscore_cols, low_thresh, high_thresh),
        axis=1
    )
    df["text"] = df.apply(build_prompt, axis=1)
    df = add_instruction_tuning_columns(df)
    return df


def create_station_cols():
    return ['run_1', '1000m Ski', 'roxzone_1', 'run_2', '50m Sled Push', 'roxzone_2', 'run_3',
            '50m Sled Pull', 'roxzone_3', 'run_4', '80m Burpee Broad Jump', 'roxzone_4',
            'run_5', '1000m Row', 'roxzone_5', 'run_6', '200m Farmer Carry', 'roxzone_6',
            'run_7', '100m Sandbag Lunges', 'roxzone_7', 'run_8', '100 Wall Balls']

def create_station_agg_cols():
    return ['total_time', 'work_time', 'roxzone_time', 'run_time']

def create_station_context_cols():
    return ['gender', 'age_min', 'age_max', 'run_1', '1000m Ski', 'roxzone_1', 'run_2',
            '50m Sled Push', 'roxzone_2', 'run_3', '50m Sled Pull', 'roxzone_3', 'run_4',
            '80m Burpee Broad Jump', 'roxzone_4', 'run_5', '1000m Row', 'roxzone_5', 'run_6',
            '200m Farmer Carry', 'roxzone_6', 'run_7', '100m Sandbag Lunges', 'roxzone_7',
            'run_8', '100 Wall Balls']

def extract_race_minutes_from_df(df: pd.DataFrame) -> dict:
    """
    Extrae los valores de tiempo por estación en minutos desde un DataFrame procesado por el pipeline.

    Parameters:
    df : pd.DataFrame
        DataFrame con una sola fila del usuario procesado.

    Returns:
    dict : Diccionario con los tiempos por estación en minutos.
    """
    row = df.iloc[0]
    stations = [
        'run_1', '1000m Ski', 'roxzone_1', 'run_2', '50m Sled Push', 'roxzone_2',
        'run_3', '50m Sled Pull', 'roxzone_3', 'run_4', '80m Burpee Broad Jump', 'roxzone_4',
        'run_5', '1000m Row', 'roxzone_5', 'run_6', '200m Farmer Carry', 'roxzone_6',
        'run_7', '100m Sandbag Lunges', 'roxzone_7', 'run_8', '100 Wall Balls',
        'run_time', 'work_time', 'roxzone_time', 'total_time'
    ]

    return {col: round(row[col], 2) for col in stations if col in df.columns}