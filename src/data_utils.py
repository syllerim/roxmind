import joblib
import mlflow
import os
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_all_csvs_from_folder(folder_path: str) -> pd.DataFrame:
    folder = Path(folder_path)
    all_csvs = list(folder.glob("*.csv"))

    df_list = []
    for csv_file in all_csvs:
        print(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def time_to_minutes(time_str):
    """
    Convert a time string in format 'H:MM:SS' to total minutes.

    Args:
        time_str (str): Time string in format like '0:56:47'

    Returns:
        int: Total seconds

    Examples:
        >>> time_to_seconds('0:56:47')
        3407
        >>> time_to_seconds('1:30:15')
        5415
        >>> time_to_seconds('2:00:00')
        7200
    """
    try:
        total_seconds = time_to_seconds(time_str)
        return total_seconds / 60
    except Exception as e:
        print(f"Warning: Failed to convert '{time_str}' to minutes. Error: {e}")
        return None


def time_to_seconds(time_str):
    """
    Convert a time string in format 'H:MM:SS' to total seconds.

    Args:
        time_str (str): Time string in format like '0:56:47'

    Returns:
        int: Total seconds

    Examples:
        >>> time_to_seconds('0:56:47')
        3407
        >>> time_to_seconds('1:30:15')
        5415
        >>> time_to_seconds('2:00:00')
        7200
    """
    try:
        # split the time string by ':'
        time_parts = time_str.split(':')

        if len(time_parts) != 3:
            raise ValueError("Time string must be in format 'H:MM:SS'")

        # convert each part to integer
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])

        # validate ranges
        if minutes >= 60 or seconds >= 60:
            raise ValueError("Minutes and seconds must be less than 60")

        if hours < 0 or minutes < 0 or seconds < 0:
            raise ValueError("Time values cannot be negative")

        # convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds

        return total_seconds

    except ValueError as e:
        if "invalid literal for int()" in str(e):
            raise ValueError("All time components must be valid integers") from e
        raise

def create_user_identification(race_id: str, name: str, nationality: str) -> str:
    """
    Constructs a user identification string in the format:
    <race_id>_<name>_<nationality>

    Example:
        create_user_identification("111008", "Dienst, Josef", "GER")
        => "111008_Dienst, Josef_GER"
    """
    return f"{race_id}_{name}_{nationality}"


def find_age_range(df, gender: int, age: int):
    """
    Find the age range (min_age, max_age) that matches the given gender and age.

    Parameters:
    df : pd.DataFrame
        DataFrame containing at least the columns 'gender', 'age_min', and 'age_max'.
    gender : int
        Gender encoded as integer (e.g., 0 = female, 1 = male).
    age : int
        Age of the participant.

    Returns:
    tuple
        A tuple (min_age, max_age) that contains the input age.

    Raises:
    ValueError
        If gender is None or no matching age range is found.
    """
    if gender is None:
        raise ValueError("Gender must be 'male' or 'female'")

    gender_df = df[df['gender'] == gender]
    age_min_list = sorted(gender_df['age_min'].unique())
    age_max_list = sorted(gender_df['age_max'].unique())

    # find the matching range
    for min_age, max_age in zip(age_min_list, age_max_list):
        if min_age <= age <= max_age:
            return (min_age, max_age)

    return None


def encode_gender(gender_str: str) -> int:
    """
    Converts gender string to integer encoding.

    Parameters:
    gender_str : str
        Gender as string, either 'male' or 'female' (case insensitive).

    Returns:
    int
        0 for male, 1 for female.

    Raises:
    ValueError
        If gender_str is not 'male' or 'female'.
    """
    gender_map = {"male": 0, "female": 1}
    gender_value = gender_map.get(gender_str.lower())

    if gender_value is None:
        raise ValueError("Gender must be 'male' or 'female'")

    return gender_value


def normalize_dataframe(df: pd.DataFrame, scaler, stations, suffix, PATH) -> pd.DataFrame:
    """
    Normalizes the input DataFrame using the provided scaler.
    
    Args:
        df (pd.DataFrame): The input DataFrame to normalize.
        scaler: A fitted sklearn scaler (e.g., MinMaxScaler, StandardScaler).
        stations (list): List of station columns to normalize.
        suffix (str): Suffix to append to the normalized columns.
        PATH (str): Path to the scaler file if loading from disk.
    Returns:
        pd.DataFrame: A normalized DataFrame with the same column names and index.
    """
    scaler = joblib.load(PATH)
    scaled_array = scaler.transform(df)
    normalized_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    for col in stations:
        df[f'{col}{suffix}'] = normalized_df[col]
    return df

def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes predicted_total_time and residual for each participant in the dataset.

    Parameters:
    df : pd.DataFrame
        DataFrame containing at least the following min-max scaled features:
        [
            'gender_minmax', 'age_min_minmax', 'age_max_minmax',
            'run_1_minmax', '1000m Ski_minmax', 'roxzone_1_minmax',
            'run_2_minmax', '50m Sled Push_minmax', 'roxzone_2_minmax',
            'run_3_minmax', '50m Sled Pull_minmax', 'roxzone_3_minmax',
            'run_4_minmax', '80m Burpee Broad Jump_minmax', 'roxzone_4_minmax',
            'run_5_minmax', '1000m Row_minmax', 'roxzone_5_minmax',
            'run_6_minmax', '200m Farmer Carry_minmax', 'roxzone_6_minmax',
            'run_7_minmax', '100m Sandbag Lunges_minmax', 'roxzone_7_minmax',
            'run_8_minmax', '100 Wall Balls_minmax'
        ]
        Must also contain the actual column 'total_time'.

    Returns:
    pd.DataFrame
        Original DataFrame with two new columns added:
        - 'predicted_total_time'
        - 'residual' = total_time - predicted_total_time
    """

    logged_model_uri = "runs:/8de3fb2c611346708423de9175847b88/model"
    model = mlflow.sklearn.load_model(logged_model_uri)

    feature_cols = df.columns.tolist()
    df.insert(7, 'predicted_total_time', model.predict(df[feature_cols]))
    df.insert(6, 'residual', df['total_time'] - df['predicted_total_time'])

    return df

def build_prompt(row):
    return f"""### Instruction:
    {row['input']}

    ### Context:
    {row['context']}

    ### Response:
    {row['response']}"""