import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import joblib
import os
import pandas as pd
import torch

from src.rag.vector_store import build_chroma_vector_store_from_df
from src.models import HyroxModelRequest
from src.data_utils import find_age_range
from src import pipeline
from pipeline import extract_race_minutes_from_df
from chromadb import PersistentClient
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


# global definitions
MODEL_PATH = "Syllerim/hyrox_mistral_lora_model"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_full_data_instructed_feedback.csv")
MIN_MAX_SCALER_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_minmax_scaler_stations.pkl")
AGE_SCALER_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_minmax_scaler_ages.pkl")

CHROMA_FOLDER_PATH = '../data/store'
os.makedirs(CHROMA_FOLDER_PATH, exist_ok=True)

df = None
stations_scaler = None
age_scaler = None

model = None
tokenizer = None
generator = None

# ----------------------------------------------------------------------------
def load_model():
    global df, stations_scaler, age_scaler, model, tokenizer, generator
    if model is None or tokenizer is None or generator is None or df is None or stations_scaler is None or age_scaler is None:
        df = pd.read_csv(CSV_PATH)
        stations_scaler = joblib.load(MIN_MAX_SCALER_PATH)
        age_scaler = joblib.load(AGE_SCALER_PATH)

        peft_model_id = MODEL_PATH
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        merged = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        generator = pipeline(
            "text-generation",
            model=merged,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            max_new_tokens=300,
            do_sample=True
        )
    return generator

# ----------------------------------------------------------------------------
def generate_feedback(data: HyroxModelRequest):
    
    load_model()

    # process input data through the full pipeline
    processed_df = pipeline.process_hyrox_request(data)

    # prepare query to the vector store
    query_text = "I want to find participants who had similar overall performance to me."

    race_minutes = pipeline.extract_race_minutes_from_df(processed_df)

    # perform vector search
    collection_name = "hyrox_participants"
    persist_dir = CHROMA_FOLDER_PATH
    client = client = PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    results = collection.query(query_texts=[query_text], n_results=1, metadata={"race_minutes": race_minutes})
    if not results["documents"] or not results["documents"][0]:
        return {"error": "No similar participants found."}

    # convert vector search result back into a DataFrame
    similar_participant_data = pd.DataFrame([results["documents"][0]["metadata"]])

    # analyze improvement suggestions
    perf_only_suggestions = analyze_improvement_from_similar_perf_only(processed_df, similar_participant_data)
    perf_context_suggestions = analyze_improvement_from_similar_perf_context(processed_df, similar_participant_data)

    return {
        "performance_feedback": processed_df["performance_feedback"].iloc[0],
        "perf_only_improvements": perf_only_suggestions, # based on compare with the similar participant's performance
        "perf_context_improvements": perf_context_suggestions # based on compare with the similar participant's performance
    }


def build_prompt(data):
    # convert gender to expected format ("male" -> 0, "female" -> 1)
    gender_map = {"male": 0, "female": 1}
    gender_value = gender_map.get(data.gender.lower())

   

    # prompt = f"""You are a HYROX coach. Given the following normalized performance values for a race participant, analyze their performance and give 3 suggestions for improvement and 2 areas where they performed well.
    #     gender: {data.gender},
    #     age_min: {scaled_age[0]:.4f},
    #     age_max: {scaled_age[1]:.4f},
    #     total_time: {scaled_df['total_time'].iloc[0]:.4f},
    #     work_time: {scaled_df['work_time'].iloc[0]:.4f},
    #     roxzone_time: {scaled_df['roxzone_time'].iloc[0]:.4f},
    #     run_1: {scaled_df['run_1'].iloc[0]:.4f},
    #     run_2: {scaled_df['run_2'].iloc[0]:.4f},
    #     run_3: {scaled_df['run_3'].iloc[0]:.4f},
    #     run_4: {scaled_df['run_4'].iloc[0]:.4f},
    #     run_5: {scaled_df['run_5'].iloc[0]:.4f},
    #     run_6: {scaled_df['run_6'].iloc[0]:.4f},
    #     run_7: {scaled_df['run_7'].iloc[0]:.4f},
    #     run_8: {scaled_df['run_8'].iloc[0]:.4f},
    #     1000m Ski: {scaled_df['1000m Ski'].iloc[0]:.4f},
    #     50m Sled Push: {scaled_df['50m Sled Push'].iloc[0]:.4f},
    #     50m Sled Pull: {scaled_df['50m Sled Pull'].iloc[0]:.4f},
    #     80m Burpee Broad Jump: {scaled_df['80m Burpee Broad Jump'].iloc[0]:.4f},
    #     1000m Row: {scaled_df['1000m Row'].iloc[0]:.4f},
    #     200m Farmer Carry: {scaled_df['200m Farmer Carry'].iloc[0]:.4f},
    #     100m Sandbag Lunges: {scaled_df['100m Sandbag Lunges'].iloc[0]:.4f},
    #     100 Wall Balls: {scaled_df['100 Wall Balls'].iloc[0]:.4f},
    #     roxzone_1: {scaled_df['roxzone_1'].iloc[0]:.4f},
    #     roxzone_2: {scaled_df['roxzone_2'].iloc[0]:.4f},
    #     roxzone_3: {scaled_df['roxzone_3'].iloc[0]:.4f},
    #     roxzone_4: {scaled_df['roxzone_4'].iloc[0]:.4f},
    #     roxzone_5: {scaled_df['roxzone_5'].iloc[0]:.4f},
    #     roxzone_6: {scaled_df['roxzone_6'].iloc[0]:.4f},
    #     roxzone_7: {scaled_df['roxzone_7'].iloc[0]:.4f},
    #     ### Response:"""
    # return prompt

def analyze_improvement_from_similar_perf_only(user_df: pd.DataFrame, similar_df: pd.DataFrame) -> dict:
    """
    Compares the user's weakest stations (from suggestions_perf_only) with those of a similar participant
    and estimates potential improvement.

    Returns a dict mapping station names to estimated improvement in z-score.
    """
    user_row = user_df.iloc[0]
    similar_row = similar_df.iloc[0]
    weak_stations = user_row["suggestions_perf_only"]

    improvement_dict = {}

    for station in weak_stations:
        station_key = f"{station}_zscore"
        if station_key in similar_df.columns:
            user_score = user_row[station_key]
            similar_score = similar_row[station_key]

            if similar_score < user_score:
                estimated_improvement = user_score - similar_score
                improvement_dict[station] = round(estimated_improvement, 2)

    return improvement_dict


def analyze_improvement_from_similar_perf_context(user_df: pd.DataFrame, similar_df: pd.DataFrame) -> dict:
    """
    Same logic as above but using suggestions_perf_context and zscore_context columns.

    Parameters:
    user_df : pd.DataFrame
        DataFrame containing the processed user data (single row).
    similar_df : pd.DataFrame
        DataFrame containing the most similar participant data from vector store.

    Returns:
    dict
        Dictionary mapping station names to estimated improvement in minutes.
    """
    user_row = user_df.iloc[0]
    similar_row = similar_df.iloc[0]

    weak_stations = user_row["suggestions_perf_context"]

    improvement_dict = {}

    for station in weak_stations:
        station_key = f"{station}_zscore_context"
        if station_key in similar_df.columns:
            user_score = user_row[station_key]
            similar_score = similar_row[station_key]

            if similar_score < user_score:
                estimated_improvement = user_row[station] - similar_row[station]
                improvement_dict[station] = round(estimated_improvement, 2)

    return improvement_dict
