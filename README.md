# RoxMind – AI Feedback System for HYROX Participants Open Division

This project extends my previous work on [Hyrox Performance Buddy](https://github.com/syllerim/hyrox-performance-buddy), with a deeper focus on:
- Residual-based performance analysis
- Cluster-aware feedback
- Z-score normalization (with and without user context, gender and age)
- Finshed Fine-tuning Hyrox_Fine_tuning_Mistral7B_LoRA.ipynb for tailored performance feedback which couldn't finisht in the previous work
- Semantic search using vector embeddings
- FastAPI integration to expose the full pipeline as an API endpoint, allowing any external system to send race input data and receive structured performance feedback suggestions

The final result is a full pipeline that takes structured HYROX race data from a participant, generates performance feedback, and suggests targeted areas of improvement by comparing with similar participants retrieved from a vector store.

---

## 🔍 Problem

HYROX participants often receive a single finish time, but no guidance on *why* they performed that way or *how* to improve. This project aims to:
- Predict expected performance (`total_time`)
- Detect underperforming stations via residuals and z-scores
- Deliver personalized, contextual feedback
- Retrieve similar athletes using a vector search, and suggest the stations that person performed better than user

---

## 🚀 Solution Overview

### 1. **ML Pipeline**

I created a structured pipeline to go from raw user input to performance feedback.
- **Field Identification** Normalization User Id, Age (min and max), gender
- **Preprocessing**: Normalization of input times with `MinMaxScaler`
- **Z-Score Analysis**:
  - Standardized performance per station
  - With and without demographic context (gender + age)
- **Clustering**: KMeans clustering of athlete profiles (4 clusters)
- **Prediction**: A `LightGBMRegressor` model trained to predict `total_time` to compare if user performed over or lower expectations
- **Residuals**: Difference between predicted and actual `total_time`
- **Feedback Generation**: Natural language generation using a fine-tuned Mistral model
- **Vector Search (RAG)**:
  - Races encoded into embeddings
  - Retrieval of similar participants based on semantic query
  - Comparison of similar athletes’ times to highlight where the user can realistically improve

All model training and metrics are logged via **MLflow**.

---

## 🧠 Model Training

`LightGBMRegressor` model (to predict total race time) trained using:
- Normalized features (MinMaxScaler) for better convergence
- Station-level metrics (excluding `name`, `nationality`, etc.)
- Logged and versioned using MLflow

> 📌 See `03_model_training.ipynb` for full training details.

---

## ✍🏼 Fine-Tuning the LLM (Mistral)

The final feedback is generated using a **LoRA fine-tuned Mistral 7B** model. Trained it using:
- Instruction-tuning style prompts with `### Instruction`, `### Context`, `### Response`
- Input includes predicted time, residuals, and clusters
- Data split, tokenized, and trained using HuggingFace `Trainer`
- Prompt examples are stored in a single `text` field
- Labels are automatically generated from the `input_ids`

> 📌 See `05_fine_tuning_Mistral7B_LoRA.ipynb` for details. 🤗 [huggingface - last pushed model](https://huggingface.co/Syllerim/hyrox_mistral_lora_model/commit/60f2e5177b0df4a3aa756221e1942f23a6c48d80)

---

## 📦 Project Structure

```bash
roxmind/
├── notebooks/
│   ├── 01_load_explore_clean.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_fine_tuning.ipynb
│   └── 05_fine_tuning_Mistral7B_LoRA.ipynb
├── roxmind_api/
│   ├── feedback_logic.py       # End-to-end pipeline logic
│   ├── main.py                 # FastAPI app
├── src/
│   ├── clustering.py           # KMeans & z-score logic
│   ├── data_utils.py           # Preprocessing, scaling, helpers
│   ├── models.py               # Pydantic model for input schema
│   ├── pipeline.py             # Full pipeline from request to final df
│   └── vector_store.py         # ChromaDB vector embedding logic
├── data/
│   ├── processed/
│   │   ├── hyrox_full_data_instructed_feedback.csv
│   │   ├── hyrox_minmax_scaler_stations.pkl
│   │   └── hyrox_minmax_scaler_ages.pkl
├── mlruns/                     # MLflow logs

🚧 Next Steps
	•	Build a frontend (e.g., Streamlit or Client App) to allow users to input their race data and visualize feedback interactively.
	•	Expand vector search capabilities by supporting multiple embedding models or hybrid search strategies.
	•	Integrate OpenAI for production-grade RAG: Enable users to ask questions in natural language and leverage Retrieval-Augmented Generation (RAG) using OpenAI’s API for more dynamic, conversational feedback and insights.
