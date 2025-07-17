import numpy as np
import pandas as pd
import random


def add_instruction_tuning_columns(df):
    input_templates = [
        "Generate personalized feedback for a HYROX Open Division participant based on their performance.",
        "Explain this HYROX Open Division participant's performance using their residual and cluster data.",
        "Write performance feedback for a HYROX Open participant given their actual and predicted times.",
        "What would you tell this HYROX Open athlete based on their data?",
    ]
    def build_context(row):
        try:
            return (
                f"gender: {row.get('gender', 'N/A')}, "
                f"age_range: {row.get('age_min', 'N/A')}-{row.get('age_max', 'N/A')}, "
                f"total_time: {int(row.get('total_time', -1))}, "
                f"predicted_time: {int(round(row.get('predicted_total_time', -1)))}, "
                f"residual: {int(round(row.get('residual', 0)))}, "
                f"cluster_perf_only: {row.get('cluster_perf_only', 'N/A')}, "
                f"cluster_perf_context: {'N/A' if row.get('cluster_perf_context', -1) == -1 else row.get('cluster_perf_context')}"
            )
        except Exception as e:
            return f"Error building context: {str(e)}"

    df = df.copy()
    df["input"] = [random.choice(input_templates) for _ in range(len(df))]
    df["context"] = df.apply(build_context, axis=1)
    df["response"] = df.get("performance_feedback", "")

    return df