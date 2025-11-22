import pandas as pd
import joblib
from xgboost import XGBClassifier
from .feature_engineering import extract_features_from_window
from .windowing import get_window_indices

def load_model_and_features(model_path, feature_cols_path):
    model = XGBClassifier()
    model.load_model(model_path)
    feature_cols = joblib.load(feature_cols_path)
    return model, feature_cols

def anomaly_score_from_raw(df_raw, fs, model, feature_cols, win_sec=1.0):
    indices, win_size = get_window_indices(len(df_raw), fs, win_sec)
    scores = []

    for (start, end) in indices:
        window = df_raw.iloc[start:end]
        feats = extract_features_from_window(window, fs)
        x = pd.DataFrame([feats])[feature_cols]
        p_fault = model.predict_proba(x)[0,1]
        scores.append(p_fault)

    if not scores:
        return None, []

    avg_score = float(sum(scores) / len(scores))
    return avg_score, scores

