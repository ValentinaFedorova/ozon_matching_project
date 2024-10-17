import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pickle
import joblib
import json


def extract_text_from_row(row):

    category_text = " ".join(
        [
            " ".join(map(str, v)) if isinstance(v, list) else str(v)
            for v in list(row["categories"].values())
        ]
    )
    attributes_text = " ".join(
        [
            " ".join(map(str, v)) if isinstance(v, list) else str(v)
            for v in list(row["characteristic_attributes_mapping"].values())
        ]
    )
    return f"{category_text} {attributes_text}"


def process_attributes(df):
    df["categories"] = df["categories"].apply(json.loads)
    df["characteristic_attributes_mapping"] = df[
        "characteristic_attributes_mapping"
    ].apply(json.loads)
    df["combined_text"] = df.apply(extract_text_from_row, axis=1)
    return df


common_text = pd.Series()

for i in range(5):
    attributes_path = f"./data/train_batched/batch{i}_attributes.parquet"
    attributes = pd.read_parquet(attributes_path)
    attributes = process_attributes(attributes)
    common_text = pd.concat(
        [common_text, attributes["combined_text"]], ignore_index=True
    )

tfidf_vectorizer = TfidfVectorizer(max_features=3000)
tfidf_vectorizer.fit(common_text)
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")
