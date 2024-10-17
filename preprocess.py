import os

import numpy as np
import pandas as pd

from dataloader import BatchDataLoader
from generate_features import FeaturePipeline, TFIDFGenerator, CategorySimGenerator, CharacteristicGenerator, BERT64embdsGenerator, ResNetGenerator
from train_catboost import get_catboost_features_pipeline, get_catboost_features_pipeline_7_features
from tqdm import trange

DATA_PATH = "data/train_batched"
PREPROCESS_PATH = "data/train_preprocessed"
N_TRAIN_BATCHES = 5

def preprocess_and_save():
    feature_generator = get_catboost_features_pipeline()
    if not os.path.exists(PREPROCESS_PATH):
        raise ValueError(f"{PREPROCESS_PATH} does not exists")
    loader = BatchDataLoader(DATA_PATH, N_TRAIN_BATCHES + 1) # +1 for test
    for batch_ind in trange(N_TRAIN_BATCHES + 1):
        merged_batch = loader.get(batch_ind)
        X_train = feature_generator.generate(merged_batch)
        y_train = merged_batch["target"]
        out = pd.DataFrame()
        out["target"] = y_train
        out[[f"feature_{i}" for i in range(X_train.shape[1])]] = X_train
        out.to_parquet(f"{PREPROCESS_PATH}/batch_{batch_ind}.parquet")
# 

def prepprocess_partial_and_save():
    # менять надо это штуку
    feature_generator = get_catboost_features_pipeline_7_features()
    if not os.path.exists(PREPROCESS_PATH):
        raise ValueError(f"{PREPROCESS_PATH} does not exists")
    loader = BatchDataLoader(DATA_PATH, N_TRAIN_BATCHES + 1) # +1 for test
    for batch_ind in trange(N_TRAIN_BATCHES + 1):
        merged_batch = loader.get(batch_ind)
        X_train = feature_generator.generate(merged_batch)
        y_train = merged_batch["target"]
        out = pd.DataFrame()
        out["target"] = y_train
        out[[f"feature_{i}" for i in range(X_train.shape[1])]] = X_train
        out.to_parquet(f"{PREPROCESS_PATH}/batch_{batch_ind}.parquet")
        merged_batch.to_parquet(f"{PREPROCESS_PATH}/batch_merged_{batch_ind}.parquet")


if __name__ == "__main__":
    preprocess_and_save()