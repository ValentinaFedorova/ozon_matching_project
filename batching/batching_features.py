import os
import gc

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


def _split_df_pairs_batches(df: pd.DataFrame, n_batches: int, shuffle: bool = True):
    # splits df in given number batches
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    out = []
    for part in np.array_split(df.index, n_batches):
        out.append(df.iloc[part])
    return out


def _get_unique_ids(df_pairs: pd.DataFrame):
    # gets all unique variantid that are used in df with target
    return np.unique(np.concatenate([df_pairs["variantid1"], df_pairs["variantid2"]]))


def _get_batch_data(
    df_pairs: pd.DataFrame, attributes: pd.DataFrame, text_and_bert: pd.DataFrame
):
    unique_ids = _get_unique_ids(df_pairs)
    return (
        attributes[attributes.variantid.isin(unique_ids)],
        text_and_bert[text_and_bert.variantid.isin(unique_ids)],
    )


def split_train(path: str, save_path: str, n_batches: int, test_size=0.2):
    print("Begin reading ...")
    train_pairs = pd.read_parquet(os.path.join(path, "train.parquet"))
    attributes = pd.read_parquet(os.path.join(path, "attributes.parquet"))
    text_and_bert = pd.read_parquet(os.path.join(path, "text_and_bert.parquet"))
    print("End reading")

    # сплитим на тест и на трейн, считаем, что тест должен влезть в память :)
    train, test = train_test_split(
        train_pairs, test_size=test_size, stratify=train_pairs.target
    )

    test.reset_index(drop=True).to_parquet(os.path.join(save_path, "test.parquet"))

    # save test
    test_atrribute, test_text_and_bert = _get_batch_data(
        test, attributes, text_and_bert
    )
    test_atrribute.to_parquet(os.path.join(save_path, "test_atrributes.parquet"))
    test_text_and_bert.to_parquet(os.path.join(save_path, "test_text_and_bert.parquet"))

    # удалить прошлые ради памяти
    del test_atrribute
    del test_text_and_bert

    batches = _split_df_pairs_batches(train, n_batches)

    bar = tqdm(total=n_batches)
    for i, batch in enumerate(batches):
        batch_attribute, batch_text_and_bert = _get_batch_data(
            batch, attributes, text_and_bert
        )
        batch_attribute.to_parquet(
            os.path.join(save_path, f"batch{i}_attributes.parquet")
        )
        batch_text_and_bert.to_parquet(
            os.path.join(save_path, f"batch{i}_text_and_bert.parquet")
        )
        batch.to_parquet(os.path.join(save_path, f"batch{i}.parquet"))
        bar.update()


if __name__ == "__main__":
    split_train("data/train", "data/train_batched", 5)
