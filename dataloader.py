import pickle
import random
import json
from os.path import join

import pandas as pd
import numpy as np

from generate_features import FeaturePipeline


class BatchDataLoader:
    def __init__(self, path: str, n_batches: int) -> None:
        self.n_batches = n_batches
        self.path = path

    def _load_data(self, batch_ind: str):
        attributes_path = join(self.path, f"batch{batch_ind}_attributes.parquet")
        resnet_path = join(self.path, f"batch{batch_ind}_resnet.pkl")
        text_and_bert_path = join(self.path, f"batch{batch_ind}_text_and_bert.parquet")
        train_path = join(self.path, f"batch{batch_ind}.parquet")

        attributes = pd.read_parquet(attributes_path)

        with open(resnet_path, "rb") as file:
            resnet_b = file.read()
        resnet = pickle.loads(resnet_b)
        resnet.drop(["Unnamed: 0"], axis=1, inplace=True)
        resnet["variantid"] = resnet["variantid"].astype("Int64")

        text_and_bert = pd.read_parquet(text_and_bert_path)
        pairs = pd.read_parquet(train_path)

        return attributes, resnet, text_and_bert, pairs

    def __len__(self):
        return self.n_batches

    def merge_data(self, train, resnet, attributes, text_and_bert):
        train_data = train.merge(
            resnet[
                [
                    "variantid",
                    "main_pic_embeddings_resnet_v1",
                    "pic_embeddings_resnet_v1",
                ]
            ],
            left_on="variantid1",
            right_on="variantid",
            how="left",
        )
        train_data["pic_embeddings_resnet_v1"] = (
            train_data["pic_embeddings_resnet_v1"].fillna("").apply(list)
        )
        train_data = train_data.rename(
            columns={
                "main_pic_embeddings_resnet_v1": "pic_embeddings_1",
                "pic_embeddings_resnet_v1": "additional_pic_embeddings_1",
            }
        )
        train_data = train_data.drop(columns=["variantid"])

        train_data = train_data.merge(
            resnet[
                [
                    "variantid",
                    "main_pic_embeddings_resnet_v1",
                    "pic_embeddings_resnet_v1",
                ]
            ],
            left_on="variantid2",
            right_on="variantid",
            how="left",
        )
        train_data["pic_embeddings_resnet_v1"] = (
            train_data["pic_embeddings_resnet_v1"].fillna("").apply(list)
        )
        train_data = train_data.rename(
            columns={
                "main_pic_embeddings_resnet_v1": "pic_embeddings_2",
                "pic_embeddings_resnet_v1": "additional_pic_embeddings_2",
            }
        )
        train_data = train_data.drop(columns=["variantid"])

        train_data = train_data.merge(
            attributes,
            left_on="variantid1",
            right_on="variantid",
            how="left",
        )
        train_data = train_data.rename(
            columns={
                "combined_text": "text_1",
                "categories": "categories_1",
                "characteristic_attributes_mapping": "characteristic_1",
            }
        )

        train_data = train_data.drop(columns=["variantid"])

        train_data = train_data.merge(
            attributes,
            left_on="variantid2",
            right_on="variantid",
            how="left",
        )
        train_data = train_data.rename(
            columns={
                "combined_text": "text_2",
                "categories": "categories_2",
                "characteristic_attributes_mapping": "characteristic_2",
            }
        )
        train_data = train_data.drop(columns=["variantid"])

        train_data = train_data.merge(
            text_and_bert,
            left_on="variantid1",
            right_on="variantid",
            how="left",
        )
        train_data = train_data.rename(
            columns={
                "name": "name_1",
                "description": "description_1",
                "name_bert_64": "name_bert_64_1",
            }
        )
        train_data = train_data.drop(columns=["variantid"])

        train_data = train_data.merge(
            text_and_bert,
            left_on="variantid2",
            right_on="variantid",
            how="left",
        )
        train_data = train_data.rename(
            columns={
                "name": "name_2",
                "description": "description_2",
                "name_bert_64": "name_bert_64_2",
            }
        )
        train_data = train_data.drop(columns=["variantid"])

        # train_data = train_data.dropna()

        return train_data

    def get(self, batch_ind: int, sample: float = None):
        attributes, resnet, text_and_bert, pairs = self._load_data(batch_ind)
        if sample is not None:
            pairs = pairs.sample(frac=sample)
        attributes = _process_attributes(attributes)
        train_data = self.merge_data(pairs, resnet, attributes, text_and_bert)
        return train_data

    def __getitem__(self, key: int):
        return self.get(key)


class SubmissionDataLoader(BatchDataLoader):
    ATTRIBUTES_PATH = "./data/test/attributes_test.parquet"
    RESNET_PATH = "./data/test/resnet_test.parquet"
    TEXT_AND_BERT_PATH = "./data/test/text_and_bert_test.parquet"
    VAL_PATH = "./data/test/test.parquet"

    def __init__(self) -> None:
        pass

    def _resnet_to_normal(self, resnet: pd.DataFrame):
        # np.vstack(resnet.pic_embeddings_resnet_v1.iloc[0]).shape
        resnet.pic_embeddings_resnet_v1 = resnet.pic_embeddings_resnet_v1.apply(
            lambda x: None if x is None else np.vstack(x)
        )
        resnet.main_pic_embeddings_resnet_v1 = (
            resnet.main_pic_embeddings_resnet_v1.apply(lambda x: x[0])
        )
        return resnet

    def get(self):
        resnet = pd.read_parquet(self.RESNET_PATH, engine="pyarrow")
        resnet = self._resnet_to_normal(resnet)
        attributes = pd.read_parquet(self.ATTRIBUTES_PATH, engine="pyarrow")
        text_and_bert = pd.read_parquet(self.TEXT_AND_BERT_PATH, engine="pyarrow")
        pairs = pd.read_parquet(self.VAL_PATH, engine="pyarrow")
        attributes = _process_attributes(attributes)
        x = self.merge_data(pairs, resnet, attributes, text_and_bert)
        return x, pairs


class TestSubmissionDataLoader(SubmissionDataLoader):
    ATTRIBUTES_PATH = "./data/train_batched/batch5_attributes.parquet"
    RESNET_PATH = "./data/tmp.parquet"
    TEXT_AND_BERT_PATH = "./data/train_batched/batch5_text_and_bert.parquet"
    VAL_PATH = "./data/train_batched/batch5.parquet"

    def get(self):
        resnet = pd.read_parquet(self.RESNET_PATH, engine="pyarrow")
        resnet = self._resnet_to_normal(resnet)
        attributes = pd.read_parquet(self.ATTRIBUTES_PATH, engine="pyarrow")
        text_and_bert = pd.read_parquet(self.TEXT_AND_BERT_PATH, engine="pyarrow")
        pairs = pd.read_parquet(self.VAL_PATH, engine="pyarrow").iloc[:500]
        resnet.variantid = np.unique(
            np.concatenate([pairs["variantid1"], pairs["variantid2"]])
        )

        attributes = _process_attributes(attributes)
        x = self.merge_data(pairs, resnet, attributes, text_and_bert)

        return x, pairs


def load_preprocessed(path: str, batch_ind: int):
    # loads cached preprocessed merged df
    # return ready to train pair of x_train, target
    data = pd.read_parquet(f"{path}/batch_{batch_ind}.parquet")
    target = data["target"]
    data = data.drop("target", axis=1)
    return data.to_numpy(), target


def load_preprocessed_partial(
    path: str, batch_ind: int, new_feature_pipeline: FeaturePipeline
):
    merged_df = pd.read_parquet(f"{path}/batch_merged_{batch_ind}.parquet")
    new_x = new_feature_pipeline.generate(merged_df)
    del merged_df
    x, y = load_preprocessed(path, batch_ind)
    x = np.concatenate([x, new_x], axis=1)
    return x, y


def _process_attributes(df):
    df["categories"] = df["categories"].apply(json.loads)
    df["characteristic_attributes_mapping"] = df[
        "characteristic_attributes_mapping"
    ].apply(json.loads)
    df["combined_text"] = df.apply(_extract_text_from_row, axis=1)
    return df


def _extract_text_from_row(row):

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
