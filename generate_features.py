import os
import re
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from tqdm import trange, tqdm
from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import StandardScaler


class FeatureGenerator:
    """
    абстрактный класс отвественный за одну конкретную фичу из наших данных
    """

    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        "надо возращать numpy array c dim 2: (количество строк, количество фич)"
        raise NotImplementedError()


class FeaturePipeline:
    """
    пайплан сборки различных генераторов фич, чтобы получить финальные данные
    """

    def __init__(self, features_generatos: list[FeatureGenerator]) -> None:
        self.features_generatos = features_generatos

    def generate(self, merged_df: pd.DataFrame) -> np.ndarray:
        print(type(self.features_generatos[0]))
        features = self.features_generatos[0].generate(merged_df)
        for i in range(1, len(self.features_generatos)):
            print(type(self.features_generatos[i]))
            c_feature = self.features_generatos[i].generate(merged_df)
            features = np.concatenate([features, c_feature], axis=1)

        return features


class TFIDFGenerator(FeatureGenerator):
    def __init__(self, tfidf_path: str) -> None:
        self.tfidf_vectorizer = joblib.load(tfidf_path)

    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:

        text_1 = self.tfidf_vectorizer.transform(merged_data["text_1"]).toarray()
        text_2 = self.tfidf_vectorizer.transform(merged_data["text_2"]).toarray()
        return np.array(count_vector_sim(text_1, text_2)).reshape(-1, 1)


class CategoryBERTGenerator(FeatureGenerator):
    def __init__(self, use_raw_embds: bool = False) -> None:
        self.use_raw_embds = use_raw_embds

    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        transformer = TransformerTextEmbeddings()
        text_1 = transformer.generate(merged_data, "text_1")
        text_2 = transformer.generate(merged_data, "text_2")
        if self.use_raw_embds:
            return np.concatenate([text_1, text_2], axis=1)
        return np.array(count_vector_sim(text_1, text_2)).reshape(-1, 1)


class BERTGenerator(FeatureGenerator):
    def __init__(self, use_raw_embds: bool = False) -> None:
        self.use_raw_embds = use_raw_embds
        self.transformer = TransformerTextEmbeddings()

    def _get_description(self, merged_data: pd.DataFrame):
        merged_data.loc[
            merged_data["description_1"].isna(),
            "description_1",
        ] = merged_data["name_1"]
        merged_data.loc[
            merged_data["description_2"].isna(),
            "description_2",
        ] = merged_data["name_2"]
        return merged_data["description_1"], merged_data["description_2"]

    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        name_1 = self.transformer.generate(merged_data, "name_1")
        name_2 = self.transformer.generate(merged_data, "name_2")
        merged_data["description_1"], merged_data["description_2"] = (
            self._get_description(merged_data)
        )
        desc_1 = self.transformer.generate(merged_data, "description_1")
        desc_2 = self.transformer.generate(merged_data, "description_2")
        if self.use_raw_embds:
            return np.concatenate([name_1, name_2, desc_1, desc_2], axis=1)
        else:
            name_embeddings_sim = np.array(count_vector_sim(name_1, name_2)).reshape(
                -1, 1
            )
            desc_embeddings_sim = count_pic_vector_sim(
                name_embeddings_sim[:, 0],
                desc_1.reshape(-1, 1, desc_1.shape[1]),
                desc_2.reshape(-1, 1, desc_2.shape[1]),
            )
            desc_embeddings_sim = np.array(desc_embeddings_sim).reshape(-1, 1)
            return np.concatenate([name_embeddings_sim, desc_embeddings_sim], axis=1)


class NumAttributesSimGenerator(FeatureGenerator):
    def __init__(self, num_attributes_path: str, use_cos=False) -> None:
        self.num_attributes = pd.read_csv(num_attributes_path, sep=";")[
            "attribute"
        ].values
        self.use_cos = use_cos

    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        characteristic_1 = merged_data["characteristic_1"].tolist()
        characteristic_2 = merged_data["characteristic_2"].tolist()
        characteristic_num_attributes_vector_1 = []
        characteristic_num_attributes_vector_2 = []
        for i in range(len(characteristic_1)):
            cur_num_attributes_vector = [0] * len(self.num_attributes)
            for k, v in characteristic_1[i].items():
                if k in self.num_attributes and re.match(
                    r"^\d+(\.\d+)?$", v[0].strip()
                ):
                    cur_num_attributes_vector[
                        np.where(self.num_attributes == k)[0][0]
                    ] = float(v[0].strip())
            characteristic_num_attributes_vector_1.append(cur_num_attributes_vector)

        for i in range(len(characteristic_2)):
            cur_num_attributes_vector = [0] * len(self.num_attributes)
            for k, v in characteristic_2[i].items():
                if k in self.num_attributes and re.match(
                    r"^\d+(\.\d+)?$", v[0].strip()
                ):
                    cur_num_attributes_vector[
                        np.where(self.num_attributes == k)[0][0]
                    ] = float(v[0].strip())
            characteristic_num_attributes_vector_2.append(cur_num_attributes_vector)

        scaler = StandardScaler()
        scaler.partial_fit(characteristic_num_attributes_vector_1)
        scaler.partial_fit(characteristic_num_attributes_vector_2)

        characteristic_num_attributes_vector_1 = scaler.transform(
            characteristic_num_attributes_vector_1
        )
        characteristic_num_attributes_vector_2 = scaler.transform(
            characteristic_num_attributes_vector_2
        )
        if self.use_cos:
            return np.array(count_vector_sim(characteristic_num_attributes_vector_1, characteristic_num_attributes_vector_2)).reshape(-1, 1)
        else:
            characteristic_num_attributes = np.concatenate(
                [
                    characteristic_num_attributes_vector_1,
                    characteristic_num_attributes_vector_2,
                ],
                axis=1,
            )

            return characteristic_num_attributes


class ArticleGenerator(FeatureGenerator):
    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        merged_data["article_1"] = merged_data["characteristic_1"].apply(
            lambda x: (
                x.get("Артикул")[0].lower() if x.get("Артикул") is not None else "undef"
            )
        )
        merged_data["article_2"] = merged_data["characteristic_2"].apply(
            lambda x: (
                x.get("Артикул")[0].lower() if x.get("Артикул") is not None else "undef"
            )
        )
        # print(
        #     "MATCHES ARTICLES EQUALS PERCENT: ",
        #     str(
        #         merged_data[
        #             (merged_data["target"] == 1)
        #             & (merged_data["article_1"] != "undef")
        #             & (merged_data["article_1"] == merged_data["article_2"])
        #         ].shape[0]
        #         / merged_data[(merged_data["target"] == 1)].shape[0]
        #     ),
        # )
        # print(
        #     "NOT MATCHES ARTICLES EQUALS PERCENT: ",
        #     str(
        #         merged_data[
        #             (merged_data["target"] == 0)
        #             & (merged_data["article_1"] != "undef")
        #             & (merged_data["article_1"] == merged_data["article_2"])
        #         ].shape[0]
        #         / merged_data[(merged_data["target"] == 0)].shape[0]
        #     ),
        # )
        article_1 = merged_data["article_1"].values
        article_2 = merged_data["article_2"].values
        condition_1 = article_1 == article_2
        condition_2 = article_1 != "undef"
        article_sim = (condition_1 & condition_2).astype(int).reshape(-1, 1)
        return article_sim


class CategorySimGenerator(FeatureGenerator):
    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        categories_1 = merged_data["categories_1"].tolist()
        categories_2 = merged_data["categories_2"].tolist()
        category_score = []
        for i in range(len(categories_1)):
            current_score = 0
            for k, v in categories_1[i].items():
                if v == categories_2[i].get(k):
                    current_score += 1
            category_score.append(current_score / len(categories_1[i].keys()))
        return np.array(category_score).reshape(-1, 1)


class CharacteristicGenerator(FeatureGenerator):
    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        characteristic_1 = merged_data["characteristic_1"].tolist()
        characteristic_2 = merged_data["characteristic_2"].tolist()
        characteristic_keys_score = []
        characteristic_values_score = []
        for i in range(len(characteristic_1)):
            current_keys_score = 0
            current_values_score = 0
            all_keys = len(
                set(list(characteristic_1[i].keys()) + list(characteristic_2[i].keys()))
            )
            for k, values1 in characteristic_1[i].items():
                if characteristic_2[i].get(k):
                    current_keys_score += 1
                    values1 = [clean_text(x) for x in values1]
                    values2 = [clean_text(x) for x in characteristic_2[i].get(k)]
                    hasMatch = False
                    for ind1 in range(len(values1)):
                        for ind2 in range(len(values2)):
                            if values1[ind1].isdigit() and values2[ind2].isdigit():
                                if values1[ind1] == values2[ind2]:
                                    hasMatch = True
                            else:
                                if fuzz.ratio(values1[ind1], values2[ind2]) >= 85:
                                    hasMatch = True
                    if hasMatch:
                        current_values_score += 1
            characteristic_keys_score.append(current_keys_score / all_keys)
            if current_keys_score == 0:
                characteristic_values_score.append(0)
            else:
                characteristic_values_score.append(
                    current_values_score / current_keys_score
                )

        characteristic_keys_score = np.array(characteristic_keys_score).reshape(-1, 1)
        characteristic_values_score = np.array(characteristic_values_score).reshape(
            -1, 1
        )
        return np.concatenate(
            [characteristic_keys_score, characteristic_values_score], axis=1
        )


class TransformerTextEmbeddings(FeatureGenerator):
    def __init__(self, model="distiluse-base-multilingual-cased") -> None:
        # distiluse-base-multilingual-cased
        # rubert-base-cased-conversational
        # self.tokenizer = BertTokenizer.from_pretrained(model) # 'bert-base-uncased'
        # self.model = BertModel.from_pretrained(model, output_hidden_states = True) #'bert-base-uncased'
        # self.column = column
        # super().__init__()
        if not os.path.exists("weights/super_bert") or len(os.listdir("weights/super_bert")) == 0:
            print("load from internet BERT")
            os.makedirs("weights/super_bert")
            self.model = SentenceTransformer(model)
            self.model.save("weights/super_bert")
        else:
            print("load from cache BERT")
            self.model = SentenceTransformer("weights/super_bert")
        

    def generate(self, merged_data: pd.DataFrame, column: str) -> np.ndarray:
        data = merged_data[column].tolist()
        embeddings = self.model.encode(data, show_progress_bar= True, batch_size=32)
        return embeddings


class BERT64embdsGenerator(FeatureGenerator):
    # TODO: добавить чтобы возращал, просто эмбединги или комбинацию
    def __init__(self, use_raw_embds: bool = False) -> None:
        self.use_raw_embds = use_raw_embds
        super().__init__()

    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        if self.use_raw_embds:
            embd1 = np.stack(merged_data["name_bert_64_1"].values)
            embd2 = np.stack(merged_data["name_bert_64_2"].values)
            out = np.concatenate([embd1, embd2], axis=1)
            return out
        else:
            return np.array(
                count_vector_sim(
                    np.stack(merged_data["name_bert_64_1"].values),
                    np.stack(merged_data["name_bert_64_2"].values),
                )
            ).reshape(-1, 1)


class ResNetGenerator(FeatureGenerator):
    def __init__(self, use_raw_embds: bool = False):
        self.use_raw_embds = use_raw_embds

    def _get_additional_pic(self, merged_data: pd.DataFrame, main_pics: np.ndarray):
        merged_data.loc[
            merged_data["additional_pic_embeddings_1"].apply(len) == 0,
            "additional_pic_embeddings_1",
        ] = merged_data["pic_embeddings_1"].apply(lambda x: [x])
        merged_data.loc[
            merged_data["additional_pic_embeddings_2"].apply(len) == 0,
            "additional_pic_embeddings_2",
        ] = merged_data["pic_embeddings_2"].apply(lambda x: [x])
        embds1 = np.stack(
            merged_data["additional_pic_embeddings_1"].apply(lambda x: x[0]).values
        )
        embds2 = np.stack(
            merged_data["additional_pic_embeddings_2"].apply(lambda x: x[0]).values
        )
        out = np.concatenate([embds1, embds2], axis=1)
        return out

    def generate(self, merged_data: pd.DataFrame) -> np.ndarray:
        if self.use_raw_embds:
            main_pics = np.concatenate(
                [
                    np.stack(merged_data["pic_embeddings_1"].values),
                    np.stack(merged_data["pic_embeddings_2"].values),
                ],
                axis=1,
            )
            add_pics = self._get_additional_pic(merged_data, main_pics)
            return np.concatenate([main_pics, add_pics], axis=1)
        else:
            main_pic_embeddings_sim = np.array(
                count_vector_sim(
                    np.stack(merged_data["pic_embeddings_1"].values),
                    np.stack(merged_data["pic_embeddings_2"].values),
                )
            ).reshape(-1, 1)
            additional_pic_embeddings_sim = count_pic_vector_sim(
                main_pic_embeddings_sim[:, 0],
                merged_data["additional_pic_embeddings_1"].to_numpy(),
                merged_data["additional_pic_embeddings_2"].to_numpy(),
            )
            additional_pic_embeddings_sim = np.array(
                additional_pic_embeddings_sim
            ).reshape(-1, 1)
            return np.concatenate(
                [main_pic_embeddings_sim, additional_pic_embeddings_sim], axis=1
            )


## utils functions below


def clean_text(text):
    text = re.sub(r"(?![\s])\W", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_vector_sim(vector1, vector2):
    return [
        cosine_similarity([vector1[i]], [vector2[i]])[0][0] for i in range(len(vector1))
    ]


def count_pic_vector_sim(
    main_vector_sim, add_vector1, add_vector2, verbose: bool = True
):
    result = []
    if verbose:
        print("Additional pic embeddings")
        iterator = trange(len(main_vector_sim))
    else:
        iterator = range(len(main_vector_sim))
    for ind in iterator:
        cur_vector_1 = add_vector1[ind]
        cur_vector_2 = add_vector2[ind]
        if len(cur_vector_1) > 0 and len(cur_vector_2) > 0:
            _tmp = cosine_similarity(cur_vector_1, cur_vector_2)
            result.append(np.max(_tmp.flatten()))
        else:
            result.append(main_vector_sim[ind])
    return result


# def count_characteristic_sim_score(data):


if __name__ == "__main__":
    # костыли для теста
    from train_catboost import DATA_PATH, N_TRAIN_BATCHES
    from dataloader import BatchDataLoader

    loader = BatchDataLoader(DATA_PATH, N_TRAIN_BATCHES + 1)  # +1 for test
    to_test = [TransformerTextEmbeddings("text_1")]
    print(FeaturePipeline(to_test).generate(loader.get(0, sample=0.1)).shape)
