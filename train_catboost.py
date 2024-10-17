import joblib

from tqdm import tqdm, trange
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import precision_recall_curve, auc, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

from dataloader import BatchDataLoader, load_preprocessed, load_preprocessed_partial
from generate_features import (
    FeaturePipeline,
    TFIDFGenerator,
    CategorySimGenerator,
    CharacteristicGenerator,
    BERT64embdsGenerator,
    ResNetGenerator,
    ArticleGenerator,
    NumAttributesSimGenerator,
    CategoryBERTGenerator,
    BERTGenerator,
)

N_TRAIN_BATCHES = 5
MODEL_PATH = "weights/catboost/model.cbm"
TFIDF_VECTORIZER_PATH = "weights/catboost/vectorizer.pkl"
NUM_ATTRIBUTES_PATH = "weights/num_attributes.csv"
DATA_PATH = "data/train_batched"

PREPROCESSED_DATA_PATH = "data/train_preprocessed"
USE_PREPROCESSED = False
USE_PARTIAL = False


def init_model() -> CatBoostClassifier:
    _best_grid_search_params = {"depth": 14, "iterations": 300, "learning_rate": 0.1}
    _best_grid_search_params["iterations"] = 50  #
    model = CatBoostClassifier(
        task_type="CPU",
        eval_metric="PRAUC",
        od_type="Iter", 
        od_wait=100,
        **_best_grid_search_params,
    )
    return model


def evaluate_model(model, X_val, y_val):
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    prauc = auc(recall, precision)
    print(f"PRAUC: {prauc}")


def get_catboost_features_pipeline_7_features():
    # олдовые фичи из 7 чиселок
    generators = [
        TFIDFGenerator(TFIDF_VECTORIZER_PATH),
        CategorySimGenerator(),
        CharacteristicGenerator(),
        BERT64embdsGenerator(),
        ResNetGenerator(),
    ]

    return FeaturePipeline(generators)


def get_catboost_features_pipeline():
    # функция-конфиг для фичей
    generators = [
        NumAttributesSimGenerator(NUM_ATTRIBUTES_PATH, use_cos=True),
        ArticleGenerator(),
        CategorySimGenerator(),
        CharacteristicGenerator(),
        ResNetGenerator(use_raw_embds=False),
        BERTGenerator(),
        CategoryBERTGenerator(),
    ]

    return FeaturePipeline(generators)


def get_catboost_feautures_pipeline_additive():
    # функция-конфиг для фичей
    generators = [
        BERT64embdsGenerator(use_raw_embds=True),
        ResNetGenerator(use_raw_embds=True),
    ]

    return FeaturePipeline(generators)


def grid_search():
    # only for cached
    if not USE_PREPROCESSED:
        raise ValueError("Only preprocessed")

    x_train, y_train = load_preprocessed(PREPROCESSED_DATA_PATH, 0)
    for i in range(1, N_TRAIN_BATCHES):
        cur_x, cur_y = load_preprocessed(PREPROCESSED_DATA_PATH, i)
        x_train = np.concatenate([x_train, cur_x], axis=0)
        y_train = np.concatenate([y_train, cur_y], axis=0)

    y_train = y_train.reshape(-1, 1)
    x_test, y_test = load_preprocessed(PREPROCESSED_DATA_PATH, N_TRAIN_BATCHES)

    model = CatBoostClassifier(
        task_type="CPU", loss_function="Logloss", random_state=2003
    )

    param_grid = {
        "iterations": [200, 300],
        "learning_rate": [0.1, 0.2, 0.3],
        "depth": [8, 10, 12, 14, 16],
        "silent": [True],
    }
    # param_grid = {
    #     'iterations': [100],
    #     'learning_rate': [0.01],
    #     'depth': [3],
    #     'silent': [True]

    # }
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring="average_precision", n_jobs=-1, verbose=3
    )
    grid_search.fit(x_train, y_train)

    print("Grid Search - Best Hyperparameters:", grid_search.best_params_)
    print("Grid Search best score:", grid_search.best_score_)

    final_model = CatBoostClassifier(
        task_type="CPU",
        loss_function="Logloss",
        random_state=2003,
        **grid_search.best_params_,
    )
    final_model.fit(x_train, y_train)
    final_model.save_model("weights/catboost/grid_search_best.cbm")
    evaluate_model(final_model, x_test, y_test)
    joblib.dump(grid_search.cv_results_, "analysis/grid_search_results.pkl")


def main():
    model = init_model()
    feature_generator = get_catboost_features_pipeline()
    loader = BatchDataLoader(DATA_PATH, N_TRAIN_BATCHES + 1)  # +1 for test

    batch_ind = 0
    for batch_ind in trange(6):
        if USE_PREPROCESSED:
            if USE_PARTIAL:
                X_train, y_train = load_preprocessed_partial(
                    PREPROCESSED_DATA_PATH, batch_ind, feature_generator
                )
            else:
                X_train, y_train = load_preprocessed(PREPROCESSED_DATA_PATH, batch_ind)
        else:
#             merged_batch = loader.get(batch_ind, sample=0.001)
            merged_batch = loader[batch_ind]

            X_train = feature_generator.generate(merged_batch)
            y_train = merged_batch["target"]

        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_train, y_train, test_size=0.2
        # )
        # eval_pool = Pool(X_val, y_val)
        if batch_ind == 0:
            model.fit(
                X_train,
                y_train,
                # eval_set=eval_pool,
            )
        else:
            model.fit(X_train, y_train, init_model=MODEL_PATH)
            # model.fit(X_train, y_train, eval_set=eval_pool, init_model=MODEL_PATH)
        model.save_model(MODEL_PATH)

    # print("Evaluate model...")
    # model = init_model()
    # model.load_model(MODEL_PATH)
    # if USE_PREPROCESSED:
    #     if USE_PARTIAL:
    #         x_test, y_test = load_preprocessed_partial(
    #             PREPROCESSED_DATA_PATH, N_TRAIN_BATCHES, feature_generator
    #         )
    #     else:
    #         x_test, y_test = load_preprocessed(PREPROCESSED_DATA_PATH, N_TRAIN_BATCHES)
    # else:
    #     test_batch = loader[N_TRAIN_BATCHES]  # last batch is test
    #     x_test = feature_generator.generate(test_batch)
    #     y_test = test_batch["target"]

    # evaluate_model(model, x_test, y_test)
    # PRAUC: 0.96 переобучился
    # PRAUC: 0.897371972003053
    # PRAUC: 0.9295457315430298

    # PRAUC: 0.9209474587732038
    # PRAUC: 0.9215624372097894


if __name__ == "__main__":
    # grid_search()
    main()
