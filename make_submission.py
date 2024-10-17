import pandas as pd

from dataloader import TestSubmissionDataLoader, SubmissionDataLoader
from generate_features import FeatureGenerator
from train_catboost import init_model, get_catboost_features_pipeline, MODEL_PATH
from train_torch import get_torch_features_pipeline, inference
from train_torch import MODEL_PATH as TORCH_MODEL_PATH


def submit_catboost():
    # ВАЖНО!!!! при заливке в мастер не должен быть TestSubmissionDataLoader
    # Должен быть SubmissionDataLoader!!!!
    loader = SubmissionDataLoader()
    feature_pipeline = get_catboost_features_pipeline()
    # pairs немного костыль
    x_test, test_pairs = loader.get()
    x_test = feature_pipeline.generate(x_test)
    model = init_model()
    model.load_model("weights/catboost/grid_search_best.cbm")
    predictions_prob = model.predict_proba(x_test)[:, 1]
    submission = pd.DataFrame(
        {
            "variantid1": test_pairs["variantid1"],
            "variantid2": test_pairs["variantid2"],
            "target": predictions_prob,
        }
    )
    submission.to_csv("./data/submission.csv", index=False)
    
    
def submit_torch():
    loader = SubmissionDataLoader()
    feature_pipeline = get_torch_features_pipeline()
    # pairs немного костыль
    x_test, test_pairs = loader.get()
    x_test = feature_pipeline.generate(x_test)
    predictions_prob = inference(x_test)
    submission = pd.DataFrame(
        {
            "variantid1": test_pairs["variantid1"],
            "variantid2": test_pairs["variantid2"],
            "target": predictions_prob,
        }
    )
    submission.to_csv("./data/submission.csv", index=False)

if __name__ == "__main__":
    submit_catboost()
