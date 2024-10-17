import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from tqdm import trange, tqdm
from sklearn.metrics import precision_recall_curve, auc, f1_score


from dataloader import BatchDataLoader, load_preprocessed, load_preprocessed_partial
from generate_features import FeaturePipeline, TFIDFGenerator, CategorySimGenerator, CharacteristicGenerator, BERT64embdsGenerator, ResNetGenerator, NumAttributesSimGenerator, ArticleGenerator


TFIDF_VECTORIZER_PATH = "weights/catboost/vectorizer.pkl"
DATA_PATH = "data/train_batched"
N_TRAIN_BATCHES = 5
MODEL_PATH = "weights/torch/SimpleTorchNN.pth"
NUM_ATTRIBUTES_PATH = "weights/num_attributes.csv"

# использовать или нет preprocessed data
PREPROCESSED_DATA_PATH = "data/train_preprocessed"
USE_PREPROCESSED = True
USE_PARTIAL = False

# торчёвские параметры
LR = 0.001
BATCH_SIZE = 256
N_EPOCHS = 100

class SimpleTorchNN(nn.Module):
    # https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
    # гайд от самого pytorch

    def __init__(self, input_dim: int) -> None:
        super(SimpleTorchNN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)
    

class TorchDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    # тоже гайд от разрабов
    # но если кратко этот класс нужен для удобной работы с torch DataLoader

    def __init__(self, merged_df: pd.DataFrame, target: pd.DataFrame = None) -> None:
        super().__init__()
        self.merged_df = merged_df
        if not(target is None):
            self._has_target = True
            self.target = target.values
        else:
            self._has_target = False


    def __len__(self):
        return len(self.merged_df)

    def __getitem__(self, index):
        if self._has_target:
            return self.merged_df[index], self.target[index]
        else:
            return self.merged_df[index]
    

def init_model():
    # legacy для make_submission.py
    return SimpleTorchNN(3378)


def inference(features: np.array):
    model = init_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    dataset = TorchDataset(features)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    output = []
    bar = tqdm(val_loader)
    for x in bar:
        x = x.float()
        model_prob = model(x)
        model_prob = model_prob.flatten().cpu().detach().tolist()
        output = output + model_prob
        
    return output

def validation(model, val_loader, device, loss_fn):
    # ставит модель в режим инференса (нет backward propagation)
    model.eval()
    bar = tqdm(val_loader)
    avg_loss = 0
    i = 0
    predictions = []
    target = []
    for x, y in bar:
        target = target + y.flatten().tolist()
        x = x.float().to(device)
        y = y.float().to(device)
        model_prob = model(x)
        loss = loss_fn(model_prob.flatten(), y.flatten())
        avg_loss = (avg_loss * i + loss.cpu().item()) / (i + 1)
        predictions = predictions + model_prob.flatten().cpu().detach().tolist()
        # считаем PRAUC
    precision, recall, _ = precision_recall_curve(target, predictions)
    prauc = auc(recall, precision)
    return prauc, f1_score(target, np.array(predictions) > 0.5)


def train_on_batch(model, merged_df: np.array, target, batch_ind: int):
    # если есть видюха, используем её
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TorchDataset(merged_df, target)
    # Сделаем валидацию, чтобы не переобучиться
    train_ds, val_ds = random_split(dataset, [0.8, 0.2])
    # DataLoader удобный класс для побатчевой обработки данных, внутри себя он вызывает Dataset._getitem_
    # который мы описали сверху
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) # перемешиваем каждый раз порядок
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()  # binary cross entropy

    bar = tqdm(train_loader)
    bar.set_description(f"Batch: {batch_ind}")
    i = 0
    avg_train_loss = 0

    # ставит модель в режим обучения
    model.train()
    for x, y in bar:
        x = x.float().to(device)
        y = y.float().to(device)

        # обнуляем градиенты
        optimizer.zero_grad()

        # делаем forward и считаем вероятность для нашего батча
        model_prob = model(x)
        # считаем loss
        loss = loss_fn(model_prob.flatten(), y.flatten())
        # тот самый страшный backpropagation
        loss.backward()

        # меняем веса
        optimizer.step()

        avg_train_loss = (avg_train_loss * i + loss.cpu().item()) / (i + 1)
        bar.set_postfix({"avg loss": avg_train_loss})
        i += 1

    prauc, f1 = validation(model, val_loader, device, loss_fn)
    print("Validation PRAUC", prauc)
    print("Validation F1", f1)
    return prauc



def get_torch_features_pipeline():
    # функция-конфиг для фичей
    generators = [
        NumAttributesSimGenerator(NUM_ATTRIBUTES_PATH),
        ArticleGenerator(),
        TFIDFGenerator(TFIDF_VECTORIZER_PATH),
        CategorySimGenerator(),
        CharacteristicGenerator(),
        BERT64embdsGenerator(use_raw_embds=False),
        ResNetGenerator(use_raw_embds=False),
        BERT64embdsGenerator(use_raw_embds=True),
        ResNetGenerator(use_raw_embds=True),
    ]

    return FeaturePipeline(generators)


def get_torch_feautures_pipeline_additive():
    # функция-конфиг для фичей
    generators = [
        BERT64embdsGenerator(use_raw_embds=True),
        ResNetGenerator(use_raw_embds=True)
    ]

    return FeaturePipeline(generators)


def main():
    # очень костыльно

    feature_pipeline = get_torch_features_pipeline()
    loader = BatchDataLoader(DATA_PATH, N_TRAIN_BATCHES + 1) # +1 for test
    cnt = 0
    patience = 10
    best_metric = -1
    model = init_model()
    for epoch in range(N_EPOCHS):
        metrics = []
        print(f"{epoch+1}/{N_EPOCHS}")
        for batch_ind in range(5):
            if USE_PREPROCESSED:
                if USE_PARTIAL:
                    X_train, y_train = load_preprocessed_partial(PREPROCESSED_DATA_PATH, batch_ind)
                else:
                    X_train, y_train = load_preprocessed(PREPROCESSED_DATA_PATH, batch_ind)
            else:
                merged_batch = loader[batch_ind]
                X_train = feature_pipeline.generate(merged_batch)
                y_train = merged_batch["target"]

            metric = train_on_batch(model, X_train, y_train, batch_ind)
            metrics.append(metric)
           
        avg_metric = sum(metrics) / len(metrics)
        if avg_metric > best_metric:
            torch.save(model.state_dict(), MODEL_PATH)
            best_metric = avg_metric
            cnt = 0
        else:
            cnt += 1
        print(cnt, best_metric)
        if cnt > patience:
            print("Early stopping")
            break
    if USE_PREPROCESSED:
        if USE_PARTIAL:
            x_test, y_test = load_preprocessed(PREPROCESSED_DATA_PATH, N_TRAIN_BATCHES)
        else:
            x_test, y_test = load_preprocessed(PREPROCESSED_DATA_PATH, N_TRAIN_BATCHES)
    else:
        test_merged_batch = loader[N_TRAIN_BATCHES]
        x_test = feature_pipeline.generate(test_merged_batch)
        y_test = test_merged_batch["target"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TorchDataset(x_test, y_test)
    loss_fn = nn.BCELoss()  # binary cross entropy
    print("TEST")
    prauc = validation(model, DataLoader(dataset, batch_size=BATCH_SIZE), device, loss_fn)
    print("Test PRAUC:", prauc)
#     torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()