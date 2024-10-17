import os
import pickle
from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm


def clean_main_pic(x: str):
    _str = x[1:-1].strip().removeprefix("array(")[0:-1].replace("\n", "")
    return np.array(literal_eval(_str))


def clean_add_pics(x):
    if x is np.nan:
        return x
    _tstr = (
        x.replace("array", ",").replace("(", "[").replace(")", "]").replace("\n", "")
    )
    _tstr = _tstr[:1] + _tstr[2:]
    emb = np.array(eval(_tstr)).squeeze(1)
    return emb


def clean_resnet(batch_df):
    batch_df = batch_df[batch_df.variantid != "variantid"]
    batch_df.main_pic_embeddings_resnet_v1 = (
        batch_df.main_pic_embeddings_resnet_v1.apply(clean_main_pic)
    )
    batch_df.pic_embeddings_resnet_v1 = batch_df.pic_embeddings_resnet_v1.apply(
        clean_add_pics
    )
    return batch_df


def clean(path: str):
    fnames = list(filter(lambda x: x.endswith("_resnet.csv"), os.listdir(path)))
    bar = tqdm(total=len(fnames))
    for i, fname in enumerate(fnames):
        with open(os.path.join(path, fname), "r") as f:
            cur = pd.read_csv(f)

        cur = clean_resnet(cur)

        with open(os.path.join(path, fname.replace(".csv", ".pkl")), "wb") as f:
            pickle.dump(cur, f)
        bar.update()


if __name__ == "__main__":
    clean("data/train_batched")
