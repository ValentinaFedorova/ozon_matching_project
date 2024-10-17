import json

import pandas as pd
from tqdm import trange
import re

PATH_NUM_ATTRIBUTES = "weights/num_attributes.csv"
TRAIN_FLAG = True

exclude_attributes = []
if TRAIN_FLAG:
    result = {}
    for i in trange(6):
        print(f"batch: {i}")
        attributes_path = f"./data/train_batched/batch{i}_attributes.parquet"
        attributes = pd.read_parquet(attributes_path)
        attributes["characteristic_attributes_mapping"] = attributes[
            "characteristic_attributes_mapping"
        ].apply(json.loads)
        characteristics = attributes["characteristic_attributes_mapping"].to_list()
        for characteristic in characteristics:
            for k, v in characteristic.items():
                if re.match(r"^\d+(\.\d+)?$", v[0]) and k not in exclude_attributes:
                    cur_amount = result.get(k)
                    if cur_amount:
                        result[k] = cur_amount + 1
                    else:
                        result[k] = 1
                elif result.get(k):
                    del result[k]
                    exclude_attributes.append(k)

    num_attributes = (
        pd.DataFrame.from_dict(result, orient="index", columns=["cnt"])
        .reset_index()
        .rename(columns={"index": "attribute"})
    )
    num_attributes.drop(
        num_attributes[num_attributes["attribute"] == "Артикул"].index, inplace=True
    )
    num_attributes.to_csv(PATH_NUM_ATTRIBUTES, sep=";", index=False)

else:
    num_attributes = pd.read_csv(PATH_NUM_ATTRIBUTES, sep=";")
    num_attributes.head()
