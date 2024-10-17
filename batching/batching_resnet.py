import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from batching_features import _get_unique_ids


def split_resnet(
    path_resnet: str, path: str, n_batches: int, resnet_batch_size: int = 4
):
    files = [os.path.join(path, f"batch{x}.parquet") for x in range(n_batches)] + [
        os.path.join(path, "test.parquet")
    ]
    names = [os.path.join(path, f"batch{i}_resnet.csv") for i in range(n_batches)] + [
        os.path.join(path, "test_resnet.csv")
    ]
    batches = [_get_unique_ids(pd.read_parquet(f)) for f in files]

    resnet_pq = pq.ParquetFile(path_resnet)

    # добавлять header только для первого батча
    is_first = [True for _ in range(len(batches))]
    for n in names:
        open(n, "w").close()

    bar = tqdm(total=resnet_pq.metadata.num_rows // resnet_batch_size)
    for resnet_batch in resnet_pq.iter_batches(batch_size=resnet_batch_size):
        resnet_df = resnet_batch.to_pandas()

        for i, batch in enumerate(batches):
            c_batch = resnet_df[resnet_df.variantid.isin(batch)].reset_index(drop=True)

            with open(names[i], "a") as f:
                if is_first[i]:
                    is_first[i] = False
                    c_batch.to_csv(f, index=False)
                else:
                    c_batch.to_csv(f, header="None", index=False)

        bar.update()


if __name__ == "__main__":
    path_resnet = "data/train/resnet.parquet"
    path = "data/train_batched"
    # 5 такое же как в batching_feautures.py
    split_resnet(path_resnet, path, 5, resnet_batch_size=50000)
