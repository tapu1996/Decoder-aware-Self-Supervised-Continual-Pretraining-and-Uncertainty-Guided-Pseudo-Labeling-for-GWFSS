import os
import pandas as pd
import numpy as np

folder = "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/predicted_csvs"

for fname in os.listdir(folder):
    if fname.endswith('.csv'):
        arr = pd.read_csv(os.path.join(folder, fname), header=None).to_numpy()
        assert arr.dtype == np.uint8 or arr.dtype == np.int64  # okay for int
        assert arr.shape == (512, 512), f"Wrong shape: {fname}"
        assert np.all((arr >= 0) & (arr <= 3)), f"Unexpected label in {fname}"
