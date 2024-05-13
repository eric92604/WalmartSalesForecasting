from time import time
import multiprocessing as mp
import sys
import logging
import traceback
import time
import numpy as np
import pandas as pd
from pathlib import Path
from pytorch_forecasting import (
    TimeSeriesDataSet,
    NaNLabelEncoder,
)
from pytorch_forecasting.data import GroupNormalizer


# Task: To forecast hierarchical sales for multiple Walmart stores in California, Texas, and Wisconsin.
# The dataset comes from Kaggle: M5 Forecasting - Accuracy Competition.


def downcast(df):
    # Downcast the datasets to speed up training
    # int8 /uint8: consumes 1 byte of memory, range between -128/127 or 0/255
    # bool: consumes 1 byte, true or false
    # float16 / int16 / uint16: consumes 2 bytes of memory, range between -32768 and 32767 or 0/65535
    # float32 / int32 / uint32: consumes 4 bytes of memory, range between -2147483648 and 2147483647
    # float64 / int64 / uint64: consumes 8 bytes of memory

    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    # Iterate through df and check if type can be coerced into a smaller type

    for i, t in enumerate(types):
        if "int" in str(t):
            if (
                df[cols[i]].min() > np.iinfo(np.int8).min
                and df[cols[i]].max() < np.iinfo(np.int8).max
            ):
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif (
                df[cols[i]].min() > np.iinfo(np.int16).min
                and df[cols[i]].max() < np.iinfo(np.int16).max
            ):
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif (
                df[cols[i]].min() > np.iinfo(np.int32).min
                and df[cols[i]].max() < np.iinfo(np.int32).max
            ):
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif "float" in str(t):
            if (
                df[cols[i]].min() > np.finfo(np.float32).min
                and df[cols[i]].max() < np.finfo(np.float32).max
            ):
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == object:
            if cols[i] == "date":
                df[cols[i]] = pd.to_datetime(df[cols[i]], format="%Y-%m-%d")
            else:
                df[cols[i]] = df[cols[i]].astype("category")
    return df


def main():
    # Start logging

    source_path = Path(__file__).resolve()
    source_dir = str(source_path.parent)
    log_path = source_dir + "\\worker optimizer.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s ",
    )  # Format log file
    logging.getLogger().addHandler(
        logging.StreamHandler(sys.stdout)
    )  # Print log to console

    # Read in data into dataframes
    try:
        training = TimeSeriesDataSet.load("C:\\Workspace\\Walmart_forecasting\\training_timeseriesdataset")
        for num_workers in range(2, mp.cpu_count(), 2):
            train_loader = training.to_dataloader(
                train=True,
                batch_size=64,
                num_workers=num_workers,
                persistent_workers=True,
                timeout=120,
                pin_memory=True
            )  # Create training data loader for the model using pytorch dataloader
            start = time.process_time()
            for epoch in range(1, 3):
                for i, data in enumerate(train_loader, 0):
                    pass
            end = time.process_time()
            print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

    except Exception as err:
        logging.info({err})
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()