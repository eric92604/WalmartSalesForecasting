import os
import sys
import re
from pathlib import Path
import logging
import time
import numpy as np
import pandas as pd
import pickle
import torch
import lightning.pytorch as pl
from lightning.pytorch.tuner.tuning import Tuner
from pytorch_forecasting import (
    Baseline,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    NaNLabelEncoder,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters
)


class LoggerSetup:
    @staticmethod
    def setup_logging(source_dir):
        log_path = source_dir + "\\main.log"
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s ",
        )  # Format log file
        logging.getLogger().addHandler(
            logging.StreamHandler(sys.stdout)
        )  # Print log to console
        logging.getLogger("matplotlib.font_manager").setLevel(
            logging.ERROR
        )  # Suppress debug messages from matplotlib
        logging.getLogger('PIL').setLevel(
            logging.WARNING
        )# Suppress debug messages from PIL


class DataManager:
    @staticmethod
    def downcast(df: pd.DataFrame) -> pd.DataFrame:
        # Downcast the datasets to speed up training
        # int8 /uint8: consumes 1 byte of memory, range between -128/127 or 0/255
        # bool: consumes 1 byte, true or false
        # float16 / int16 / uint16: consumes 2 bytes of memory, range between -32768 and 32767 or 0/65535
        # float32 / int32 / uint32: consumes 4 bytes of memory, range between -2147483648 and 2147483647
        # float64 / int64 / uint64: consumes 8 bytes of memory

        # Log initial memory usage

        initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # in MB
        start_time = time.time()

        for col, dtype in df.dtypes.items():
            if "int" in dtype.name:
                if (
                    df[col].min() > np.iinfo(np.int8).min
                    and df[col].max() < np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    df[col].min() > np.iinfo(np.int16).min
                    and df[col].max() < np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    df[col].min() > np.iinfo(np.int32).min
                    and df[col].max() < np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
            elif "float" in dtype.name:
                if (
                    np.finfo(np.float32).min < df[col].min() < np.finfo(np.float32).max
                    and np.finfo(np.float32).min
                    < df[col].max()
                    < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
            elif dtype == object:
                if col == "date":
                    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")
                elif col == "d":
                    df[col] = df[col].astype("str")
                else:
                    df[col] = df[col].astype("category")
        # Calculate and log memory savings

        final_memory = df.memory_usage(deep=True).sum() / 1024**2  # in MB
        memory_reduced = initial_memory - final_memory
        time_taken = time.time() - start_time

        logging.info(
            f"Downcasting completed: Reduced memory usage by {memory_reduced:.2f} MB in {time_taken:.2f} seconds."
        )
        return df

    def load_and_preprocess_data(
        self, data_dir
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        # Load data files

        start_time = time.time()

        df_calendar = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
        df_sales_train = pd.read_csv(
            os.path.join(data_dir, "sales_train_validation.csv")
        )
        df_sales_test = pd.read_csv(
            os.path.join(data_dir, "sales_train_evaluation.csv")
        )
        df_sales_prices = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))

        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")

        # Downcasting to reduce memory usage

        df_calendar = self.downcast(df_calendar)
        df_sales_train = self.downcast(df_sales_train)
        df_sales_test = self.downcast(df_sales_test)
        df_sales_prices = self.downcast(df_sales_prices)

        return df_calendar, df_sales_train, df_sales_test, df_sales_prices

    def training_data_melt_merge(
        self,
        df_calendar: pd.DataFrame,
        df_sales_training: pd.DataFrame,
        df_sales_prices: pd.DataFrame,
    ) -> pd.DataFrame:

        # Melt sales data to long format and merge with calendar and prices data

        start_time = time.time()

        df_sales_training.columns = df_sales_training.columns.str.replace(
            "^d_", "", regex=True
        )  # Remove d_ designation from days to allow for time_id column to be an integer.

        df_calendar["d"] = df_calendar["d"].replace({"d_": ""}, regex=True)
        # Similarly, remove it in calendar to allow for the left join

        df_training = pd.melt(
            df_sales_training,
            id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            var_name="d",
            value_name="sold",
        )
        # Melt the dataframe keeping the id variables of id, item_id, dept_id, cat_id,store_id, and state_id with
        # days and amount sold as values.

        df_training = pd.merge(df_training, df_calendar, on="d", how="left")
        df_training = pd.merge(
            df_training,
            df_sales_prices,
            on=["item_id", "store_id", "wm_yr_wk"],
            how="left",
        )
        df_training["d"] = df_training["d"].astype(int)

        for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            df_training[column] = df_training[column].astype(str)
        # Handle missing prices by imputing with the mean price per item

        mean_sell_prices = df_training.groupby("item_id", observed=True)[
            "sell_price"
        ].transform("mean")
        df_training["sell_price"] = df_training["sell_price"].fillna(
            mean_sell_prices.round(2)
        )

        df_training = self.downcast(df_training)

        df_training["sold"] = pd.to_numeric(df_training["d"]).astype("float32")
        # TimeSeriesDataSet preprocessing requires float32 for values

        logging.info(
            f"Data combined and cleaned in {time.time() - start_time:.2f} seconds."
        )

        return df_training

    def training_to_timeseriesdataset(
        self, df_training: pd.DataFrame
    ) -> TimeSeriesDataSet:
        # Load training data into Pytorch TimeSeriesDataSet

        start_time = time.time()
        training = TimeSeriesDataSet(
            df_training,
            group_ids=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            target="sold",
            time_idx="d",
            min_encoder_length=5,
            max_encoder_length=5,
            min_prediction_length=2,
            max_prediction_length=2,
            static_categoricals=[
                "id",
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
            ],
            time_varying_known_categoricals=[
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "weekday",
            ],
            time_varying_known_reals=[
                "sell_price",
                "wday",
                "month",
                "year",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=["sold"],
            target_normalizer=GroupNormalizer(
                groups=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                transformation="softplus",
            ),
            categorical_encoders={
                "event_name_1": NaNLabelEncoder(add_nan=True),
                "event_type_1": NaNLabelEncoder(add_nan=True),
                "event_name_2": NaNLabelEncoder(add_nan=True),
                "event_type_2": NaNLabelEncoder(add_nan=True),
                "sell_price": NaNLabelEncoder(add_nan=True),
            },
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        logging.info(
            f"Training data loaded into TimeSeriesDataSet in {time.time() - start_time:.2f} seconds."
        )

        return training

    def validation_from_timeseriesdataset(
        self, training: TimeSeriesDataSet, df_training: pd.DataFrame
    ) -> TimeSeriesDataSet:
        # Create validation and set predict=True to predict the last max_prediction_length point in time for each series

        validation = TimeSeriesDataSet.from_dataset(
            training, df_training, predict=True, stop_randomization=True
        )
        return validation

    def training_from_file(
        self, training_path, validation_path
    ) -> (TimeSeriesDataSet, TimeSeriesDataSet):
        # Load training data from previously created TimeSeriesDataSet file

        training: TimeSeriesDataSet = TimeSeriesDataSet.load(training_path)
        validation: TimeSeriesDataSet = TimeSeriesDataSet.load(validation_path)
        logging.info(
            f"Loaded training and validation timeseriesdataset from {training_path} and {validation_path}"
        )

        return training, validation

    def save_to_timeseriesdataset(
        self,
        training: TimeSeriesDataSet,
        training_path,
        validation: TimeSeriesDataSet,
        validation_path,
    ):
        # Saving training and validation TimeSeriesDataSet for re-use to speed up training process.

        TimeSeriesDataSet.save(training, training_path)
        TimeSeriesDataSet.save(validation, validation_path)
        logging.info(
            f"Training and validation timeseriesdataset saved to {training_path} and {validation_path}"
        )


class ModelSetup:
    def __init__(self, training: TimeSeriesDataSet, validation: TimeSeriesDataSet):
        self.training_dataset = training
        self.validation_dataset = validation

    def setup_trainer(self, num_epochs):
        if torch.cuda.is_available():  # Check if a GPU is available and set up
            trainer = pl.Trainer(
                max_epochs=num_epochs, gradient_clip_val=0.1, devices=1, accelerator="auto"
            )  # Enabling auto accelerator
            torch.set_float32_matmul_precision(
                "medium"
            )  # Setting matrix multiplication precision to medium
            logging.info("CUDA is available. GPU will be used for training.")
        else:
            trainer = pl.Trainer(max_epochs=10, gradient_clip_val=0.1)
            logging.info("CUDA is not available. Training will default to CPU.")
        return trainer

    def model_dataloader(self, batch_size=64, worker_size=4, persistent_workers=True):
        start_time = time.time()
        train_dataloader = self.training_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=worker_size,
            persistent_workers=persistent_workers,
            timeout=0,
            pin_memory=True,
        )  # Create training data loader for the model using pytorch dataloader
        logging.info(
            f"Training data loader created in {time.time() - start_time:.2f} seconds."
        )

        start_time = time.time()
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False,
            batch_size=batch_size * 10,
            num_workers=worker_size,
            persistent_workers=persistent_workers,
            timeout=0,
            pin_memory=True
        )  # Create validation data loader for the model using pytorch dataloader
        logging.info(
            f"Validation data loader created in {time.time() - start_time:.2f} seconds."
        )

        return train_dataloader, val_dataloader


class ModelTrainer:
    def __init__(self, training, trainer, train_dataloader, val_dataloader):
        self.training_dataset = training
        self.trainer = trainer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def optimal_learning_rate(
        self,
        hidden_size=8,
        hidden_continuous_size=8,
        min_lr=1e-6,
        max_lr=10,
        num_training=100,
        mode="exponential",
        show_fig=False,
    ):
        # Find the optimal learning rate

        start_time = time.time()
        pl.seed_everything(42, workers=True)
        tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.177827,
            hidden_size=hidden_size,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=1,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=hidden_continuous_size,  # set to <= hidden_size
            loss=QuantileLoss(),
            optimizer="Ranger",
            log_interval=1,
        )
        logging.info(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

        results = Tuner(self.trainer).lr_find(
            tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            max_lr=max_lr,
            min_lr=min_lr,
            num_training=num_training,
            mode=mode,
        )
        logging.info(
            f"Optimal learning rate found in {time.time() - start_time:.2f} seconds."
        )
        suggested_lr = results.suggestion()

        if show_fig:
            fig = results.plot(suggest=True)
            fig.show()
        logging.info(f"Suggested optimal learning rate: {suggested_lr}")

        return suggested_lr

    def optimize_hyperparameters(self):
        # Hyperparameter optimization with optuna

        start_time = time.time()
        study = optimize_hyperparameters(
            self.train_dataloader,
            self.val_dataloader,
            model_path="optuna_test",
            n_trials=200,
            max_epochs=50,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
            # use Optuna to find ideal learning rate or use in-built learning rate finder
        )

        with open("test_study.pkl", "wb") as fout:
            pickle.dump(study, fout)  # save study results

        logging.info(
            f"Optimal hyperparameters found in {time.time() - start_time:.2f} seconds."
        )
        logging.info(study.best_trial.params)  # show best hyperparameters

        return study.best_trial.params

    def train_model(self, hidden_size=8, hidden_continuous_size=8):
        # Train the TemporalFusionTransformer model w/ seed 42

        start_time = time.time()
        pl.seed_everything(42, workers=True)

        tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.03,
            hidden_size=hidden_size,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=1,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=hidden_continuous_size,  # set to <= hidden_size
            loss=QuantileLoss(),
            optimizer="Ranger",
            log_interval=1,
        )
        logging.info(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
        logging.info(f"Model trained in {time.time() - start_time:.2f} seconds.")
        self.trainer.fit(tft, self.train_dataloader, self.val_dataloader)

        return


def main(
    source_dir,
    timeseriesdataset_from_file: bool = False,
    find_optimal_lr: bool = False,
    find_optimal_hyperparameters: bool = False,
    batch_size=64,
    worker_size=4,
    persistent_workers=True,
    num_epochs=10,
    hidden_size=8,
    hidden_continuous_size=8,
    min_lr=1e-6,
    max_lr=10,
    num_training=100,
    mode="exponential",
):

    training_path = source_dir + "\\training_timeseriesdataset"
    validation_path = source_dir + "\\validation_timeseriesdataset"

    if timeseriesdataset_from_file:
        # Load pre-processed timeseriesdataset from file

        training_timeseriesdataset, validation_timeseriesdataset = (
            DataManager().training_from_file(training_path, validation_path)
        )
    else:
        df_calendar, df_sales_train, df_sales_test, df_sales_prices = (
            DataManager().load_and_preprocess_data(source_dir + "\\data")
        )
        # Load data into dataframes

        df_training = DataManager().training_data_melt_merge(
            df_calendar, df_sales_train, df_sales_prices
        )
        # Melt training data and join calendar and sales prices data

        training_timeseriesdataset = DataManager().training_to_timeseriesdataset(
            df_training
        )
        # Convert training data into a TimeSeriesDataSet

        validation_timeseriesdataset = DataManager().validation_from_timeseriesdataset(
            training_timeseriesdataset, df_training
        )
        # Create validation data

        DataManager().save_to_timeseriesdataset(
            training_timeseriesdataset,
            training_path,
            validation_timeseriesdataset,
            validation_path,
        )
        # Save training and validation timeseriesdataset to file

        del df_calendar, df_sales_train, df_sales_prices, df_training
    model_setup = ModelSetup(training_timeseriesdataset, validation_timeseriesdataset)
    trainer = model_setup.setup_trainer(num_epochs)
    train_dataloader, val_dataloader = model_setup.model_dataloader(
        batch_size, worker_size, persistent_workers
    )

    model_trainer = ModelTrainer(
        training_timeseriesdataset, trainer, train_dataloader, val_dataloader
    )

    optimal_lr = 0.03   # Initialize a reasonable learning rate
    if find_optimal_lr:
        try:
            optimal_lr = model_trainer.optimal_learning_rate(
                hidden_size=hidden_size,
                hidden_continuous_size=hidden_continuous_size,
                min_lr=min_lr,
                max_lr=max_lr,
                num_training=num_training,
                mode=mode,
                show_fig=False,
            )
        except Exception as err:
            logging.info(err)
    logging.info(f"Learning Rate set to {optimal_lr}")

    if find_optimal_hyperparameters:
        best_trial_params = model_trainer.optimize_hyperparameters()
    model_trainer.train_model()  # Fit network

    # load the best model according to the validation loss

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # calculate mean absolute error on validation set

    predictions = best_tft.predict(
        val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu")
    )
    MAE()(predictions.output, predictions.y)
