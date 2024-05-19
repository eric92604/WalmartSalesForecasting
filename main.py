import main_functions
import logging
import traceback
from pathlib import Path

# Task: To forecast hierarchical sales for multiple Walmart stores in California, Texas, and Wisconsin.
# The dataset comes from Kaggle: M5 Forecasting - Accuracy Competition.
# Enable tensorboard dashboard through "python -m tensorboard.main --logdir=lightning_logs/"

if __name__ == "__main__":
    source_path = Path(__file__).resolve()
    source_dir = str(source_path.parent)

    main_functions.LoggerSetup.setup_logging(source_dir)
    try:

        main_functions.main(source_dir,
                            test=False,  # Enable using much smaller testing dataset
                            timeseriesdataset_from_file=True,
                            # Load timeseriesdataset from file rather than generating it, cuts down initial load time
                            find_optimal_lr=False,  # Enable finding optimal learning rate
                            find_optimal_hyperparameters=False,  # Enable finding optimal hyperparameters
                            find_optimal_batch_size=False,  # currently debugging variable batch size hparam
                            from_checkpoint=True,
                            batch_size=1024,  # Batch size for data loaders
                            worker_size=8,  # Number of workers for data loaders
                            persistent_workers=True,  # Should leave as true unless disabling multiprocessing
                            num_epochs=50  # Number of epochs to fit model
                            )
    except Exception as err:
        logging.info({err})
        logging.error(traceback.format_exc())
