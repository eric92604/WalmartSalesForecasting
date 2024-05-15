import main_functions
import logging
import traceback
from pathlib import Path

# Task: To forecast hierarchical sales for multiple Walmart stores in California, Texas, and Wisconsin.
# The dataset comes from Kaggle: M5 Forecasting - Accuracy Competition.

if __name__ == "__main__":
    source_path = Path(__file__).resolve()
    source_dir = str(source_path.parent)

    main_functions.LoggerSetup.setup_logging(source_dir)
    try:

        main_functions.main(source_dir,
                            test=True,  # Enable using much smaller testing dataset
                            timeseriesdataset_from_file=False,
                            # Load timeseriesdataset from file rather than generating it, cuts down initial load time
                            find_optimal_lr=False,  # Enable finding optimal learning rate
                            find_optimal_hyperparameters=False, # Enable finding optimal hyperparameters
                            find_optimal_batch_size=False,  # currently debugging variable batch size hparam
                            batch_size=1024,  # Batch size for data loaders
                            worker_size=4,  # Number of workers for data loaders
                            persistent_workers=True,  # Should leave as true unless disabling multiprocessing
                            num_epochs=10  # Number of epochs to fit model
                            )
    except Exception as err:
        logging.info({err})
        logging.error(traceback.format_exc())
