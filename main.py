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
                            timeseriesdataset_from_file=True,
                            find_optimal_lr=False,
                            find_optimal_hyperparameters=True,
                            batch_size=128,
                            worker_size=4,
                            persistent_workers=True,
                            num_epochs=5
                            )
    except Exception as err:
        logging.info({err})
        logging.error(traceback.format_exc())
