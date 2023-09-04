#!/bin/bash

# activate the virtual environment
source .venv/bin/activate

## Run the python scripts

# cleaning the data
python3 main_cleaning_notebook.py

# feature engineering etc.
python3 data_wrangling.py

# baseline model
python3 models_baseline.py

# best model for our case
python3 Miauboost.py