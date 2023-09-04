#!/bin/bash

# activate the virtual environment
source .venv/bin/activate

## Run the python scripts

# cleaning the data
python3 src/main_cleaning_notebook.py

# feature engineering etc.
python3 src/data_wrangling.py

# baseline model
python3 src/models_baseline.py

# best model for our case
python3 src/Miauboost.py