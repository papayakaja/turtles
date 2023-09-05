# ML Project -- Turtle rescue forecast

This repository contains our work on the ML project done during Spiced Academy's data science course.

The data used for this is: [turtle rescue forecast](https://zindi.africa/competitions/turtle-rescue-forecast-challenge) from Zindi.

The deliverables are:
* [slides](turtle_stakeholder_slides.pdf) for a stakeholder presentation
* [jupyter notebooks](notebooks/) for data science/technical audience
* scripts for generating and running model from the terminal -- see files final_script.sh and src/predict.py

---

## The project's Kanban board on github

[Kanban board](https://github.com/users/RiptideDS/projects/1) of this 

## Requirements and Environment

Requirements:
- pyenv with Python: 3.11.3

Environment: 
For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Data: 
Please download the .csv files containing the data as provided by Zindi, unzip them and put them into the subfolder data/ 


## Usage

In order to run the script that will
* load and clean the raw data
* perform some data wrangling and basic feature engineering 
* train the model and make predictions
* test and evaluate the predictions
* perform error analysis,
run

```bash
source final_script.sh
```

In order to test prediction on a test set you created run:

```bash
python3 src/predict.py models/cat_boost_model.sav data/X_test.csv data/y_test.csv
```


