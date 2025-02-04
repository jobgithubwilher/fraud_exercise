# Training Notebook

This notebook creates a ML (Machine Learning) model to identify fraudsters. According to
a fraud detection experts, fraudulent orders can often only be identified in the context
of other orders.

Having used the identified fraud patterns, it is built a simple, elegant and performant
ML algorithm that accepts or rejects orders in real-time. 

## Setup

- Install [Anaconda](https://www.anaconda.com/)
- Set the channel priority to strict to avoid issues with the environment creation taking long time.
  - `conda config --set channel_priority strict`
- Run the following commands (in either the terminal or an Anaconda Prompt):
  - `conda env create -f golden_scenario_env.yml`
  - `conda activate golden_scenario_env`

In VS Code, open the [training_notebook.ipynb](training_notebook.ipynb) file and connect to the golden_scenario_env kernel

You need to setup the environment as an `ipykernel` to use it from the Jupyter notebook. To do it run inside of the conda activated environment:

`python -m ipykernel install --user --name golden_scenarios_env --display-name "Golden Scenarios Env"`


# Fraud Detection Exercise

## Overview
This project focuses on detecting fraudulent transactions using machine learning. It includes data preprocessing, feature engineering, model training, and evaluation.

## Project Structure

fraud_exercise/
├── artifacts/                  # Trained models (RandomForest, XGBoost)
├── data/                       # Dataset (data.parquet)
├── utils/                      # Utility scripts for data processing and modeling
│   ├── data_enrich.py          # Feature enrichment
│   ├── data_profiling.py       # Exploratory data analysis
│   ├── feature_discrimination.py # Feature selection based on discrimination
│   ├── feature_selection.py    # Feature importance calculations
│   ├── model_trainer.py        # Training models (RandomForest, XGBoost)
│   ├── train_test_splitter.py  # Splitting dataset into train and test
├── training_notebook.ipynb     # Jupyter notebook for model training
├── golden_scenario_env.yml     # Conda environment setup
├── README.md                   # Project documentation
├── challenge.pdf               # Exercise description

## 0. Installation

**Using Conda:**

```
conda env create -f golden_scenario_env.yml
conda activate fraud_env  # Adjust the environment name if needed
```

## 1. Training Notebook

For a detailed overview of the training methodology, open [training_notebook.ipynb](training_notebook.ipynb). The process begins with **data discovery**, assessing data quality and structure, followed by **feature engineering** to enhance predictive power. **Statistical analysis and visualization** provide insights into variable relationships and distributions.

To prevent overfitting, **feature selection algorithms** identify the most relevant predictors. The notebook then evaluates **three methodologies** to determine the optimal machine learning model. Finally, **Explainable AI (XAI) techniques**, including **permutation feature importance** and **SHAP values**, ensure transparency by explaining how the model makes decisions.

This structured approach balances model performance with interpretability, guiding the development of a robust predictive framework