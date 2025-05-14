import pandas as pd
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(file_path):
    return pd.read_csv(file_path)