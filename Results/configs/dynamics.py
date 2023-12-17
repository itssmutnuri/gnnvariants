import json
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(script_dir, 'config.json')

def load_config():
    with open(config_file_path, 'r') as file:
        config = json.load(file)
        for key, value in config.items():
            globals()[key] = value

load_config()