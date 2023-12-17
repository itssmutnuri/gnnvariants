import json

import os

script_dir = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(script_dir, 'config.json')


def load_config():
    with open(config_file_path, 'r') as file:
        return json.load(file)

config = load_config()

T = config.get('T', None)
variants = config.get('variants', [])
IS_DEBUG = config.get('IS_DEBUG', False)
test_var = config.get('test_var', False)