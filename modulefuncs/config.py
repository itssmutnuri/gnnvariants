import json
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(script_dir, '../config.json')

def load_config():
    with open(config_file_path, 'r') as file:
        return json.load(file)

config = load_config()

# Updated to match the keys from your JSON
is_graph = config.get('is_graph', True)
ADJ_bool = config.get('ADJ_bool', True)
Flights_bool = config.get('Flights_bool', False)
self_loops = config.get('self_loops', True)
EW_bool = config.get('EW_bool', True)
topX_bool = config.get('topX_bool', False)
topX_C = config.get('topX_C', 60)
dom_thresh = config.get('dom_thresh', 0.3333)
use_r = config.get('use_r', True)
use_S = config.get('use_S', True)
min_epochs = config.get('min_epochs', 3)
max_epochs = config.get('max_epochs', 100)
T = config.get('T', 4)
reg_bool = config.get('reg_bool', True)
early_stopper_patience = config.get('early_stopper_patience', 3)
early_stopper_delta = config.get('early_stopper_delta', 5)
variants_path = config.get('variants_path', 'data/all_vars21_clustered_NEW.csv')
countries_path = config.get('countries_path', 'data/countries_clustered.csv')
device = config.get('device', 'cpu')
ITERATION_NAME = config.get('ITERATION_NAME', 'GNN_Default')
IS_DEBUG = config.get('IS_DEBUG', False)
