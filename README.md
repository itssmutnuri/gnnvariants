# Code for the GNN Variants project

## Setup Environment

To set up the environment for the GNN Variants project, use the provided `environment.yml` file. Create a Conda environment using the following commands:

```bash
conda env create -f environment.yml
conda activate projectGNN
```

This will create a Conda environment named projectGNN with the specified dependencies.

## Running the Main Script
The main script, main.py, is the entry point for your project. Execute it using the following command:

```bash
python main.py
Ensure you have activated the Conda environment (conda activate projectGNN) before running the script.
```
## Configuration
All variable configurations are specified in the config.json file. Adjust the values in this file according to your requirements:
- **is_graph:** Boolean indicating whether the data represents a graph (`true`) or not (`false`).

- **ADJ_bool:** Boolean indicating whether the graphs uses country adjacency to construct the adjacency matrix (`true`) or not (`false`).

- **Flights_bool:** Boolean indicating whether flights data is used to construct the adjacency matrix (`true`) or not (`false`).

- **self_loops:** Boolean indicating whether self-loops are allowed in the graph (`true`) or not (`false`).

- **EW_bool:** Boolean indicating whether edge weights are present in the graph (`true`) or not (`false`).

- **topX_bool:** Boolean indicating whether to filter noisy data by using data from the top X% of countries which reported cases (`true`) or not (`false`).

- **topX_C:** An integer specifying the value of X when `topX_bool` is `true`.

- **dom_thresh:** A floating-point value representing the dominance threshold.

- **use_r:** Boolean indicating whether to use 'r' (`true`) or 'p' (`false`).

- **use_S:** Boolean indicating whether to use 'S' (`true`) or not (`false`).

- **min_epochs:** Minimum number of training epochs.

- **max_epochs:** Maximum number of training epochs.

- **T:** An integer representing the value of timesteps to be used a features.

- **reg_bool:** Boolean indicating whether we are doing a regression (`true`) or classification (`false`).

- **early_stopper_patience:** An integer specifying the patience for early stopping.

- **early_stopper_delta:** An integer specifying the delta for early stopping.

- **variants_path:** Path to the file containing variant data (`data/all_vars21_clustered_NEW.csv`).

- **countries_path:** Path to the file containing country data (`data/countries_clustered.csv`).

- **device:** Device for computation (`"cpu"` or `"gpu"`). GPU IS CURRENTLY NOT SUPPORTED

- **ITERATION_NAME:** A string specifying the name of the iteration (`"GNN_Test"`) Results will be saved in a folder with this name followed by a timestamp.

- **IS_DEBUG:** Boolean indicating whether debugging mode is enabled (`true`) or not (`false`).

## Custom Models
Define custom models in the models.py file. The model to be used in main.py should be implemented as the class ModelM.

## Visualization of Results
After training, visualize the results using the viz_script.py file. Provide the directory where the results are stored by modifying the variables directory_name and csv_directory
