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
All variable configurations are specified in the config.json file. Adjust the values in this file according to your requirements.

## Custom Models
Define custom models in the models.py file. The model to be used in main.py should be implemented as the class ModelM.

## Visualization of Results
After training, visualize the results using the viz_script.py file. Provide the directory where the results are stored by modifying the variables directory_name and csv_directory
