import os
import pandas as pd
import csv

BASE_PATH = 'Results'
EXPERIMENTS = ['Baseline_20231017_221644', 
               'woS_20231018_003757', 
               'T1_20231018_032927', 'T3_20231018_060100', 'T4_20231018_084943',
               'eS_linearval_20231018_152235','eS_medianval_20231018_120836', 
               'eS_sautoenc_20231018_174049', 'eS_sautoenc_20231024_174311']

EVALS = ['f11', 'MAE1', 'MAE2', 'date']

def write_experiment_data(experiments, base_path):
    output_folder = os.path.join(base_path, f"combined_{pd.Timestamp.now():%Y%m%d_%H%M%S}")
    os.makedirs(output_folder, exist_ok=True)
    
    sample_folder = os.path.join(base_path, experiments[0])
    files = [f for f in os.listdir(sample_folder) if os.path.isfile(os.path.join(sample_folder, f)) and f != "status.csv"]
    
    for file_name in files:
        combined_df = pd.DataFrame()
        for experiment in experiments:
            file_path = os.path.join(base_path, experiment, file_name)
            df = pd.read_csv(file_path)
            for evalV in EVALS:
                if evalV in df.columns:
                    combined_df[f"{evalV}_{experiment}"] = df[evalV]
                else:
                    print(f"'pred' column not found in file {file_path}. Skipping...")
        combined_df.to_csv(os.path.join(output_folder, file_name), index=False)

    return output_folder

def extract_info_to_csv(folder_path):
    # Get a list of CSV files in the folder and sort them
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])

    # Initialize empty lists to store data
    f11_values = []
    mae1_values = []
    mae2_values = []

    # Iterate through sorted filenames
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Extract the second to last row
        if len(df) < 2:
            # If the file has less than 2 rows, add zeros to the lists
            f11_values.append([0] * 8)
            mae1_values.append([0] * 8)
            mae2_values.append([0] * 8)
        else:
            second_to_last_row = df.iloc[-2]
            f11_values.append(second_to_last_row.filter(like='f11').tolist())
            mae1_values.append(second_to_last_row.filter(like='MAE1').tolist())
            mae2_values.append(second_to_last_row.filter(like='MAE2').tolist())

    # Calculate the differences between the first element and each element in the lists
    f11_diff = [[f11[0] - value for value in f11] for f11 in f11_values]
    mae1_diff = [[mae1[0] - value for value in mae1] for mae1 in mae1_values]
    mae2_diff = [[mae2[0] - value for value in mae2] for mae2 in mae2_values]

    # Create a DataFrame and save it to a CSV file
    output_df = pd.DataFrame({
        'filename': filenames,
        'f11_diff': f11_diff,
        'mae1_diff': mae1_diff,
        'mae2_diff': mae2_diff
    })

    output_csv = os.path.join(folder_path, 'status.csv')
    output_df.to_csv(output_csv, index=False)

# Call the functions
output_folder = write_experiment_data(EXPERIMENTS, BASE_PATH)
extract_info_to_csv(output_folder)

