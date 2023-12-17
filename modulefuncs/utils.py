import csv
import os

def append_to_csv(filepath, values, header=None):
    # Function implementation
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists and header is not None:
            writer.writerow(header)
        
        writer.writerow(values)