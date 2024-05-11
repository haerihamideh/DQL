import pandas as pd
import os

# Function to process a single input file
def process_data(input_file):
    # Load raw data
    raw_data = pd.read_csv(input_file)

    # Calculate the previous 70 days' volume for each row
    for i in range(1, 71):
        raw_data[f'FIRST_{i}'] = raw_data['<FIRST>'].shift(i)

    # Drop rows with NaN values due to shifting
    raw_data.dropna(inplace=True)

    # Select required columns
    processed_data = raw_data[['<DTYYYYMMDD>','<FIRST>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>', '<OPEN>'] + [f'FIRST_{i}' for i in range(1, 71)]]

    return processed_data