import pandas as pd
import os

# Function to process a single pair of stock and polarity data
def merge_data(stock_data_path, polarity_data_directory, output_directory):
    # Load processed stock data
    processed_stock_data = pd.read_csv(stock_data_path)

    # Extract stock name from the stock data path
    stock_name = os.path.basename(stock_data_path).split('.')[0]

    # Construct path to polarity data for the current stock
    polarity_data_path = os.path.join(polarity_data_directory, stock_name, 'daily_average_polarity.csv')

    # Check if polarity data file exists
    if os.path.exists(polarity_data_path):
        # Load average polarity data
        average_polarity_data = pd.read_csv(polarity_data_path)

        # Merge dataframes on the 'Date'/'DTYYYYMMDD' column with left join
        merged_data = pd.merge(processed_stock_data, average_polarity_data, left_on='<DTYYYYMMDD>', right_on='Date', how='left')

        # Fill missing values in 'Average_Polarity' column with zeros
        merged_data['Average_Polarity'].fillna(0, inplace=True)

    else:
        # If polarity data file does not exist, fill 'Average_Polarity' column with zeros
        merged_data = processed_stock_data
        merged_data['Average_Polarity'] = 0

    # Filter data for years 2016 and 2017
    merged_data['Year'] = merged_data['<DTYYYYMMDD>'].astype(str).str[:4]
    merged_data = merged_data[(merged_data['Year'] == '2016') | (merged_data['Year'] == '2015')]

    # Sort the data by date
    merged_data.sort_values(by='<DTYYYYMMDD>', inplace=True)

    # Remove the 'Year' column
    merged_data.drop(columns=['Year'], inplace=True)

    # Edit column names to remove '<' and '>'
    merged_data.columns = merged_data.columns.str.replace('<', '').str.replace('>', '')

    # Save merged data to a new CSV file
    output_path = os.path.join(output_directory, f'{stock_name}_merged_data_filtered.csv')
    merged_data.to_csv(output_path, index=False)

    print(f"Processed data saved to: {output_path}")