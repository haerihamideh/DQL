import os
import pandas as pd

# Define the path to the directory containing CSV files
directory = r'C:\Users\ASUS\Desktop\elham\dataset\FA_DataSet_XML\khodro\train_out'

# Initialize a dictionary to store sentiment scores for each date
sentiment_scores = {}

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        
        # Extract the date from the filename
        date = filename.split('_')[1].split('_')[0]
        
        # Calculate the average sentiment score for each date
        avg_sentiment_score = df['Polarity'].mean()
        
        # Update the sentiment_scores dictionary
        if date in sentiment_scores:
            sentiment_scores[date].append(avg_sentiment_score)
        else:
            sentiment_scores[date] = [avg_sentiment_score]

# Calculate the overall average sentiment score for each date
average_sentiment_scores = {date.replace('-', ''): sum(scores) / len(scores) for date, scores in sentiment_scores.items()}

# Convert the dictionary to DataFrame
result_df = pd.DataFrame(list(average_sentiment_scores.items()), columns=['Date', 'Average Sentiment Score'])

# Write the DataFrame to a CSV file
output_file = 'daily_average_polarity.csv'
result_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
