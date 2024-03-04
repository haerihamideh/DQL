import os
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict

# Directory paths
input_dir = 'C:\\Users\\ASUS\\Desktop\\elham\\dataset\\FA_DataSet_XML\\vnaft\\train'
output_dir_individual = 'C:\\Users\\ASUS\\Desktop\\elham\\dataset\\FA_DataSet_XML\\vnaft\\train_out'
output_dir_avg = 'C:\\Users\\ASUS\\Desktop\\elham\\dataset\\FA_DataSet_XML\\vnaft'
lexicon_csv_path = 'C:\\Users\\ASUS\\Desktop\\elham\\dataset\\FA_DataSet_XML\\persian_sentiment_lexicon.csv'

# Load the Persian sentiment lexicon from the CSV file
persian_sentiment_lexicon = {}
with open(lexicon_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        word, sentiment = row
        persian_sentiment_lexicon[word] = float(sentiment)

# Default polarity values for sentiment cases
DEFAULT_POLARITY = {
    'Positive': 0.1,  # Set a positive polarity for Positive sentiment
    'Negative': -0.1  # Set a negative polarity for Negative sentiment
}

# Create output directories if they don't exist
if not os.path.exists(output_dir_individual):
    os.makedirs(output_dir_individual)

if not os.path.exists(output_dir_avg):
    os.makedirs(output_dir_avg)

# Dictionary to store daily polarity totals and counts
daily_polarity_totals = defaultdict(float)
daily_polarity_counts = defaultdict(int)

# Process each XML file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.xml'):
        input_file_path = os.path.join(input_dir, filename)
        
        # Load the XML file
        tree = ET.parse(input_file_path)
        root = tree.getroot()

        # Open individual output CSV file for writing results
        output_file_individual_path = os.path.join(output_dir_individual, filename.replace('.xml', '_out.csv'))
        with open(output_file_individual_path, 'w', newline='', encoding='utf-8') as output_file_individual:
            # Create a CSV writer object for individual output
            csv_writer_individual = csv.writer(output_file_individual)
            csv_writer_individual.writerow(['Text', 'Sentiment', 'Polarity'])

            # Iterate through each message in the XML
            for message in root.findall('./Message'):
                msg_element = message.find('Msg')
                if msg_element is not None and msg_element.text:
                    text = msg_element.text.strip()
                    sentiment = message.find('Sentiment').text

                    # Perform sentiment analysis
                    if sentiment:
                        if sentiment == 'Buy':
                            label = 'Positive'
                            polarity = DEFAULT_POLARITY['Positive']
                        elif sentiment == 'Sell':
                            label = 'Negative'
                            polarity = DEFAULT_POLARITY['Negative']
                        else:
                            label = sentiment
                            polarity = None
                    else:
                        # If sentiment is not provided, analyze the sentiment of the message
                        words = text.split()
                        polarity = sum(persian_sentiment_lexicon.get(word, 0) for word in words)

                        # Classify sentiment as positive or negative based on polarity
                        if polarity > 0:
                            label = 'Positive'
                        else:
                            label = 'Negative'
                            if polarity == 0:
                                polarity = DEFAULT_POLARITY['Negative']

                    # Write the result to the individual output file
                    csv_writer_individual.writerow([text, label, polarity])
                else:
                    print(f"Warning: Message element or its text is missing in file {filename}")

                # Extract date from filename
                date = filename.split('_')[1].split('.')[0]

                # Accumulate polarity for the corresponding date
                daily_polarity_totals[date] += polarity
                daily_polarity_counts[date] += 1

# Calculate average polarity for each day
daily_average_polarity = {date: total / count for date, total in daily_polarity_totals.items() for count in daily_polarity_counts.values()}

# Write daily average polarity to a CSV file
output_csv_path_avg = os.path.join(output_dir_avg, 'daily_average_polarity.csv')
with open(output_csv_path_avg, 'w', newline='', encoding='utf-8') as csvfile:
    writer_avg = csv.writer(csvfile)
    writer_avg.writerow(['Date', 'Average_Polarity'])
    for date, average_polarity in daily_average_polarity.items():
        writer_avg.writerow([date, average_polarity])

print("Average polarity calculated and saved to 'daily_average_polarity.csv'.")
print("Individual sentiment analysis completed. Results saved in 'train_out' directory.")
