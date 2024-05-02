import os
import csv
import xml.etree.ElementTree as ET
from hazm import Normalizer
from textblob import TextBlob

# Define a function to normalize Persian text
def normalize_persian_text(text):
    if text is None:
        return ""
    normalizer = Normalizer()
    normalized_text = normalizer.normalize(text)
    return normalized_text

# Function to parse XML file and extract messages
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    messages = []

    for message in root.findall('.//Message'):
        msg_text = message.find('Msg').text
        normalized_text = normalize_persian_text(msg_text)
        sentiment = message.find('Sentiment').text if message.find('Sentiment') is not None else None
        messages.append((normalized_text, sentiment))
    
    return messages

# Function to perform sentiment analysis on Persian text
def analyze_sentiment(message, sentiment):
    if sentiment == 'Buy':
        sentiment_score = 0.5
    elif sentiment == 'Sell':
        sentiment_score = -0.5
    else:
        blob = TextBlob(message)
        sentiment_score = blob.sentiment.polarity
    # Classify sentiment scores
    if sentiment_score > 0:
        sentiment_label = 'Positive'
    elif sentiment_score < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = None
    return sentiment_score, sentiment_label

# Function to write results to CSV file
def write_to_csv(file_path, data):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Message', 'Sentiment Score', 'Sentiment'])
        writer.writerows(data)

# Main function
def main():
    input_dir = 'C:\\Users\\ASUS\\Desktop\\elham\\dataset\\FA_DataSet_XML\\khodro\\train'
    output_dir = 'C:\\Users\\ASUS\\Desktop\\elham\\dataset\\FA_DataSet_XML\\khodro\\train_out'
    total_results = 0

    # Process each XML file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(input_dir, filename)
            messages = parse_xml(file_path)
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output.csv")

            # Analyze sentiment for each message
            sentiments = []
            for msg, sentiment in messages:
                sentiment_score, sentiment_label = analyze_sentiment(msg, sentiment)
                if sentiment_label is not None:
                    sentiments.append((msg, sentiment_score, sentiment_label))

            # Write sentiment results to CSV file
            write_to_csv(output_file, sentiments)
            total_results += len(sentiments)

    print(f"Total Results: {total_results}")

if __name__ == "__main__":
    main()
