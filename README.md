# Stock Market Trading Prediction Model with Persian Sentiment Analysis

## Overview

Welcome to the Stock Market Trading Prediction Model project! This project integrates sentiment analysis of Persian tweets related to stock market companies with deep learning techniques to predict stock market trends and facilitate trading decisions. By combining human sensibility from social media data with advanced machine learning algorithms, this project aims to provide traders with predictive models that can analyze historical stock data, identify patterns, and make accurate predictions about future market movements.

## Motivation

The stock market is known for its complexity and volatility, making it challenging for investors to make informed decisions. Traditional financial analysis methods may not capture the full spectrum of market sentiment and trends. By leveraging machine learning and sentiment analysis on Persian tweets, this project seeks to enhance traditional stock market prediction models with additional insights derived from human sentiment expressed on social media.

## Methodology

### Data Collection and Preprocessing

- Persian tweets related to specific stock market companies are collected from social media platforms.
- The collected tweets are preprocessed to extract relevant features and sentiments.
- The sentiment of each tweet is classified as positive or negative based on the sentiment expressed in the text.
- Polarity scores are assigned to each tweet based on the sentiment lexicon and user-defined sentiment labels.

### Sentiment Analysis

- Sentiment analysis is performed on the collected tweets to determine the overall sentiment towards each stock market company.
- The sentiment analysis results are aggregated to calculate daily average polarity scores for each company.

### Model Architecture

- LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) deep learning architectures are employed for stock market prediction.
- Both models are trained on historical stock market data along with the calculated daily average polarity scores.
- The models consist of multiple layers of LSTM or GRU cells, followed by dropout layers to prevent overfitting, and a dense output layer for classification.

### Training and Evaluation

- The models are trained on a subset of the historical data, with a portion reserved for validation to monitor training progress and prevent overfitting.
- Evaluation metrics such as accuracy, loss, and reward are used to assess the performance of the models on both the training and validation datasets.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- Keras
- NumPy
- Matplotlib

## Getting Started

To get started with the Stock Market Trading Prediction Model project, follow these steps:

1. Clone or download the project repository to your local machine.
2. Install the required dependencies mentioned in the Requirements section.
3. Run the preprocessing script to collect and preprocess Persian tweets, perform sentiment analysis, and calculate daily average polarity scores.
4. Run the `main.py` file using Python to train and evaluate the LSTM and GRU models for stock market prediction.

## Results

Upon executing the project, you can expect the following outcomes:

- Training and validation loss plots for LSTM and GRU models, illustrating the convergence of the models during training.
- Training reward plots for LSTM and GRU models, showcasing the performance of each model over epochs.
- Evaluation metrics such as accuracy and loss for both the LSTM and GRU models.
- Daily average polarity scores for each stock market company, providing insights into market sentiment derived from Persian tweets.
