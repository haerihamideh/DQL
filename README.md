# Stock Market Trading Prediction Model with Persian Sentiment Analysis

## Overview

Welcome to the Stock Market Trading Prediction Model project! This project integrates sentiment analysis of Persian tweets related to stock market companies with deep learning techniques to predict stock market trends and facilitate trading decisions. By combining human sensibility from social media data with advanced machine learning algorithms, this project aims to provide traders with predictive models that can analyze historical stock data, identify patterns, and make accurate predictions about future market movements.

## Motivation

The stock market is known for its complexity and volatility, making it challenging for investors to make informed decisions. Traditional financial analysis methods may not capture the full spectrum of market sentiment and trends. By leveraging machine learning and sentiment analysis on Persian tweets, this project seeks to enhance traditional stock market prediction models with additional insights derived from human sentiment expressed on social media.

## Methodology

### Data Collection and Preprocessing

1. **Data Collection:** Persian tweets related to specific stock market companies are collected from social media platforms.
2. **Preprocessing:** The collected tweets are preprocessed to extract relevant features and sentiments.
3. **Sentiment Classification:** The sentiment of each tweet is classified as positive or negative based on the sentiment expressed in the text.
4. **Polarity Score Assignment:** Polarity scores are assigned to each tweet based on the sentiment lexicon and user-defined sentiment labels.

### Sentiment Analysis

1. **Overall Sentiment Determination:** Sentiment analysis is performed on the collected tweets to determine the overall sentiment towards each stock market company.
2. **Polarity Score Aggregation:** The sentiment analysis results are aggregated to calculate daily average polarity scores for each company.

### Learning Phase (Combination of Deep Q Learning and Sentiment Analysis)

During the learning phase, the project utilizes a combination of Deep Q Learning (DQL) and sentiment analysis in the reward function. This approach integrates insights from both historical market data and sentiment analysis of social media data to enhance the trading strategy. The reward function is designed to incentivize the model to make decisions that align with both market trends and sentiment analysis, leading to more informed trading decisions.

### Model Architecture

1. **LSTM and GRU Models:** Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) deep learning architectures are employed for stock market prediction.
2. **Training Data:** Both models are trained on historical stock market data along with the calculated daily average polarity scores.
3. **Model Layers:** The models consist of multiple layers of LSTM or GRU cells, followed by dropout layers to prevent overfitting, and a dense output layer for classification.

### Training and Evaluation

1. **Training Process:** The models are trained on a subset of the historical data, with a portion reserved for validation to monitor training progress and prevent overfitting.
2. **Evaluation Metrics:** Evaluation metrics such as accuracy, loss, and reward are used to assess the performance of the models on both the training and validation datasets.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- Keras
- NumPy
- Matplotlib

## Getting Started

To get started with the Stock Market Trading Prediction Model project, follow these steps:

1. **Clone or Download:** Clone or download the project repository to your local machine.
2. **Install Dependencies:** Install the required dependencies mentioned in the Requirements section.
3. **Preprocessing:** Run the preprocessing script to collect and preprocess Persian tweets, perform sentiment analysis, and calculate daily average polarity scores.
4. **Training and Evaluation:** Run the main.py file using Python to train and evaluate the LSTM and GRU models for stock market prediction.

## Results

Upon executing the project, you can expect the following outcomes:

For each stock market company:
- **three Models are implemented:**
  - LSTM Metrics:
  - GRU Metrics:
  - Combined of LSTM and GRU Metrics:
- **these metrics are generated:**
  - Loss
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)

These metrics provide insights into the performance of the LSTM, GRU, and combined models in predicting stock market trends, considering both historical data and sentiment analysis.

