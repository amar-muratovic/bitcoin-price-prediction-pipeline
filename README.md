# bitcoin-price-prediction-pipeline
 
This script is a machine learning pipeline for predicting the closing price of Bitcoin using time series data. 
It starts by reading in a csv file containing Bitcoin data, resampling it, and splitting it into training and testing sets. 
Then, it fits and predicts using four different models: LinearRegression, BayesianRidge, SVR and RandomForestRegressor. The script then takes the average of these predictions, and prints the result. 
Next, it performs a grid search to optimize the parameters of a linear regression model, and then trains a neural network using the Keras library. 
Finally, it makes predictions using this neural network, prints the predictions and the best parameters and scores of the grid search.

# How to use the Bitcoin Price Prediction Pipeline

## Prerequisites

* Python 3.x
* pandas
* sklearn
* keras
* requests

## Installation

1. Clone the repository or download the script

git clone https://github.com/your-username/Bitcoin-Price-Prediction-Pipeline.git


2. Install the required packages

pip install [package]


## Usage

1. Make sure you have a .csv file containing Bitcoin data in the same directory as the script. 
The file should have columns named "Date", "Open", "High", "Low", "Close" and "Volume".

2. Run the script by using the command:

python main.py


## Note

The script uses the provided data to train a model and make predictions. 
The predictions may not be accurate due to the volatility of the Bitcoin market. 
The script is for demonstration purpose only.

# Updates and suggestions are more than welcome!