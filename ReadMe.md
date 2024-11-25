Stock Price Prediction with Sentiment Analysis
Overview
This project is a web application that predicts stock prices using historical stock data and sentiment analysis from news articles. It utilizes an LSTM neural network and provides an intuitive web interface for training and testing the model.

Features
Train Model: Upload sentiment data in JSON format and train the LSTM model.
Test Model: Test the model on specific stock tickers, with an option to optimize the model for a particular stock.
Image Gallery: View generated images from the training process, including sentiment analysis and stock price trends.
Loading Indicators: Visual feedback during long-running operations like training and testing.
Visualization: Graphs showing training loss, predicted vs. actual stock prices, and sentiment analysis.
Requirements
Python 3.7 or higher
See requirements.txt for a full list of dependencies.
Installation
Clone the Repository

git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
Install Dependencies
bash

pip install -r requirements.txt
Directory Structure
Ensure your project directory includes the following:

your_project/
├── app.py
├── main.py
├── model_functions.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── train.html
│   ├── test.html
│   ├── result.html
│   ├── train_result.html
│   └── gallery.html
├── static/
│   └── (generated images)
├── models/
│   └── (saved models and scalers)
├── data/
│   └── (your JSON data files)
├── requirements.txt
└── README.md
Usage
Starting the Application
Run the Flask app:

python app.py
Access the application at http://localhost:5000.

Training the Model
Navigate to Train Model.
Upload your sentiment JSON data files.
Click Start Training.
A loading indicator will appear during training.
Upon completion, you will be redirected to a page displaying the training loss graph.
Testing the Model
Navigate to Test Model.
Enter the stock ticker symbol (e.g., AAPL).
Optionally, check Optimize Model for this Stock.
Click Run Test.
A loading indicator will appear during testing.
Results will display evaluation metrics and a plot of predicted vs. actual prices.
Viewing Generated Images
Navigate to Image Gallery.

View images such as:

Article Counts: Number of news articles per stock.
Price, News, and Sentiment: Visualizations for each stock.

Acknowledgements
Flask: For the web framework.
TensorFlow: For building the neural network.
yfinance: For fetching stock data.
Matplotlib & Seaborn: For data visualization.
Bootstrap: For styling the web interface.
If you need any more assistance or have further questions, feel free to ask!