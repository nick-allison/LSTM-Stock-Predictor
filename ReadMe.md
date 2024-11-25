# Stock Price Prediction with Sentiment Analysis

## Overview
This project is a web application that predicts stock prices using historical stock data and sentiment analysis from news articles. It utilizes an **LSTM neural network** and provides an intuitive web interface for training and testing the model.

---

## Features
- **Train Model:** Upload sentiment data in JSON format and train the LSTM model.
- **Test Model:** Test the model on specific stock tickers, with an option to optimize the model for a particular stock.
- **Image Gallery:** View generated images from the training process, including sentiment analysis and stock price trends.
- **Loading Indicators:** Visual feedback during long-running operations like training and testing.
- **Visualization:** Graphs showing training loss, predicted vs. actual stock prices, and sentiment analysis.

---

## Requirements
- **Python 3.7 or higher**
- See `requirements.txt` for a full list of dependencies.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction

## Install dependencies
pip install -r requirements.txt

## Directory structure
your_project/
├── app.py               # Main Flask application
├── main.py              # Script to initialize and handle data processing
├── model_functions.py   # LSTM model training and testing functions
├── templates/           # HTML templates for the web app
│   ├── base.html
│   ├── index.html
│   ├── train.html
│   ├── test.html
│   ├── result.html
│   ├── train_result.html
│   └── gallery.html
├── static/              # Static files for images, CSS, JS
│   └── generated/       # Folder for storing generated images
├── models/              # Directory to save trained models and scalers
│   ├── lstm_model.h5
│   └── scaler.pkl
├── data/                # Folder to store sentiment data in JSON
│   └── sample_data.json
├── requirements.txt     # Dependencies for the project
└── README.md            # Project documentation

## Usage
Starting the Application
Run the Flask app:
python app.py

Access the application at http://localhost:5000.

## Training the Model
1. Navigate to Train Model.
2. Upload your sentiment JSON data files.
3. Click Start Training.
4. A loading indicator will appear during training.
5. Upon completion, you will be redirected to a page displaying the training loss graph.

## Testing the Model
1. Navigate to Test Model.
2. Enter the stock ticker symbol (e.g., AAPL).
3. Optionally, check Optimize Model for this Stock.
4. Click Run Test.
5. A loading indicator will appear during testing.
6. Results will display evaluation metrics and a plot of predicted vs. actual prices.

## Viewing Generated Images
Navigate to Image Gallery to view images such as:

- Article Counts: Number of news articles per stock.
- Price, News, and Sentiment: Visualizations for each stock.

## Acknowledgements
* Flask: For the web framework.
* TensorFlow: For building the neural network.
* yfinance: For fetching stock data.
* Matplotlib & Seaborn: For data visualization.
* Bootstrap: For styling the web interface.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your updates.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

