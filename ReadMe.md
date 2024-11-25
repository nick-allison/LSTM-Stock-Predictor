# Stock Price Prediction with Sentiment Analysis

## Overview
This project is a **web application** that predicts stock prices using **historical stock data** and **sentiment analysis** from news articles. It utilizes an **LSTM neural network** and provides an intuitive **web interface** for training and testing the model.

---

## Features
- **Train Model:** Upload sentiment data in JSON format and train the LSTM model.
- **Test Model:** Test the model on specific stock tickers, with an option to optimize the model for a particular stock.
- **Image Gallery:** View generated images from the training process, including sentiment analysis and stock price trends.
- **Loading Indicators:** Visual feedback during long-running operations like training and testing.
- **Visualization:** Graphs showing training loss, predicted vs. actual stock prices, and sentiment analysis.

---

## Requirements and Installation

This project requires **Python 3.7 or higher**. Follow the steps below to set up and run the application:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/stock-price-prediction.git
    cd stock-price-prediction
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Directory Structure**:
    ```plaintext
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
    ```

4. **Start the Application**:
    Run the Flask app:
    ```bash
    python app.py
    ```
    Access the application at [http://localhost:5000](http://localhost:5000).

5. **Train the Model**:
    - Navigate to **Train Model** in the web interface.
    - Upload your sentiment JSON data files.
    - Click **Start Training**.
    - A loading indicator will appear during training.
    - Upon completion, you will be redirected to a page displaying the training loss graph.

6. **Test the Model**:
    - Navigate to **Test Model** in the web interface.
    - Enter the stock ticker symbol (e.g., `AAPL`).
    - Optionally, check **Optimize Model for this Stock**.
    - Click **Run Test**.
    - A loading indicator will appear during testing.
    - Results will display evaluation metrics and a plot of predicted vs. actual prices.

7. **View Generated Images**:
    - Navigate to **Image Gallery** to view images such as:
        - **Article Counts**: Number of news articles per stock.
        - **Price, News, and Sentiment**: Visualizations for each stock.

---

## Acknowledgements
- **[Flask](https://flask.palletsprojects.com/):** For the web framework.
- **[TensorFlow](https://www.tensorflow.org/):** For building the neural network.
- **[yfinance](https://github.com/ranaroussi/yfinance):** For fetching stock data.
- **[Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/):** For data visualization.
- **[Bootstrap](https://getbootstrap.com/):** For styling the web interface.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your updates.

---

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

