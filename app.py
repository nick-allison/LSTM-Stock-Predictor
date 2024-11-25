from flask import Flask, render_template, request, redirect, url_for, flash
import os
import shutil
from model_functions import train_model, test_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Configure paths
MODEL_DIR = os.path.join(app.root_path, 'models')
DATA_DIR = os.path.join(app.root_path, 'data')
STATIC_DIR = os.path.join(app.root_path, 'static')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Save uploaded files
        uploaded_files = request.files.getlist('json_files')
        if not uploaded_files:
            flash('No files uploaded.')
            return redirect(request.url)
        # Clear existing data
        shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)
        for file in uploaded_files:
            file.save(os.path.join(DATA_DIR, file.filename))
        # Run training
        try:
            flash('Training started. Please wait...')
            model_path = os.path.join(MODEL_DIR, 'model.keras')
            model = train_model(DATA_DIR, model_path)
            if model is not None:
                flash('Model trained successfully.')
                return redirect(url_for('train_result'))
            else:
                flash('Model training failed.')
                return redirect(request.url)
        except Exception as e:
            flash(f'Error during training: {e}')
            return redirect(request.url)
    return render_template('train.html', title='Train Model')

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        ticker = request.form['ticker']
        optimize = 'optimize' in request.form
        flash('Testing started. Please wait...')
        # Run testing
        try:
            model_path = os.path.join(MODEL_DIR, 'model.keras')
            image_filename = f'result_{ticker}.png'
            image_path = os.path.join(STATIC_DIR, image_filename)
            results = test_model(ticker, DATA_DIR, model_path, optimize=optimize, image_path=image_path)
            if results:
                return render_template('result.html', title='Test Result', message=f'Test completed for {ticker}.', image_path=image_filename, results=results)
            else:
                flash('Model testing failed.')
                return redirect(request.url)
        except Exception as e:
            flash(f'Error during testing: {e}')
            return redirect(request.url)
    return render_template('test.html', title='Test Model')

@app.route('/gallery')
def gallery():
    # Get list of 'price_news_sentiment' images
    price_news_images = []
    for filename in os.listdir(STATIC_DIR):
        if filename.endswith('_price_news_sentiment.png'):
            price_news_images.append(filename)
    
    # Get the 'article_counts.png' image
    article_counts_image = 'article_counts.png' if os.path.exists(os.path.join(STATIC_DIR, 'article_counts.png')) else None
    
    return render_template('gallery.html', title='Image Gallery', 
                           price_news_images=price_news_images, 
                           article_counts_image=article_counts_image)

@app.route('/train_result')
def train_result():
    image_filename = 'model_loss.png'
    return render_template('train_result.html', title='Training Result', image_path=image_filename)

if __name__ == '__main__':
    app.run(debug=True)
