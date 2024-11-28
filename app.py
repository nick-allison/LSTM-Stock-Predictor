from flask import Flask, render_template, request, redirect, url_for, flash
import os
import glob
import shutil

from model_functions import train_model, test_model

DATA_DIR = 'data'
MODEL_DIR = 'models'
STATIC_DIR = 'static'

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__)

app.secret_key = os.urandom(24)

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Check if 'use_existing_data' is selected
        use_existing_data = 'use_existing_data' in request.form

        # Get the model filename from the form
        model_filename = request.form.get('model_filename')
        if not model_filename:
            flash('Please provide a model filename.')
            return redirect(request.url)

        # Append .keras extension
        model_filename = model_filename + '.keras'
        model_path = os.path.join(MODEL_DIR, model_filename)

        if use_existing_data:
            # Use data files already present in DATA_DIR
            data_files = glob.glob(os.path.join(DATA_DIR, '*.json'))
            if not data_files:
                flash('No data files found in the data directory.')
                return redirect(request.url)
            flash('Training started using existing data. Please wait...')
        else:
            # Handle file uploads
            uploaded_files = request.files.getlist('json_files')
            if uploaded_files and uploaded_files[0].filename != '':
                # Save uploaded files to the DATA_DIR
                for file in uploaded_files:
                    filename = file.filename
                    file.save(os.path.join(DATA_DIR, filename))
                flash('Training started with uploaded data files. Please wait...')
            else:
                flash('No data files uploaded.')
                return redirect(request.url)

        # Run training
        try:
            model = train_model(DATA_DIR, model_path)
            if model is not None:
                flash(f'Model trained successfully and saved as {model_filename}.')
                return redirect(url_for('train_result'))
            else:
                flash('Model training failed.')
                return redirect(request.url)
        except Exception as e:
            flash(f'Error during training: {e}')
            return redirect(request.url)
    return render_template('train.html', title='Train Model')

@app.route('/train_result')
def train_result():
    return render_template('train_result.html', title='Training Result')

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        model_filename = request.form.get('model_filename')
        sequence_length = int(request.form.get('sequence_length', 60))
        if not model_filename:
            flash('Please select a model.')
            return redirect(request.url)

        model_path = os.path.join(MODEL_DIR, model_filename)

        # Create a temporary directory for testing data
        temp_test_dir = os.path.join(DATA_DIR, 'temp_test')
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir)
        os.makedirs(temp_test_dir, exist_ok=True)

        # Handle file uploads
        uploaded_files = request.files.getlist('json_files')
        if uploaded_files and uploaded_files[0].filename != '':
            # Save uploaded files to the temporary test directory
            for file in uploaded_files:
                filename = file.filename
                file.save(os.path.join(temp_test_dir, filename))
            flash('Testing started with uploaded data files. Please wait...')
        else:
            # Use default test data
            flash('No data files uploaded. Using default test data.')
            default_test_files = glob.glob(os.path.join(DATA_DIR, '*.json'))
            if not default_test_files:
                flash('No default test data available.')
                return redirect(request.url)
            for file in default_test_files:
                shutil.copy(file, temp_test_dir)

        # Run testing
        try:
            image_filename_normalized = f'result_normalized_{ticker}.png'
            image_filename_actual = f'result_actual_{ticker}.png'
            image_path_normalized = os.path.join(STATIC_DIR, image_filename_normalized)
            image_path_actual = os.path.join(STATIC_DIR, image_filename_actual)
            results = test_model(ticker, temp_test_dir, model_path, image_path_normalized=image_path_normalized, image_path_actual=image_path_actual, sequence_length=sequence_length)
            if results:
                flash('Testing completed.')
                return render_template('result.html', title='Test Result', message=f'Test completed for {ticker}.', image_path_normalized=image_filename_normalized, image_path_actual=image_filename_actual, results=results)
            else:
                flash('Model testing failed.')
                return redirect(request.url)
        except Exception as e:
            flash(f'Error during testing: {e}')
            return redirect(request.url)
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_test_dir)
    else:
        # Get list of available models
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
        if not models:
            flash('No models available. Please train a model first.')
            return redirect(url_for('train'))
        return render_template('test.html', title='Test Model', models=models)
        
@app.route('/result')
def result():
    return render_template('result.html', title='Result')

if __name__ == '__main__':
    app.run(debug=True)