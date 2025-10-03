from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import base64
from werkzeug.utils import secure_filename
from PIL import Image
import io
from datetime import datetime
from model_inference import get_model

app = Flask(__name__)
# IMPORTANT: Change this to a long, random string for security
app.secret_key = 'super-secret-key-change-me'

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# --- Pre-warm the model on startup ---
model = None
try:
    print("="*50)
    print("Attempting to load model on startup...")
    model = get_model()
    print("✅ Model pre-loading successful!")
    print("="*50)
except Exception as e:
    print("="*50)
    print(f"❌ CRITICAL: Failed to load model on startup: {e}")
    print("="*50)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_display(image_path):
    """Preprocess the uploaded image for creating a base64 string for the HTML"""
    try:
        img = Image.open(image_path).convert('RGB')
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"Error preprocessing image for display: {e}")
        return None

@app.route('/')
def index():
    """Main page with the image upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, prediction, and render the results page"""
    if model is None:
        flash('Model is not available due to a startup error. Please check server logs.')
        return redirect(url_for('index'))
        
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        flash('Invalid file. Please upload a PNG, JPG, or JPEG image.')
        return redirect(request.url)
    
    # Create a secure, unique filename
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Prepare image for display on the results page
    img_str = preprocess_image_for_display(filepath)
    if not img_str:
        flash('Error processing uploaded image.')
        return redirect(url_for('index'))
    
    # Run prediction
    try:
        prediction_result = model.predict(filepath)
        
        # This line fixes the "Image Processed: No" issue
        prediction_result['image_processed'] = True 
        
        prediction_result['processed_image'] = img_str
        prediction_result['original_filename'] = file.filename
        return render_template('results.html', result=prediction_result)
    except Exception as e:
        print(f"Prediction error: {e}")
        flash(f'An error occurred during prediction. Please try again.')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Open your browser and navigate to: http://12.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)