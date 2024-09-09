from flask import Flask, render_template, request, url_for, redirect, flash
import os
from model_load import load_model
from preprocessing import Config
from model_load import get_predictions
from werkzeug.utils import secure_filename
from config_module import Config
from PIL import Image
import pickle
import cv2


app = Flask(__name__)
app.secret_key = "supersecretkey"

# Define the directory where uploaded files will be saved
UPLOAD_FOLDER = 'static/uploaded_mri'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
OUTPUT_FOLDER = 'static/output_mri'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Paths for config and model weights
config_output_filename = os.path.join('./model/', 'model_frcnn_config_test.pickle')

# Load the model using the utility function
model_rpn, model_classifier_only, C = load_model(config_output_filename)

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    if imagefile and allowed_file(imagefile.filename):
        filename = secure_filename(imagefile.filename)
        
        # Save the original image in the 'uploaded_mri' folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)

        try:
            # Verify that the file is an image
            img = Image.open(image_path)
            img.verify()

            predictions = get_predictions(image_path, model_rpn, model_classifier_only, C, C.class_mapping)
            
            # Load the image using OpenCV to draw bounding boxes
            img = cv2.imread(image_path)
            for pred in predictions:
                class_name = pred["class"]
                prob = pred["prob"]
                (x1, y1, x2, y2) = pred["bbox"]

                # Draw rectangle for the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save the image with bounding boxes in 'output_mri' folder
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], "output_" + filename)
            cv2.imwrite(output_image_path, img)

            # Debugging output
            print(f"Predictions: {predictions}")
            return redirect(url_for('index', 
                                    filename=filename, 
                                    prediction=class_name, 
                                    confidence=prob, 
                                    output_image_path='output_mri/output_' + filename) + '#test-section')

        except (IOError, Image.UnidentifiedImageError):
            flash("Invalid image file. Please upload a valid image in PNG or JPG format.")
            return redirect(url_for('index'))
    else:
        flash("Allowed image types are - png, jpg, jpeg")
        return redirect(url_for('index'))


@app.route('/', methods=['GET'])
def index():
    filename = request.args.get('filename')
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    output_image_path =request.args.get('output_image_path')
    return render_template('index.html', 
                           filename=filename, 
                           prediction=prediction, 
                           confidence=confidence,
                           output_image_path=output_image_path)


if __name__== "__main__":
    app.run(debug=True)