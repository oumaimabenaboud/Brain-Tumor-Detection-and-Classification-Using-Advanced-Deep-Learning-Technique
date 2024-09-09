# Smart Healthcare System for Brain Tumor Detection and Classification Using Advanced Deep Learning Techniques

This repository contains the resources and code for the **Smart Healthcare System for Brain Tumor Detection and Classification Using Advanced Deep Learning Techniques** project. The project leverages an enhanced Faster R-CNN framework with a hybrid VGG-16 and ResNet architecture for detecting and classifying brain tumors from MRI images. This system is designed to assist healthcare professionals in making accurate and efficient diagnoses by automating the detection process.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Dataset Description](#dataset-description)


## Project Overview

Brain tumor detection and classification is a critical task in medical imaging that requires precise analysis of MRI scans. This project aims to develop an AI-powered system that automatically detects and classifies three types of brain tumors—glioma, meningioma, and pituitary tumors—using advanced deep learning techniques.

The model is deployed as a web application called **NeuroScan AI**, where users can upload MRI images and receive real-time results on tumor detection and classification.

## Features

- **Automated Detection and Classification**: Detects and classifies brain tumors into glioma, meningioma, and pituitary tumors.
- **Web-based Interface**: Easy-to-use interface for uploading MRI images and receiving diagnostic results.
- **Real-time Results**: Provides real-time predictions of tumor type, along with confidence scores.
- **Visualization**: Displays bounding boxes around detected tumors on the uploaded MRI image.

## Repository Structure
```bash
📦Brain-Tumor-Detection-and-Classification-Using-Advanced-Deep-Learning-Technique
 ┣ 📂dataset                               # Directory containing the dataset and annotations
 ┃ ┣ 📂annotations                         # Annotations related to tumor regions (in various formats)
 ┃ ┃ ┣ 📜annotation.txt                    # Annotations for train data in text format
 ┃ ┃ ┣ 📜original_annotations.csv          # Original CSV file containing annotations with tumor borders
 ┃ ┃ ┣ 📜output_annotations.csv            # Preprocessed CSV file containing annotations with bounding boxs
 ┃ ┃ ┣ 📜test_annotation.txt               # Annotations for test data in text format
 ┃ ┃ ┣ 📜test_annotations.csv              # CSV annotations for test dataset
 ┃ ┃ ┗ 📜train_annotations.csv             # CSV annotations for training dataset
 ┃ ┣ 📂images                              # Directory containing converted MRI images (from .mat to .jpg)
 ┃ ┣ 📂mat_files                           # Original brain tumor MRI images in MATLAB .mat format
 ┃ ┣ 📂test                                # Test dataset images
 ┃ ┣ 📂train                               # Training dataset images
 ┃ ┗ 📜data_preparation.ipynb              # Jupyter notebook for preprocessing and dataset preparation
 ┣ 📂fasterr-cnn                           # Directory containing Faster R-CNN model code and scripts
 ┣ 📂NeuroScan AI                          # Directory for web-based application files
 ┃ ┣ 📂model                               # Pre-trained models for Faster R-CNN hybrid architecture
 ┃ ┃ ┣ 📜model_frcnn_config_test.pickle    # Model configuration file for Faster R-CNN
 ┃ ┃ ┗ 📜model_frcnn_hybrid_new_test.hdf5  # Pre-trained model weights (Faster R-CNN with hybrid VGG-16/ResNet)
 ┃ ┣ 📂static                              # Static files for web app (CSS, JS, fonts, images)
 ┃ ┃ ┣ 📂css                          
 ┃ ┃ ┣ 📂fonts                        
 ┃ ┃ ┣ 📂img                          
 ┃ ┃ ┣ 📂js                           
 ┃ ┃ ┣ 📂output_mri                   # Directory where processed MRI results are saved
 ┃ ┃ ┗ 📂uploaded_mri                 # Directory for uploaded MRI scans from users
 ┃ ┣ 📂templates                      # HTML templates for Flask web app
 ┃ ┃ ┗ 📜index.html                   
 ┃ ┣ 📜app.py                         # Flask application for running the NeuroScan AI web app
 ┃ ┣ 📜config_module.py               # Configuration module for model loading and app settings
 ┃ ┣ 📜model_load.py                  # Script to load pre-trained Faster R-CNN model
 ┃ ┗ 📜preprocessing.py               # Script for preprocessing MRI images before prediction
 ┣ 📂Nouveau dossier                  
 ┗ 📜README.md                        # Readme file explaining the project details and usage

```

## Dataset Description

The dataset used in this project is from FigShare, available at [FigShare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427). This dataset contains MRI images of brain tumors, including glioma, meningioma, and pituitary tumors. The images and the annotations corresponding to the tumor regions were originally stored in MATLAB `.mat` file format. To simplify image processing for this project, we converted the `.mat` files to `.jpg` format using a custom MATLAB script.
### Conversion Process

The following MATLAB script was used to convert the `.mat` images to `.jpg` format. The MRI images are normalized, converted to `uint8`, and saved as `.jpg` files. This process ensures that the images are ready for use in common deep learning frameworks like TensorFlow or PyTorch.

### MATLAB to JPEG Conversion Script

```matlab
% Define the folder where the .mat files are located and the folder to save .jpg images
matFilesFolder = 'mat_files'; % Folder containing .mat files
outputImagesFolder = 'pics';  % Folder to save the converted .jpg images

% Check for the existence of the output image folder
if ~exist(outputImagesFolder, 'dir')
    mkdir(outputImagesFolder); % Create the folder if it doesn't exist
end

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(matFilesFolder, '*.mat'));

for k = 1:length(matFiles)
    % Load each .mat file
    matFilePath = fullfile(matFilesFolder, matFiles(k).name);
    load(matFilePath); % Load the .mat file contents
    
    % Extract the image data from the loaded .mat file
    image_data = cjdata.image;
    
    % Convert the image to uint8 format if necessary
    if isa(image_data, 'int16')
        image_data_double = double(image_data);
        % Normalize the data to the range 0 to 255
        image_data_uint8 = uint8(255 * (image_data_double - min(image_data_double(:))) / ...
            (max(image_data_double(:)) - min(image_data_double(:))));
    else
        image_data_uint8 = image_data; % If already in a compatible format
    end
    
    % Save the image as a JPEG file with a numbered name
    imageFileName = sprintf('%d.jpg', k); % Name the image file using numbering
    imwrite(image_data_uint8, fullfile(outputImagesFolder, imageFileName)); % Save as .jpg
end

disp('Images have been successfully saved to the folder.');
```
### Annotations Extraction
In addition to the image conversion, the annotations related to the tumor regions were extracted from the `.mat` files and saved as `.txt` files. These annotations include the tumor type, bounding boxes. For detailed annotations extraction steps, you can refer to the `data_preparation.ipynb` notebook available in this repository.

## Model Training
To enhance model robustness, advanced data augmentation techniques were applied, including horizontal and vertical flipping, as well as 90-degree rotations. The training spanned over 380 epochs, with 4,000 iterations per epoch.

- **Framework**: TensorFlow
- **Model**: Faster R-CNN with a hybrid VGG-16 and ResNet backbone
- **Optimizer**: Adam for both the RPN and the classifier, and the Stochastic Gradient Descent (SGD) optimizer
is used to compile the combined model of RPN and classifier.
- **Batch Size**: 4 Regions of Interest (RoIs) per batch
- **Training Time**: Approximately 48 hours

For detailed training steps, refer to `training_script.ipnyb`.

---

## Results

The model demonstrated strong performance in detecting and classifying brain tumors from MRI images, achieving an overall **mean Average Precision (mAP)** of 91.97% across the following tumor types:

- **Glioma**: 85.2% AP
- **Meningioma**: 93.4% AP
- **Pituitary Tumor**: 97.3% AP

The results are visualized with bounding boxes and confidence scores on the uploaded MRI images. Detailed results and visualizations are available in the `.ipynb` notebook.

---

## Deployment

The model is deployed as a web application using Flask. Users can upload MRI images through the **NeuroScan AI** web interface, and the system returns real-time predictions of tumor type, confidence scores, and bounding box visualizations.

# NeuroScan AI

---

### System Architecture

<p align="center">
  <img src="" alt="System Architecture Diagram">
</p>
<br>
<i><p align="center">System Architecture Diagram</p></i>

The system architecture of **NeuroScan AI** consists of the following components:

1. **Frontend Interface**: Provides a drag-and-drop user interface for users to upload MRI images for brain tumor detection.
2. **Flask Backend**: Handles the uploaded MRI images, processes them, and invokes the AI model to generate predictions.
3. **AI Model**: A pre-trained model (Faster R-CNN with a hybrid VGG-16 and ResNet architecture) deployed in the backend to detect and classify brain tumors.

---

### NeuroScan AI Web Interface

<p align="center">
  <img src="" alt="NeuroScan AI Web Interface">
</p>
<br>
![web_archi](https://github.com/user-attachments/assets/cc728340-bee9-4918-9b07-e285c89e92ac)


The **NeuroScan AI** interface offers a simple and intuitive platform for users:

- **(a) Welcome Section**: Provides an introduction and explanation of the platform's purpose.
- **(b) Features Section**: Outlines the system's capabilities and unique benefits.
- **(c) Image Upload & Result Display**: Users can upload their MRI images, and the system will return the tumor detection and classification results, including confidence scores.
- **(d) Footer**: Contains a contact form for inquiries and additional details about the platform.

---

### Backend Integration

The backend is powered by Flask, which processes the uploaded MRI images and connects with the AI model for analysis. After processing, Flask returns the prediction results to the frontend in real time.

### To deploy the backend:

1. **Install Flask and other dependencies**:
   ```bash
   pip install Flask tensorflow

