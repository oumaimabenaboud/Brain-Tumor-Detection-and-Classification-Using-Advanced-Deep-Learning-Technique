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



### Steps to Install

1. Clone the repository:
  
