**Driver Drowsiness Detection System**

This project is an AI-based system that detects driver drowsiness in real time using computer vision and deep learning. It monitors eye behavior through a webcam and gives an alert when signs of drowsiness are detected.

**Project Overview**

Driver fatigue is a major cause of road accidents.
This system uses a trained CNN model to classify eye states (open/closed) and runs a real-time detection system using webcam input.

The workflow includes data preprocessing, model training, evaluation, and real-time deployment.

**System Breakdown**
**Data Pipeline**

**Contributor: Himanshu Bish**t

Dataset cleaning and preprocessing (clean_dataset.py)
Image resizing, normalization, and augmentation using ImageDataGenerator
Preparing data for training
Checking class distribution and sample images
Model Training and Evaluation
**
Contributor: Ayush Dobhal
**
CNN model design (model.py)
Model training on eye state dataset
Performance evaluation using accuracy, loss graphs
Confusion matrix and classification report
Comparison with MobileNetV2
Real-Time Detection

**Contributor: Neeraj Bisht**

Loading trained model for inference
Real-time webcam processing using OpenCV (realtime.py)
Face and eye-based prediction in live video
Displaying drowsy/alert status on screen
Alert sound trigger when drowsiness is detected

**Technologies Used**
Python
OpenCV
TensorFlow / Keras
NumPy
Matplotlib
scikit-learn
imutils

**Workflow**

Data Preprocessing
↓
Model Training
↓
Model Evaluation
↓
Saved Model
↓
Real-Time Detection using Webcam
 
**Key Features** 
Real-time detection using webcam
CNN-based eye state classification
Live bounding box and status display
Alert system for drowsiness detection
Complete ML pipeline from data to deployment

**Contributions**

The project was developed in a team, with each member working on a separate module:

Model design and training
Data preprocessing pipeline
Real-time detection system
