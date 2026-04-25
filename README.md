🧠 Driver Drowsiness Detection System

An AI-powered real-time system that detects driver drowsiness using computer vision and deep learning. It monitors eye states through a webcam and triggers alerts when signs of fatigue are detected, helping reduce road accidents caused by driver sleepiness.

🚀 Project Overview

Driver fatigue is one of the leading causes of road accidents worldwide.
This system uses a Convolutional Neural Network (CNN) to classify eye states as open or closed, and performs real-time detection using webcam input.

The project covers the complete ML pipeline:

Data preprocessing
Model training
Evaluation
Real-time deployment
⚙️ System Architecture
Data Collection & Preprocessing
        ↓
CNN Model Training
        ↓
Model Evaluation
        ↓
Saved Model (.h5)
        ↓
Real-Time Webcam Detection
📊 Data Pipeline

👨‍💻 Contributor: Himanshu Bisht

Dataset cleaning and preprocessing (clean_dataset.py)
Image resizing, normalization, and augmentation
Data augmentation using ImageDataGenerator
Class distribution analysis
Sample image visualization
🧠 Model Training & Evaluation

👨‍💻 Contributor: Ayush Dobhal

CNN architecture design (model.py)
Training on eye-state dataset (open/closed)
Model performance evaluation:
Accuracy & loss curves
Confusion matrix
Classification report
Comparison with MobileNetV2
🎥 Real-Time Detection System

👨‍💻 Contributor: Neeraj Bisht

Loading trained CNN model for inference
Real-time webcam processing using OpenCV (realtime.py)
Face and eye detection in live video stream
On-screen status display:
🟢 Alert
🔴 Drowsy
Audio alert trigger when drowsiness is detected
🧪 Technologies Used
Python 🐍
OpenCV 👁️
TensorFlow / Keras 🤖
NumPy
Matplotlib 📊
scikit-learn
imutils
⭐ Key Features
🎥 Real-time drowsiness detection using webcam
🧠 CNN-based eye state classification
📦 End-to-end ML pipeline (data → model → deployment)
📊 Model evaluation with visual performance metrics
🔊 Audio alert system for safety warnings
🖥️ Live bounding box + status display
👥 Contributors
Himanshu Bisht → Data preprocessing pipeline
Ayush Dobhal → Model design & training
Neeraj Bisht → Real-time detection system
📌 Workflow Summary
Data preprocessing & augmentation
CNN model training
Model evaluation
Saving trained model
Real-time inference using webcam
