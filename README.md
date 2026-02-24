# Driver Drowsiness Detection System

This project is an AI-based Driver Drowsiness Detection System
designed to monitor a driver in real time and detect signs of fatigue using computer vision
and deep learning. The system analyzes eye behavior through a webcam feed and triggers 
an alert when drowsiness is detected.

---

## Project Overview

Driver fatigue is a major cause of road accidents. 
This project addresses the problem by using
a deep learning model combined with real-time computer vision.
A Shallow Convolutional Neural Network (CNN) is trained to classify eye states, and the trained model is then
deployed in a real-time application using a webcam.

---

## System Architecture

The project is divided into three interconnected modules:

### 🟡 Data Pipeline
- Dataset cleaning and preprocessing
- Image resizing, normalization, and augmentation
- Preparing data generators for model training
- Verifying class balance and sample visualization

### 🔴 Model Architecture & Evaluation
- Designing and implementing a Shallow CNN architecture
- Training the model on preprocessed eye images
- Plotting accuracy and loss graphs
- Evaluating performance using confusion matrix and classification report
- Comparing performance with a pre-trained MobileNetV2 model

### 🟢 Real-Time Detection Application
- Loading the trained model
- Capturing live video feed using OpenCV
- Detecting face and eye regions
- Predicting drowsy/alert state in real time
- Displaying bounding boxes and status labels
- Triggering alert sound when drowsiness is detected

---

## Technology Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- imutils
- Matplotlib
- scikit-learn

---

## Workflow Connection

```text
Data Preprocessing
       ↓
Model Training & Evaluation
       ↓
Saved Trained Model (.h5)
       ↓
Real-Time Webcam Detection
