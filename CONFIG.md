# Configuration Parameters – Real-Time Drowsiness Detection

This document explains the key configuration values used in
realtime.py for driver drowsiness detection.

------------------------------------------------------------

## 1. IMG_SIZE

IMG_SIZE = (128, 128)

This defines the resolution at which each video frame is resized
before being passed to the deep learning model.

Reason:
The model was trained using images of size 128×128.
Using the same size during real-time inference ensures
consistent and reliable predictions.

------------------------------------------------------------

## 2. THRESHOLD

THRESHOLD = 0.5

This value is used to decide whether the driver is alert or drowsy.

If prediction < 0.5  → Driver is considered DROWSY  
If prediction ≥ 0.5 → Driver is considered ALERT

The model uses a sigmoid activation function, producing
outputs between 0 and 1.

------------------------------------------------------------

## 3. ALERT_FRAMES

ALERT_FRAMES = 20

This parameter defines how many consecutive drowsy frames
are required before showing a drowsiness alert.

Purpose:
This avoids false alarms caused by blinking or momentary
eye closure and ensures that alerts are triggered only
when sustained drowsiness is detected.