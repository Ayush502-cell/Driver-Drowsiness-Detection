# ============================================================
#  FILE: model.py
#  YOU (Main Developer) — CNN Architecture + Training
#  Purpose: Build, train and evaluate the DrowsyDetectNet
#           Shallow CNN (based on IEEE Access 2024 paper)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Import data pipeline from teammate 1
from data_pipeline import create_generators

# ─── CONFIG (from paper) ───────────────────────────────────
IMG_SIZE = (128, 128, 3)
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
MODEL_PATH = "drowsy_model.h5"
SEED = 42
# ────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─── STEP 1: BUILD MODEL ───────────────────────────────────
def build_shallow_cnn():
    """
    DrowsyDetectNet Shallow CNN Architecture
    Exactly as described in:
    'DrowsyDetectNet: Driver Drowsiness Detection Using
     Lightweight CNN With Limited Training Data'
    IEEE Access, 2024

    Architecture:
    Input(128x128x3)
    → Conv(32, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.2)
    → Conv(64, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.2)
    → Conv(128,3x3) + ReLU + MaxPool(2x2)
    → Conv(128,1x1) + ReLU + MaxPool(2x2)
    → Flatten → FC(128) → Sigmoid Output
    """
    model = Sequential(
        [
            # ── Block 1: 32 filters, 3×3 ──
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=IMG_SIZE),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            # ── Block 2: 64 filters, 3×3 ──
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            # ── Block 3: 128 filters, 3×3 ──
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            # ── Block 4: 128 filters, 1×1 ──
            Conv2D(128, (1, 1), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            # ── Classifier Head ──
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),  # binary output: Drowsy or Alert
        ],
        name="DrowsyDetectNet",
    )

    return model


# ─── STEP 2: COMPILE MODEL ─────────────────────────────────
def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── STEP 3: TRAIN MODEL ───────────────────────────────────
def train_model(model, train_gen, val_gen):
    callbacks = [
        # Stop early if val_loss stops improving for 10 epochs
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        # Save the best model automatically
        ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print("\n🚀 Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n✔ Training complete. Best model saved to: {MODEL_PATH}")
    return history


# ─── STEP 4: PLOT RESULTS ──────────────────────────────────
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(
        history.history["accuracy"],
        label="Train Accuracy",
        color="#2ecc71",
        linewidth=2,
    )
    axes[0].plot(
        history.history["val_accuracy"],
        label="Val Accuracy",
        color="#e74c3c",
        linewidth=2,
        linestyle="--",
    )
    axes[0].set_title("Training & Validation Accuracy", fontsize=13)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(
        history.history["loss"], label="Train Loss", color="#2ecc71", linewidth=2
    )
    axes[1].plot(
        history.history["val_loss"],
        label="Val Loss",
        color="#e74c3c",
        linewidth=2,
        linestyle="--",
    )
    axes[1].set_title("Training & Validation Loss", fontsize=13)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DrowsyDetectNet Training Results", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("✔ Training graphs saved as training_history.png")


# ─── STEP 5: EVALUATE MODEL ────────────────────────────────
def evaluate_model(model, test_gen):
    print("\n--- MODEL EVALUATION ON TEST SET ---")

    # Overall accuracy
    loss, accuracy = model.evaluate(test_gen, verbose=0)
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    # Predictions
    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())

    # Classification report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix — DrowsyDetectNet", fontsize=13)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("✔ Confusion matrix saved as confusion_matrix.png")

    return accuracy


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   DrowsyDetectNet — SHALLOW CNN TRAINING")
    print("   Based on: IEEE Access 2024 Paper")
    print("=" * 55)

    # Load data (from Teammate 1's pipeline)
    print("\n[1/5] Loading data...")
    train_gen, val_gen, test_gen = create_generators()

    # Build model
    print("\n[2/5] Building model...")
    model = build_shallow_cnn()
    model = compile_model(model)
    model.summary()

    # Train
    print("\n[3/5] Training...")
    history = train_model(model, train_gen, val_gen)

    # Plot
    print("\n[4/5] Plotting results...")
    plot_training_history(history)

    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    final_accuracy = evaluate_model(model, test_gen)

    print("\n" + "=" * 55)
    print(f"   ✅ FINAL TEST ACCURACY: {final_accuracy * 100:.2f}%")
    print(f"   Model saved to: {MODEL_PATH}")
    print("=" * 55)
