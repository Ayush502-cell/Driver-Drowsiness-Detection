# ============================================================
#  FILE: data_pipeline.py
#  TEAMMATE 1 — Data Pipeline
#  Purpose: Load, augment and prepare data for model training
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONFIG ────────────────────────────────────────────────
TRAIN_DIR  = 'dataset_binary/train'
TEST_DIR   = 'dataset_binary/test'
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
# ────────────────────────────────────────────────────────────


def create_generators():
    """
    Creates and returns train, validation and test data generators.
    - Training data: normalized + augmented
    - Validation data: normalized only (split from training)
    - Test data: normalized only
    """

    # Training generator WITH augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # normalize pixel values to [0,1]
        rotation_range=10,         # randomly rotate images by up to 10 degrees
        horizontal_flip=True,      # randomly flip images horizontally
        zoom_range=0.1,            # randomly zoom in/out by 10%
        width_shift_range=0.1,     # randomly shift image left/right
        height_shift_range=0.1,    # randomly shift image up/down
        validation_split=0.2       # reserve 20% of training data for validation
    )

    # Test generator - ONLY normalization, NO augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training set (80% of train folder)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',       # binary = Drowsy(0) or Alert(1)
        subset='training',
        shuffle=True,
        seed=42
    )

    # Validation set (20% of train folder)
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )

    # Test set (from test folder)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False              # IMPORTANT: keep order for confusion matrix
    )

    return train_generator, val_generator, test_generator


def show_class_distribution(train_generator, test_generator):
    """
    Prints and plots class distribution to check imbalance.
    """
    print("\n--- CLASS DISTRIBUTION ---")
    print("Classes:", train_generator.class_indices)

    train_labels = train_generator.classes
    test_labels  = test_generator.classes

    train_counts = {cls: int(np.sum(train_labels == idx))
                    for cls, idx in train_generator.class_indices.items()}
    test_counts  = {cls: int(np.sum(test_labels == idx))
                    for cls, idx in test_generator.class_indices.items()}

    print("\nTraining samples:")
    for cls, count in train_counts.items():
        print(f"  {cls}: {count}")

    print("\nTest samples:")
    for cls, count in test_counts.items():
        print(f"  {cls}: {count}")

    # Plot bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(train_counts.keys(), train_counts.values(), color=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Training Set Distribution')
    axes[0].set_ylabel('Number of Images')

    axes[1].bar(test_counts.keys(), test_counts.values(), color=['#e74c3c', '#2ecc71'])
    axes[1].set_title('Test Set Distribution')
    axes[1].set_ylabel('Number of Images')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    plt.show()
    print("\n✔ Class distribution chart saved as class_distribution.png")


def show_sample_images(train_generator):
    """
    Displays a grid of sample images from the training set.
    """
    class_names = {v: k for k, v in train_generator.class_indices.items()}
    images, labels = next(train_generator)

    plt.figure(figsize=(12, 6))
    for i in range(min(12, len(images))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[int(labels[i])], fontsize=9)
        plt.axis('off')

    plt.suptitle('Sample Training Images', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150)
    plt.show()
    print("✔ Sample images saved as sample_images.png")


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   DATA PIPELINE — TEAMMATE 1")
    print("=" * 50)

    train_gen, val_gen, test_gen = create_generators()

    print(f"\n✔ Training batches  : {len(train_gen)}")
    print(f"✔ Validation batches: {len(val_gen)}")
    print(f"✔ Test batches      : {len(test_gen)}")

    show_class_distribution(train_gen, test_gen)
    show_sample_images(train_gen)

    print("\n✅ Data pipeline ready! Generators created successfully.")
    print("   Import create_generators() in model.py to use them.")