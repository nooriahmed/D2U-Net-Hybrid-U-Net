import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, PReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import PReLU

# Paths for dataset
train_images_path = "Path/Polyp-dataset/trainx/" 
train_masks_path = "Path/Polyp-dataset/trainy/" 
test_images_path = "Path/Polyp-dataset/testx/"   
test_masks_path = "Path/Polyp-dataset/testy/"   
val_images_path = "Path/Polyp-dataset/validationx/" 
val_masks_path = "Path/Polyp-dataset/validationy/"  

# Parameters
IMAGE_SIZE = (60, 60)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
DROPOUT_RATE = 0.5

# Function to preprocess images and masks
def preprocess_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    # Decode and resize
    image = tf.image.decode_jpeg(image, channels=3)
    mask = tf.image.decode_jpeg(mask, channels=1)

    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    mask = tf.image.resize(mask, IMAGE_SIZE) / 255.0

    mask = tf.cast(mask > 0.5, tf.float32)  # Binary mask
    return image, mask

# Function to create tf.data dataset
def create_dataset(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    return dataset

# Create datasets
train_dataset = create_dataset(train_images_path, train_masks_path)
test_dataset = create_dataset(test_images_path, test_masks_path)
val_dataset = create_dataset(val_images_path, val_masks_path)

# Function to align dataset sizes
def align_dataset_sizes(train_ds, val_ds, test_ds):
    train_count = tf.data.experimental.cardinality(train_ds).numpy()
    val_ds = val_ds.take(train_count)
    test_ds = test_ds.take(train_count)
    return train_ds, val_ds, test_ds

train_dataset, val_dataset, test_dataset = align_dataset_sizes(train_dataset, val_dataset, test_dataset)

def lightweight_dual_path_unet(input_shape=(60, 60, 3), dropout_rate=0.5):
    inputs = Input(input_shape)

    # Path 1: Global Features (Contextual Information)
    # Encoder 1
    conv1_1 = SeparableConv2D(32, (3, 3), padding='same')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = PReLU()(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = SeparableConv2D(64, (3, 3), padding='same')(pool1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = PReLU()(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    # Path 2: Local Features (Edges)
    # Encoder 2
    conv1_2 = SeparableConv2D(32, (3, 3), padding='same')(inputs)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = PReLU()(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_2 = SeparableConv2D(64, (3, 3), padding='same')(pool1_2)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = PReLU()(conv2_2)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    # Bottleneck: Merging Features from Both Paths with Residual Connection
    merged = Concatenate()([pool2_1, pool2_2])
    shortcut = merged  # Residual path

    conv3 = SeparableConv2D(128, (3, 3), padding='same')(merged)
    conv3 = BatchNormalization()(conv3)
    conv3 = PReLU()(conv3)
    conv3 = Dropout(dropout_rate)(conv3)
    conv3 = SeparableConv2D(128, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Add()([conv3, shortcut])  # Residual connection
    conv3 = PReLU()(conv3)

    # Decoder 1 (Global Features)
    up4_1 = UpSampling2D(size=(2, 2))(conv3)
    up4_1 = Concatenate()([up4_1, conv2_1])
    conv4_1 = SeparableConv2D(64, (3, 3), padding='same')(up4_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = PReLU()(conv4_1)

    up5_1 = UpSampling2D(size=(2, 2))(conv4_1)
    up5_1 = Concatenate()([up5_1, conv1_1])
    conv5_1 = SeparableConv2D(32, (3, 3), padding='same')(up5_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = PReLU()(conv5_1)

    # Decoder 2 (Local Features)
    up4_2 = UpSampling2D(size=(2, 2))(conv3)
    up4_2 = Concatenate()([up4_2, conv2_2])
    conv4_2 = SeparableConv2D(64, (3, 3), padding='same')(up4_2)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = PReLU()(conv4_2)

    up5_2 = UpSampling2D(size=(2, 2))(conv4_2)
    up5_2 = Concatenate()([up5_2, conv1_2])
    conv5_2 = SeparableConv2D(32, (3, 3), padding='same')(up5_2)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = PReLU()(conv5_2)

    # Merge Outputs from Both Decoders
    merged_output = Concatenate()([conv5_1, conv5_2])
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(merged_output)

    model = Model(inputs, outputs)
    return model


# Compile model
model = lightweight_dual_path_unet()
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3]
)

# Fit the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, verbose=1)

# Save weights
model.save_weights('dual_path_unet_weights-polyp.weights.h5') # Changed the filename to include '.weights.h5'
print("Model weights saved successfully.")

# Plot training and validation metrics
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_history(history)
