
import numpy as np
import matplotlib.pyplot as plt

#addede-here
# Initialize the model
model = dual_path_unet()

# Load weights into the model
model.load_weights('dual_path_unet_weights-brain.weights.h5')
print("Model weights loaded successfully.")

# Prediction and visualization
def preprocess_image(image):
    return np.expand_dims(image, axis=0)  # Add batch dimension

def predict_and_visualize(model, test_dataset, index=201):
    # Fetch a batch of test images and masks
    test_images, test_masks = list(test_dataset.take(1))[0]

    # Extract specific image and mask
    image = test_images[index].numpy()
    ground_truth = test_masks[index].numpy()

    # Predict
    prediction = model.predict(preprocess_image(image))[0]

    # Visualization
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(ground_truth.squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(prediction.squeeze(), cmap='gray')

    plt.show()

# Example usage
predict_and_visualize(model, test_dataset, index=3)
