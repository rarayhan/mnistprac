import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)  # Adjusted input size
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))

        return F.log_softmax(x, dim=1)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", weights_only=False))
model.eval()



# Preprocessing function
def preprocess_image(image_path, save_path="processed_image.png"):
    """
    Preprocess the input image to match MNIST format and save the processed image.
    """
    try:
        # Open and convert to grayscale
        image = Image.open(image_path).convert("L")  # Convert to grayscale

        # Invert colors if the background is dark
        image = ImageOps.invert(image)

        # Enhance contrast
        image = ImageEnhance.Contrast(image).enhance(2.0)

        # Crop the digit and center it
        np_image = np.array(image)
        non_empty_columns = np.where(np_image.min(axis=0) < 255)[0]
        non_empty_rows = np.where(np_image.min(axis=1) < 255)[0]
        if non_empty_columns.any() and non_empty_rows.any():
            crop_box = (min(non_empty_columns), min(non_empty_rows),
                        max(non_empty_columns), max(non_empty_rows))
            image = image.crop(crop_box)

        # Resize to 20x20 and center in a 28x28 canvas
        image = image.resize((20, 20), Image.Resampling.LANCZOS)

        new_image = Image.new("L", (28, 28), (0))  # Create a black 28x28 canvas
        new_image.paste(image, (4, 4))  # Center the digit

        # Save the processed image
        new_image.save(save_path)

        return new_image
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Prediction function
def predict_digit(image_path):
    """
    Predict the digit from the given handwritten image.
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None

        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(processed_image).unsqueeze(0).to(device)
        image_tensor = transforms.Normalize((0.5,), (0.5,))(image_tensor)  # Normalize

        # Forward pass through the model
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
        probabilities = torch.exp(output)  # Convert log probabilities to probabilities
        print("Class probabilities:", probabilities)

        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Input the image path
image_path = "3.png"
processed_image_path = "processed_image.png"
predicted_digit = predict_digit(image_path)


if predicted_digit is not None:
    print(f"The predicted digit is: {predicted_digit}")
    print(f"Processed image saved as: {processed_image_path}")
else:
    print("Failed to predict the digit.")
