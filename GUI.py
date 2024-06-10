import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('simpleCNN_2.h5')

# Define your classes
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Function to classify the image
def classify_image(image_path):
    try:
        print("Image path:", image_path)  # Dodatkowy wydruk kontrolny
        image = Image.open(image_path)
        image = image.resize((150, 150))  # Resize image according to your model's input shape
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        result = model.predict(image)
        predicted_class_index = np.argmax(result)
        predicted_class = classes[predicted_class_index]
        messagebox.showinfo("Prediction", f"The image belongs to: {predicted_class}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to open file dialog and get image path
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        print("Selected file:", file_path)  # Dodatkowy wydruk kontrolny
        classify_image(file_path)

# Create main window
root = tk.Tk()
root.title("Image Classifier")

# Create labels and buttons
label = tk.Label(root, text="Click below to select an image to classify:")
label.pack(pady=10)

classify_button = tk.Button(root, text="Classify Image", command=open_image)
classify_button.pack(pady=5)

# Run the application
root.mainloop()
