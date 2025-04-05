# ASL Alphabet Recognition

A deep learning project for American Sign Language (ASL) alphabet recognition using Convolutional Neural Networks.

## 📝 Project Overview

This project implements a computer vision system that recognizes American Sign Language (ASL) alphabet signs from images. The system uses a Convolutional Neural Network (CNN) trained on the ASL Alphabet dataset to classify hand gestures into 29 classes (26 letters, plus "delete", "nothing", and "space").

![ASL Alphabet Example](https://i.imgur.com/your-image-url.png)

## 🚀 Features

- Recognition of all 26 ASL alphabet letters (A-Z)
- Recognition of special signs (delete, nothing, space)
- High accuracy model (>99.5% on test set)
- Multiple deployment options:
  - Keras model
  - TensorFlow SavedModel
  - TensorFlow Lite for mobile deployment
  - TensorFlow.js for web deployment

## 🔧 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/asl-alphabet-recognition.git
cd asl-alphabet-recognition
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the [ASL Alphabet dataset](https://www.kaggle.com/grassknoted/asl-alphabet) from Kaggle, which contains:

- 87,000 images (200x200 pixels)
- 29 classes (A-Z, delete, nothing, space)
- 3,000 images per class

The dataset is split as follows:

- Training set: 60,900 images (70%)
- Validation set: 17,400 images (20%)
- Test set: 8,700 images (10%)

## 🧠 Model Architecture

The CNN architecture consists of:

```
- Input layer (224x224x3)
- Rescaling layer (1/255)
- Convolutional layers:
  - Conv2D (32 filters, 3x3 kernel, ReLU activation)
  - MaxPooling (2x2)
  - Conv2D (64 filters, 3x3 kernel, ReLU activation)
  - MaxPooling (2x2)
  - Conv2D (128 filters, 3x3 kernel, ReLU activation)
  - MaxPooling (2x2)
- Flatten layer
- Dense layer (512 units, ReLU activation)
- Dropout (0.5)
- Output layer (29 units, softmax activation)
```

## 🏋️ Training Process

The model was trained with the following configuration:

- Optimizer: AdamW
- Loss function: Sparse Categorical Crossentropy
- Batch size: 256
- Early stopping (patience=2)
- Model checkpoint (save best model)

Training results:

- Training accuracy: 98.15%
- Validation accuracy: 99.70%
- Test accuracy: 99.57%

The training process and performance visualizations can be found in the `notebook_train.ipynb` notebook.

## 🔍 Usage

### Using the Python API

```python
from image_classifier import CNNImageClassifier

# Initialize the classifier
model = CNNImageClassifier()

# Predict from an image file
pred_class, pred_label, probabilities = model("path/to/image.jpg")
print(f"Predicted ASL sign: {pred_label}")
print(f"Confidence: {probabilities[pred_class]:.4f}")
```

### Using the Interactive Notebook

1. Open the `notebook_inference.ipynb` notebook in Jupyter
2. Run all cells
3. Use the file upload widget to select an image
4. View the prediction results

## 📦 Model Deployment

### TensorFlow Lite

The model is exported to TensorFlow Lite format for mobile deployment:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### TensorFlow.js

For web deployment, the model is converted to TensorFlow.js format:

```bash
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model saved_model/ tfjs_model/
```

## 📂 Project Structure

```
asl-alphabet-recognition/
├── notebook_train.ipynb        # Training notebook
├── notebook_inference.ipynb    # Inference notebook
├── image_classifier/
│   ├── __init__.py
│   └── CNNImageClassifier.py   # Model loading and prediction class
├── best_model.keras            # Trained Keras model
├── saved_model/                # TensorFlow SavedModel format
├── tfjs_model/                 # TensorFlow.js model
├── model.tflite                # TensorFlow Lite model
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 📊 Performance

The model achieves:

- 99.57% accuracy on the test set
- Fast inference time (suitable for real-time applications)

## 🔜 Future Improvements

- Real-time video recognition
- Support for full ASL sentences and phrases
- Transfer learning with more efficient architectures (MobileNet, EfficientNet)
- Deployment as a mobile application

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet) created by Akash
- TensorFlow and Keras documentation and community
