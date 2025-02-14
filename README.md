# Handwritten-Digit-Recognition-using-CNN

## Overview
This project uses Convolutional Neural Networks (CNN) to recognize handwritten digits. Trained on the MNIST dataset, the model can accurately predict single and double-digit numbers from user input or uploaded images. A graphical user interface (GUI) built with Tkinter provides an interactive way to draw or upload images for model predictions.

## Features
- **Handwritten Digit Recognition**: The model identifies digits using a CNN-based architecture trained on the MNIST dataset.
- **User Interface**: A GUI application built with Tkinter allows users to draw digits and see predictions in real-time.
- **Model Persistence**: The model can be saved and reloaded for further use without retraining.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/Handwritten-Digit-Recognition-CNN.git
    ```
2. Navigate into the project directory:
    ```bash
    cd Handwritten-Digit-Recognition-CNN
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Run the Jupyter Notebook** to train and evaluate the CNN model on the MNIST dataset.
2. **Launch the GUI** using the provided Tkinter application script to interact with the model:
    ```bash
    python gui_application.py
    ```
3. **Draw or upload an image** of a handwritten digit to see the model’s prediction.

## Model Training
The CNN model is trained on the MNIST dataset using TensorFlow and Keras. You can adjust model parameters and training settings in the provided Jupyter Notebook.

## Acknowledgments
- The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for digit images.
- TensorFlow and Keras for model building.
- Tkinter for the GUI.
