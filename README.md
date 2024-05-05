<img width="1273" alt="Screenshot 2024-05-05 at 3 02 56 PM" src="https://github.com/mana9512/streamlit-facial-emotion-recognition/assets/60699342/5e3bd30f-d51e-46e0-9e04-56c86e26d617">

# streamlit-facial-emotion-recognition

# Facial Emotion Recognition Integrated with Streamlit

This project implements facial emotion recognition using two models: a custom CNN model and a ResNet50 model. It allows users to view real-time facial emotion predictions through a Streamlit user interface.

## Overview

Facial emotion recognition is a computer vision task that involves detecting emotions from images of human faces. This project uses a dataset containing 35,000 grayscale images, each with a resolution of 48x48 pixels. The dataset is split into training and test sets for model training and evaluation.

## Models

| Model           | Description                                                                                        |
|-----------------|----------------------------------------------------------------------------------------------------|
| Custom CNN      | A convolutional neural network (CNN) model implemented using Keras.                                |
|                 | - Consists of multiple convolutional and pooling layers followed by fully connected layers.        |
|                 | - Trained on the provided dataset to recognize emotions from facial images.                        |
|                 |                                       |
| ResNet50        | A pre-trained deep convolutional neural network (CNN) architecture.                                |
|                 | - Utilizes residual connections to mitigate vanishing gradient problem.                            |
|                 | - Fine-tuned on the facial emotion dataset using transfer learning.                                |

## Streamlit UI

The Streamlit user interface allows users to interact with the facial emotion recognition models in real-time. Users can access their webcam to capture live video feed, and the models will predict the dominant emotion present in the detected faces. The predicted emotion labels are displayed directly on the faces in the video feed.

## Dataset

The dataset used in this project consists of grayscale images of human faces, each labeled with one of several emotion categories. The images are preprocessed and resized to 48x48 pixels before being used for model training and evaluation.

## Usage

To run the project locally:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the Streamlit application using the command `streamlit run app.py`.
4. Access the application in your web browser and interact with the models.
