# Brain Tumor Detection with Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?style=flat&logo=pytorchlightning&logoColor=white)](https://lightning.ai/)

A deep learning project for detecting brain tumors in MRI scans using convolutional neural networks.

## Table of Contents
1. [About The Project](#about-the-project)
2. [Model Architecture](#model-architecture)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Results](#results)

## About The Project
This project implements a CNN-based binary classifier to detect the presence of brain tumors in MRI scans. The model is built using PyTorch and PyTorch Lightning for efficient training and experimentation.

## Model Architecture
The model consists of:
- Two convolutional layers with max pooling
- Three fully connected layers
- Dropout regularization
- Binary classification output with sigmoid activation

## Dataset
The dataset comprises MRI brain scans in two categories:
- Images with tumors (positive cases)
- Images without tumors (negative cases)

Images are preprocessed by:
- Converting to grayscale
- Resizing to 250x200 pixels
- Normalizing pixel values
- Applying data augmentation (horizontal flips)

## Training
The model is trained using:
- Binary cross entropy loss
- Adam optimizer
- Model pruning for efficiency
- GPU acceleration
- TensorBoard logging

## Results
Training metrics and model performance can be monitored through TensorBoard logs in the `logs/Brain_Tumor_Detection` directory.
