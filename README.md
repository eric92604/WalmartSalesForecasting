# Temporal Fusion Transformer Model Training and Prediction for Walmart Sales Forecasting
This repository contains code for training and predicting time series data using the Temporal Fusion Transformer (TFT) model implemented with PyTorch and PyTorch Lightning. The dataset being used is taken from Kaggle - M5 Forecasting Accuracy competition.

## Setup
Ensure you have Python 3.10 installed, then install the packages from requirements.txt

## Structure

DataManager: Handles data loading, preprocessing, and memory management.
ModelSetup: Sets up the Temporal Fusion Transformer model, including the training environment.
ModelTrainer: Manages the training process, including finding optimal learning rates, batch sizes, and hyperparameters.
