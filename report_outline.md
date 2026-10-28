# Project Report Outline

## Title
Real-Time Smart Agriculture Prediction Using Azure Machine Learning and IoT Edge

## Abstract
This project demonstrates a hardware-free Internet of Things (IoT) and machine learning workflow for smart agriculture. A Python simulator generates sensor values for soil moisture, temperature, humidity, and rainfall. A Random Forest classifier trained on synthetic data predicts whether irrigation is required. Predictions are executed locally to demonstrate edge inference, and a Streamlit dashboard displays current readings, probabilities, and recent history. The trained model can optionally be registered in Azure Machine Learning.

## Problem Statement
Farmers need timely indications of when crops may require irrigation. Physical deployments can be costly for a classroom prototype, so this project simulates sensor data and evaluates a machine learning workflow entirely in software.

## Objectives
1. Generate simulated IoT sensor readings.
2. Prepare a labeled dataset and train a classification model.
3. Evaluate the model on a held-out test set.
4. Run predictions locally to simulate edge inference.
5. Display readings and results in a dashboard.
6. Optionally register the model in Azure Machine Learning.

## System Architecture
Python sensor simulator → feature table → Random Forest classifier → local edge-style inference → Streamlit dashboard. Optional cloud model management through Azure ML.

## Dataset
The dataset is generated locally. Features: soil moisture (%), temperature (°C), humidity (%), rainfall (mm). Target: irrigation needed (1) or not needed (0). The labels are generated using a transparent heuristic with slight noise; the dataset is synthetic and not a substitute for field data.

## Methodology
- Generate 2,500 synthetic examples.
- Split into 80% training and 20% test data.
- Train a Random Forest classifier.
- Evaluate using accuracy, classification report, and confusion matrix.
- Save model with joblib.
- Simulate new readings and predict locally.

## Results
Insert the test accuracy and confusion matrix printed by `python train_model.py`. Add screenshots of the dashboard and prediction history.

## Limitations
Synthetic data may not represent real crop, soil, or climate behavior. The local inference stage simulates edge computing but is not a deployment to an actual edge device. Azure ML registration is optional and requires an Azure workspace.

## Conclusion
The prototype demonstrates the main software components of an IoT-edge ML workflow without physical hardware. Future work could use real agronomic data, crop-specific models, and an actual edge device if available.
