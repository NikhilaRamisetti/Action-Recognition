# Action-Recognition

### Human Activity Recognition with Sequence Classification

**Develop Learning-Based Models for Sequence Classification**

This repository contains the code and report for sequence-based classification models for Human Activity Recognition. The problem involves classifying activities from accelerometer and gyroscope sensor data, and we explore various deep learning models for this task.

## Problem Statement

The dataset involves human activity recognition, a multi-class sequence classification problem using sensor data from accelerometers and gyroscopes. The goal is to classify activities such as walking, walking upstairs, walking downstairs, sitting, standing, and lying based on the sensor inputs.

DataSet: https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset

## Implementing Learning-Based Models

In this subtask, we cover the following steps:

1. Importing the raw data.
2. Processing the data.
3. Building and explaining various models for sequence classification.

We implement the following models:
1. Simple RNN
2. LSTM
3. Bidirectional Recurrent Network
4. CNN
5. Reinforcement Learning

### Importing the raw data

#### Import necessary libraries
``` python3
#Importing necessary libraries to import and process the data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib notebook
from pandas import Series, DataFrame, read_csv

from numpy import mean,std,dstack
from keras.utils import to_categorical
```

### Processing the Data

The data is imported and processed into input sequences for model training. The input data includes linear acceleration and angular velocity. One-hot encoding is applied to the output classes to convert them into binary vectors.

### Building the Models

We discuss the architecture and implementation details of various models, including LSTM, Simple RNN, Bidirectional RNN, and CNN. These models are chosen because they can capture temporal dependencies and are suitable for sequence data.

### Reporting Performance and Evaluating Models

We evaluate the model performance and report the results, including metrics like accuracy, precision, recall, F1-score, confusion matrices, and Area Under the Receiver Operating Characteristic (AUROC) scores. The report includes detailed analysis and visualizations for each model.

### Results

#### LSTM Results
- Accuracy: 0.904
- AUROC: 0.992

#### CNN – LSTM Results
- Accuracy: 0.875
- AUROC: 0.974

#### Simple RNN Results
- Accuracy: 0.876
- AUROC: 0.983

#### Bidirectional – RNN Results
- Accuracy: 0.913
- AUROC: 0.987

## Justification and Evaluation Findings

The report provides insights into the performance of various neural network architectures for human activity recognition. Key findings and justifications include:

- Impact of model variations on performance.
- Effect of network parameters like layers, activation functions, and optimizers.
- Importance of data pre-processing and post-processing.
- Evaluation metrics including accuracy, AUROC, and visualizations.
- Implications and further considerations for improving model performance.

In conclusion, this repository covers the development of models, performance evaluation, and key findings for effective human activity recognition. The code and report provide valuable insights for researchers and practitioners working on sequence classification problems.

## References

[1] [Comparative Study on Classic Machine Learning Algorithms](https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222).

[2] [Combination of Naive Bayes Classifier and K-Nearest Neighbor (cNK) in the Classification-Based Predictive Models](https://www.researchgate.net/publication/289846895_Combination_of_Naive_Bayes_Classifier_and_K-Nearest_Neighbor_cNK_in_the_Classification_Based_Predictive_Models).

[3] [Keras Recurrent Layers Documentation](https://keras.io/api/layers/recurrent_layers/).

[4] [How to Develop RNN Models for Human Activity Recognition](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/).

[5] [An Introduction to Recurrent Neural Networks and the Math that Powers Them](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/).

[6] [Recall, Precision, F1, ROC, AUC, and Everything in Machine Learning](https://medium.com/swlh/recall-precision-f1-roc-auc-and-everything-542aedf322b9).

[7] [Wasserstein GAN - ArXiv, 1701.07875](https://arxiv.org/abs/1701.07875).

[8] [Conditional Generative Adversarial Nets - ArXiv, 1411.1784](https://arxiv.org/abs/1411.1784).

[9] [Generative Adversarial Nets - Neural Information Processing Systems (NIPS)](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf).