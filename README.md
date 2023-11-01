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
``` python3
#loading the training linear acceleration data
train_dir= '/content/drive/MyDrive/UCI HAR Dataset/train/Inertial Signals'
x_acc_train = pd.read_csv(train_dir+'/body_acc_x_train.txt', header=None, delim_whitespace=True)
y_acc_train = pd.read_csv(train_dir+'body_acc_y_train.txt', header=None, delim_whitespace=True)
z_acc_train = pd.read_csv(train_dir+'body_acc_z_train.txt', header=None, delim_whitespace=True)

#laoding the output training data
Y_train= pd.read_csv('/content/drive/MyDrive/CS5062_AssessmentII_Dataset/train/y_train.txt', header=None, delim_whitespace=True)
```

### Processing the Data

The data is imported and processed into input sequences for model training. The input data includes linear acceleration and angular velocity. One-hot encoding is applied to the output classes to convert them into binary vectors.
```python3
# The vectors are stacked along the second axis (axis=2) to create a single input sequence.
X_train = np.stack((x_acc_train, y_acc_train, z_acc_train, x_gyro_train, y_gyro_train, z_gyro_train), axis=2)
X_test = np.stack((x_acc_test, y_acc_test, z_acc_test, x_gyro_test, y_gyro_test, z_gyro_test), axis=2)
# The shape of X_train(input_sequence) will be (7352,128,6).
```

```python3
# zero-offset class values
Y_train = Y_train - 1
Y_test = Y_test - 1
# one hot encode y
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
```

### Building the Models

We discuss the architecture and implementation details of various models, including LSTM, Simple RNN, Bidirectional RNN, and CNN. These models are chosen because they can capture temporal dependencies and are suitable for sequence data.
```python3
# define an LSTM model
def lstm_model(X_train, Y_train, X_test, Y_test):
 n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]
 model = Sequential()
 model.add(LSTM(128, input_shape=(n_timesteps,n_features)))
 model.add(Dropout(0.5))
 model.add(Dense(100, activation='relu'))
 model.add(Dense(n_outputs, activation='softmax'))
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model
```

```python3
# Defining the convolutional LSTM model
from keras.layers import ConvLSTM2D
import matplotlib.pyplot as plt

def convLSTM_model(X_train, Y_train, X_test, Y_test):
 # define model
 verbose, epochs, batch_size = 0, 25, 64
 n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]
 # reshape into subsequences (samples, time steps, rows, cols, channels)
 n_steps, n_length = 4, 32
 X_train = X_train.reshape((X_train.shape[0], n_steps, 1, n_length, n_features))
 X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
 print(X_train.shape)
 print(X_test.shape)
 # define model
 model = Sequential()
 model.add(ConvLSTM2D(filters=128, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
 model.add(Dropout(0.5))
 model.add(Flatten())
 model.add(Dense(100, activation='relu'))
 model.add(Dense(n_outputs, activation='softmax'))
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
```

### Reporting Performance and Evaluating Models

We evaluate the model performance and report the results, including metrics like accuracy, precision, recall, F1-score, confusion matrices, and Area Under the Receiver Operating Characteristic (AUROC) scores. The report includes detailed analysis and visualizations for each model.
```python3
# Predict on the test set
def resultMetrics(model, history):
  y_pred = model.predict(X_test)
  y_pred_classes = np.argmax(y_pred, axis=1)
  y_true = np.argmax(Y_test, axis=1)
  # Accuracy
  accuracy = accuracy_score(y_true, y_pred_classes)
  print("Accuracy: %.3f" % accuracy)
  # Precision, recall, and classification report
  report = classification_report(y_true, y_pred_classes)
  print("Classification Report:\n", report)
  plt_conf_matrix(Y_test, y_pred)
  # AUROC
  auroc = roc_auc_score(Y_test, y_pred, multi_class='ovr')
  print("AUROC: %.3f" % auroc)
  plt_auroc(y_true, y_pred,6)
  # Plot training history
  plot_training(history)
```

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


[1] [Keras Recurrent Layers Documentation](https://keras.io/api/layers/recurrent_layers/).

[2] [How to Develop RNN Models for Human Activity Recognition](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/).

[3] [An Introduction to Recurrent Neural Networks and the Math that Powers Them](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/).

[4] [Recall, Precision, F1, ROC, AUC, and Everything in Machine Learning](https://medium.com/swlh/recall-precision-f1-roc-auc-and-everything-542aedf322b9).

