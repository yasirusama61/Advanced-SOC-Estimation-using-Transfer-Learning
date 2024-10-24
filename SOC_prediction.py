#!/usr/bin/env python
# coding: utf-8

import scipy.io
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go


# Function to load .mat file and return input and target dataframes
def load_mat_file(file_path, input_columns, target_column):
    mat_file = scipy.io.loadmat(file_path)
    X = mat_file['X'].T
    Y = mat_file['Y'].T
    df_X = pd.DataFrame(X, columns=input_columns)
    df_Y = pd.DataFrame(Y, columns=[target_column])
    return pd.concat([df_X, df_Y], axis=1)


# Function to create sequences from the data
def create_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq), np.array(y_seq)


# Load Training Data
train_file = 'TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat'
input_columns = ['Voltage', 'Current', 'Temperature', 'Avg_voltage', 'Avg_current']
target_column = 'SOC'
df_train = load_mat_file(train_file, input_columns, target_column)

# Split into features and target
X_train = df_train[input_columns]
y_train = df_train[target_column]

# Create sequences for training
timesteps = 100
X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, timesteps)

# Define the LSTM Model
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
model = Sequential()
model.add(LSTM(30, input_shape=input_shape))
model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the Model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

# Setup callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

# Load Test Data for multiple temperature conditions

def load_test_data(test_file_path, input_columns, target_column):
    df_test = load_mat_file(test_file_path, input_columns, target_column)
    X_test = df_test[input_columns]
    y_test = df_test[target_column]
    return create_sequences(X_test.values, y_test.values, timesteps)

# Test on 0°C
test_0C_file = 'Test/02_TEST_LGHG2@0degC_Norm_(05_Inputs).mat'
X_test_seq_0C, y_test_seq_0C = load_test_data(test_0C_file, input_columns, target_column)

# Test on -10°C
test_minus10C_file = 'Test/01_TEST_LGHG2@n10degC_Norm_(05_Inputs).mat'
X_test_seq_minus10C, y_test_seq_minus10C = load_test_data(test_minus10C_file, input_columns, target_column)

# Test on 25°C
test_25C_file = 'Test/04_TEST_LGHG2@25degC_Norm_(05_Inputs).mat'
X_test_seq_25C, y_test_seq_25C = load_test_data(test_25C_file, input_columns, target_column)

# Test on 10°C
test_10C_file = 'Test/03_TEST_LGHG2@10degC_Norm_(05_Inputs).mat'
X_test_seq_10C, y_test_seq_10C = load_test_data(test_10C_file, input_columns, target_column)

# Load Validation Data (for validation during training)
validation_file = '01_TEST_LGHG2@n10degC_Norm_(05_Inputs).mat'
X_val_seq, y_val_seq = load_test_data(validation_file, input_columns, target_column)

# Train the model
history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=250, validation_data=(X_val_seq, y_val_seq), callbacks=[early_stopping, reduce_lr])

# Evaluate the model and plot results for each temperature

def evaluate_and_plot(model, X_test_seq, y_test_seq, temp_label):
    # Evaluate model
    y_pred_val = model.predict(X_test_seq)
    mae = mean_absolute_error(y_test_seq, y_pred_val)
    mse = mean_squared_error(y_test_seq, y_pred_val)
    r2 = r2_score(y_test_seq, y_pred_val)
    print(f"{temp_label} - Mean Absolute Error: {mae}, MSE: {mse}, R-squared: {r2}")

    # Plot predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test_seq))), y=y_test_seq, mode='lines', name='Actual SOC'))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred_val))), y=y_pred_val.flatten(), mode='lines', name='Predicted SOC', line=dict(dash='dash')))
    fig.update_layout(title=f"Actual vs Predicted SOC at {temp_label}", xaxis_title="Samples", yaxis_title="SOC", legend_title="Legend", height=600, width=1000)
    fig.write_html(f"actual_vs_predicted_soc_at_{temp_label.replace('°', '')}_plot.html")

# Evaluate for each test case
evaluate_and_plot(model, X_test_seq_0C, y_test_seq_0C, '0°C')
evaluate_and_plot(model, X_test_seq_minus10C, y_test_seq_minus10C, '-10°C')
evaluate_and_plot(model, X_test_seq_25C, y_test_seq_25C, '25°C')
evaluate_and_plot(model, X_test_seq_10C, y_test_seq_10C, '10°C')

# Save the model
model.save('lstm_model.h5')

