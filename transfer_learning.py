# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import scipy.io

# Load the pre-trained model with custom MSE loss function
custom_objects = {'mse': MeanSquaredError()}
pretrained_model = load_model('lstm_model_2.h5', custom_objects=custom_objects)
pretrained_model.summary()

# Load and preprocess the new dataset for transfer learning
data = pd.read_csv("data_for_tl.csv")

# Step 1: Feature Engineering
window_size = 10  # Rolling window size for averages
data['Avg_voltage'] = data['Voltage [V]'].rolling(window=window_size).mean()
data['Avg_current'] = data['Current [A]'].rolling(window=window_size).mean()
data.dropna(inplace=True)

# Additional interaction and temporal features
data['Voltage_Current_Interaction'] = data['Voltage [V]'] * data['Current [A]']
data['Temp_Current_Interaction'] = data['Cell Temperature [C]'] * data['Current [A]']
data['Temp_Rolling_Avg'] = data['Cell Temperature [C]'].rolling(window=5, min_periods=1).mean()

# Define target and feature variables
X_new = data[['Voltage [V]', 'Current [A]', 'Cell Temperature [C]', 'Avg_voltage', 'Avg_current', 
              'Voltage_Current_Interaction', 'Temp_Current_Interaction', 'Temp_Rolling_Avg']].values
y_new = data['SOC'].values

# Step 2: Create Sequences for LSTM
sequence_length = 100
def create_sequences(data, target, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i + seq_len])
        y_seq.append(target[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

X_new_seq, y_new_seq = create_sequences(X_new, y_new, sequence_length)

# Step 3: Split Data and Normalize
train_size = int(0.7 * len(X_new_seq))
val_size = int(0.15 * len(X_new_seq))

X_train, X_val, X_test = X_new_seq[:train_size], X_new_seq[train_size:train_size+val_size], X_new_seq[train_size+val_size:]
y_train, y_val, y_test = y_new_seq[:train_size], y_new_seq[train_size:train_size+val_size], y_new_seq[train_size+val_size:]

# Initialize and fit scalers
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

X_train_normalized = scaler_features.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
X_val_normalized = scaler_features.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)
X_test_normalized = scaler_features.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

y_train_normalized = scaler_target.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val_normalized = scaler_target.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
y_test_normalized = scaler_target.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Display dataset shapes
print(f"X_train shape: {X_train_normalized.shape}, y_train shape: {y_train_normalized.shape}")
print(f"X_val shape: {X_val_normalized.shape}, y_val shape: {y_val_normalized.shape}")
print(f"X_test shape: {X_test_normalized.shape}, y_test shape: {y_test_normalized.shape}")

# Step 4: Set Up Transfer Learning Model
for layer in pretrained_model.layers[:-3]:  # Freeze all but the last three layers
    layer.trainable = False

transfer_model = Sequential([
    LSTM(30, input_shape=(sequence_length, X_new_seq.shape[2])),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# Compile the Model
optimizer = Adam(learning_rate=0.000005)
transfer_model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

# Step 5: Callbacks for Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the Model
history = transfer_model.fit(
    X_train_normalized, y_train_normalized,
    epochs=100,
    batch_size=128,
    validation_data=(X_val_normalized, y_val_normalized),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate on Test Data
val_loss, val_mse = transfer_model.evaluate(X_test_normalized, y_test_normalized)
print(f"Validation Loss: {val_loss}, Validation MSE: {val_mse}")

# Step 6: Additional Evaluation Metrics
y_pred_new = transfer_model.predict(X_test_normalized)
mae = mean_absolute_error(y_test_normalized, y_pred_new)
r2 = r2_score(y_test_normalized, y_pred_new)
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Save the Fine-tuned Model
transfer_model.save("fine_tuned_soc_model.h5")

# Step 7: Visualization
# Loss Curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Transfer Learning: Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# SOC Prediction Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test_normalized, label='Actual SOC')
plt.plot(y_pred_new, label='Predicted SOC', linestyle='--')
plt.title('Comparison of Actual and Predicted SOC Values')
plt.xlabel('Sample Index')
plt.ylabel('SOC')
plt.legend()
plt.show()

# Interactive Plot with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(y_test_normalized))), y=y_test_normalized.flatten(), mode='lines', name='Actual SOC'))
fig.add_trace(go.Scatter(x=list(range(len(y_pred_new))), y=y_pred_new.flatten(), mode='lines', name='Predicted SOC', line=dict(dash='dash')))
fig.update_layout(
    title="Actual vs Predicted SOC for Transfer Learning",
    xaxis_title="Samples",
    yaxis_title="SOC",
    legend_title="Legend",
    height=600,
    width=1000
)
fig.write_html("actual_vs_predicted_soc_tl.html")
