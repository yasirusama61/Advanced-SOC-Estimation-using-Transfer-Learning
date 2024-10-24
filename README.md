# Advanced SOC Estimation using Transfer Learning

This repository contains the project for **State of Charge (SOC) Estimation** using **Transfer Learning** techniques applied to lithium-ion battery data. The project leverages **LSTM (Long Short-Term Memory)** models to predict SOC, and aims to improve performance using transfer learning across different battery chemistries and operating conditions.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to estimate the SOC of lithium-ion batteries using advanced machine learning techniques, specifically Transfer Learning. This allows us to pre-train a model on one battery dataset and fine-tune it to perform well on new datasets with potentially different chemistries, temperatures, or charge/discharge cycles.

The project consists of the following key tasks:
- Data preprocessing and feature engineering for battery datasets.
- Training an LSTM model for SOC prediction.
- Applying transfer learning for cross-battery predictions.
- Visualization and analysis of model performance.

## Data Description

The dataset used for this project is the **LG 18650HG2 Li-ion Battery Data** provided by:
- **Philip Kollmeyer**, **Carlos Vidal**, **Mina Naguib**, **Michael Skells**.
- Published: 6 March 2020.
- DOI: [10.17632/cp3473x7xv.3](https://doi.org/10.17632/cp3473x7xv.3).

### Experimental Setup

- A brand new **3Ah LG HG2 cell** was tested in an 8 cu.ft. thermal chamber with a **75amp, 5 volt Digatron Firing Circuits Universal Battery Tester**.
- Voltage and current accuracy: **0.1% of full scale**.
- Tests were performed at six different temperatures, with the battery charged at **1C rate** to 4.2V, with a 50mA cut-off.

### Tests Included:
1. **Pulse Discharge HPPC Test**: 1C, 2C, 4C, and 6C discharge tests and 0.5C, 1C, 1.5C, and 2C charge tests.
2. **C/20 Discharge and Charge Test**.
3. **Discharge Tests at 0.5C, 2C, and 1C**: Tests performed before UDDS (Urban Dynamometer Driving Schedule) and Mix3 cycles.
4. **Drive Cycles**: Includes UDDS, HWFET, LA92, and US06 cycles.
5. **Temperature Variations**: Ambient temperatures of 40°C, 25°C, 10°C, 0°C, -10°C, and -20°C. Tests repeated with reduced regeneration current limits for temperatures below 10°C.

### Dataset Structure

The dataset includes both **raw** and **processed** data:
- **Raw Data**: Stored in the folder `LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020`. This data contains unprocessed voltage, current, temperature, and SOC measurements.
- **Processed Data**: Found in `LG_HG2_Prepared_Dataset_McMasterUniversity_Jan_2020`, this data is normalized and contains five columns:
  - Voltage (normalized)
  - Current (normalized)
  - Temperature (normalized)
  - Rolling averages of voltage
  - Rolling averages of current

The processed data has been pre-normalized and cleaned to prepare it for training machine learning models.

### Data Columns:
- `Time [s]`: Time in seconds.
- `Voltage [V]`: Measured cell terminal voltage.
- `Current [A]`: Measured current in amps.
- `Temperature [°C]`: Measured battery case temperature (middle of the battery).
- `SOC`: State of Charge (target for prediction).

## Requirements

The project is built using Python, and the following dependencies are required:

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow` (for building and training the LSTM model)
- `pybamm` (for battery simulation and modeling)
- `sklearn` (for preprocessing and evaluation metrics)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yasirusama61/Advanced-SOC-Estimation-using-Transfer-Learning.git
   cd Advanced-SOC-Estimation-using-Transfer-Learning

2. Install the required dependencies:
  `pip install -r requirements.txt`
3. Set up your environment (optional, using virtualenv):
  `python -m venv venv`
  `source venv/bin/activate ` # Linux/Mac
  `venv\Scripts\activate`  # Windows
4. Usage:
   ### Training the LSTM Model
    To train the LSTM model on battery data:
    - Prepare your dataset and ensure it has the required columns (Voltage, Current, Temperature, SOC).
    - Run the training script:
      `python train_lstm.py --data <data/> --epochs 50`
  
  ### Transfer Learning
  - To apply transfer learning, you can fine-tune the pre-trained LSTM model on a new dataset:

    - Load the pre-trained model.
    - Fine-tune the model:
        `python transfer_learning.py --pretrained_model <path-to-model> --data <path-to-new-dataset>`

  ### Model Architecture

The LSTM model used for SOC estimation consists of the following layers:

- **Input Layer**: The model takes a sequence input with the shape corresponding to the time-series data (voltage, current, temperature, etc.).
- **LSTM Layer**: 
  - **Units**: 30
  - **Purpose**: Captures the temporal dependencies in the SOC data.
- **Dense Layer 1**:
  - **Units**: 64
  - **Activation**: ReLU
  - **Regularization**: L2 regularization (`l2`)
- **Dropout Layer 1**: 
  - **Dropout Rate**: 0.3 (to prevent overfitting)
- **Dense Layer 2**:
  - **Units**: 32
  - **Activation**: ReLU
  - **Regularization**: L2 regularization (`l2`)
- **Dropout Layer 2**:
  - **Dropout Rate**: 0.3
- **Output Layer**:
  - **Units**: 1
  - **Activation**: Sigmoid (for regression output in the range [0, 1])

### Model Compilation

The model is compiled using the following settings:
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam optimizer with a learning rate of `0.001`
- **Metrics**: MSE (Mean Squared Error)

### Training Strategy

The model is trained with early stopping and learning rate reduction strategies:
- **Early Stopping**: Monitors validation loss, stops training if no improvement is observed for 3 consecutive epochs, and restores the best weights.
- **Reduce Learning Rate on Plateau**: Reduces the learning rate by a factor of 0.2 if the validation loss does not improve for 2 epochs. The minimum learning rate is set to `0.0001`.

### Training Configuration

- **Epochs**: 50
- **Batch Size**: 250
- **Validation Data**: Provided validation dataset (`X_val_seq`, `y_val_seq`).
- **Callbacks**: Early stopping and learning rate reduction are used as callbacks.