import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

# Dummy dataset (time series)
X = np.random.rand(1000, 10, 1)
y = np.random.rand(1000, 1)

def build_model(model_type):
    model = Sequential()

    if model_type == "RNN":
        model.add(SimpleRNN(32, input_shape=(10, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(32, input_shape=(10, 1)))
    elif model_type == "GRU":
        model.add(GRU(32, input_shape=(10, 1)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train all models
for model_type in ["RNN", "LSTM", "GRU"]:
    print(f"\nTraining {model_type}")
    model = build_model(model_type)
    model.fit(X, y, epochs=3, batch_size=32)