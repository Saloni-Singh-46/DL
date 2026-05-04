from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generator
generator = Sequential([
    Dense(16, activation='relu', input_dim=10),
    Dense(1, activation='sigmoid')
])

# Discriminator
discriminator = Sequential([
    Dense(16, activation='relu', input_dim=1),
    Dense(1, activation='sigmoid')
])

generator.summary()
discriminator.summary()
