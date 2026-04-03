import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# -------- LOAD DATA --------
X = np.load("X_head.npy")
y = np.load("y_head.npy")
X=(X-X.mean(axis=0))/X.std(axis=0)

# -------- TRAIN TEST SPLIT --------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- MODEL --------
model = Sequential([
    Dense(128, activation='relu', input_shape=(3,)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------- TRAIN --------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

# -------- SAVE --------
model.save("../../models/head_model.keras")

print("✅ Model trained and saved!")