import os
import pandas as pd
import numpy as np

DATA_DIR = "../../data/head_dataset"

labels = ["normal", "tilt_forward", "tilt_side", "look_away"]

X = []
y = []

label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder = os.path.join(DATA_DIR, label)

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)

            df = pd.read_csv(path)

            for _, row in df.iterrows():
                X.append(row.values)
                y.append(label_map[label])

X = np.array(X)
y = np.array(y)

# Save
np.save("X_head.npy", X)
np.save("y_head.npy", y)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)