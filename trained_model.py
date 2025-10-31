# trained_model.py

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import json

train_csv = "DATASET/Training_set.csv"
train_dir = "DATASET/train"
model_file = "butterfly_model.h5"

train_df = pd.read_csv(train_csv)
print("Training CSV columns:", train_df.columns)
print(train_df.head())

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="filename",
    y_col="label",
    subset="training",
    target_size=(128,128),
    class_mode="categorical",
    batch_size=32
)

val_data = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="filename",
    y_col="label",
    subset="validation",
    target_size=(128,128),
    class_mode="categorical",
    batch_size=32
)

num_classes = len(train_data.class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop]
)

model.save(model_file)
print(f"Model saved to {model_file}")

with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("Class indices saved to class_indices.json")
