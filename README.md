# ARCS (Aircraft Recognition and Classification System) Documentation

## Overview

ARCS (Aircraft Recognition and Classification System) is a deep learning-based project designed to classify different types of aircraft from images. The system leverages TensorFlow and Keras for model creation and training, and includes functions for data preprocessing, model evaluation, and visualization.

ARCS provides three different versions of the model:
1. **Custom Model**: A custom-built convolutional neural network (CNN).
2. **ResNet50 Model**: A model based on the ResNet50 architecture.
3. **EfficientNetB3 Model**: A model based on the EfficientNetB3 architecture.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/HMouaziz/ARCS.git
    cd ARCS
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Models

To run the ARCS models, execute the following script:

```sh
python -m core
```

This script will:
1. Import necessary libraries.
2. Load and preprocess the data.
3. Create and train the model.
4. Evaluate the model.
5. Display results and save the model if desired.

### Custom Model Example

```python
from printy import printy
import functions
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

printy("Required libraries were successfully imported...", 'n')

directory = './data/1/crop'

# Split data into dataframes
train_df, valid_df, test_df = functions.split_data(directory)

# Get Generators
batch_size = 40
train_gen, valid_gen, test_gen = functions.create_model_data(train_df, valid_df, test_df, batch_size)

# Show image sample
functions.show_images(train_gen)

# Show label count
functions.plot_label_count(train_df, 'train')

# Model Creation
model = tf.keras.models.Sequential([
    layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(43, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Plot model
plot_model(model, to_file="custom_model.png", show_shapes=True)

# Train model
history = model.fit(x=train_gen, epochs=1, validation_data=valid_gen)

# Plot history
functions.plot_training(history)

# Evaluate model
train_score = model.evaluate(train_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(test_gen, verbose=1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

# Get predictions
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)

# Generate Confusion Matrix and Classification Report
g_dict = test_gen.class_indices
classes = list(g_dict.keys())

cm = confusion_matrix(test_gen.classes, y_pred)
functions.plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')

print(classification_report(test_gen.classes, y_pred, target_names=classes))

# Save Model
save = input("Save Model? (y/n)")
if save == "y":
    model.save("saved_model/ARCM(custom)")
else:
    print("Ending run...")
```

### ResNet50 Model Example

To use the ResNet50 model, follow the same steps as the custom model but with the ResNet50 architecture.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

# Base model
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(43, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, validation_data=valid_gen, epochs=1)
```

### EfficientNetB3 Model Example

To use the EfficientNetB3 model, follow the same steps as the custom model but with the EfficientNetB3 architecture.

```python
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.keras import Sequential

# Base model
base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

# Model creation
model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer='l2', activity_regularizer='l1', bias_regularizer='l1', activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(43, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, validation_data=valid_gen, epochs=1)
```

## Data Handling

### Data Splitting

The data is split into training, validation, and test sets using the `split_data` function:

```python
train_df, valid_df, test_df = functions.split_data(directory)
```

### Data Generators

Data generators are created for training, validation, and test datasets using the `create_model_data` function:

```python
train_gen, valid_gen, test_gen = functions.create_model_data(train_df, valid_df, test_df, batch_size)
```

## Visualization

### Image Samples

Display a sample of images from the training generator:

```python
functions.show_images(train_gen)
```

### Label Count Plot

Plot the count of labels in the training data:

```python
functions.plot_label_count(train_df, 'train')
```

### Training History Plot

Plot the training and validation accuracy and loss:

```python
functions.plot_training(history)
```

### Confusion Matrix Plot

Plot the confusion matrix:

```python
functions.plot_confusion_matrix(cm, classes, title='Confusion Matrix')
```

## Saving the Model

After training, you can choose to save the model:

```python
save = input("Save Model? (y/n)")
if save == "y":
    model.save("saved_model/ARCM(custom)")
else:
    print("Ending run...")
```

## Functions Overview

### Data Handling Functions

- **define_paths(directory)**: Generates file paths and labels from a directory.
- **define_df(files, classes)**: Creates a dataframe from file paths and labels.
- **full_data(directory)**: Splits data into training, validation, and test sets from a single directory.
- **tr_ts_data(tr_dir, ts_dir)**: Splits data into training, validation, and test sets from separate directories.
- **tr_val_ts_data(tr_dir, val_dir, ts_dir)**: Splits data into training, validation, and test sets from separate directories.
- **split_data(tr_dir, val_dir=None, ts_dir=None)**: Dynamically splits data based on the directory structure.
- **create_model_data(train_df, valid_df, test_df, batch_size)**: Creates data generators from dataframes.

### Visualization Functions

- **show_images(gen)**: Displays a sample of images from a data generator.
- **plot_label_count(df, plot_title)**: Plots the count of labels in a dataframe.
- **plot_labels(df, lcount, labels, values, plot_title)**: Helper function to plot labels.
- **plot_training(hist)**: Plots training and validation accuracy and loss history.
- **plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues)**: Plots a confusion matrix.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed by Halim Mouaziz @ project-hephaestus.com © 2024
