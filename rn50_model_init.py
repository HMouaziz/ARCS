#Settings
verbose = False

# Imports
import pathlib as pathlib
import cv2
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from printy import printy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, applications
from tensorflow.keras.utils import image_dataset_from_directory, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

printy("Required libraries were successfully imported...",'n')

#Load Datasets
directory = r"H:\MyRepositories\ARCS\data\1\crop"
path_for_data = pathlib.Path(directory)

train_df = image_dataset_from_directory(path_for_data,
                                        image_size = (128, 128),
                                        validation_split = 0.3,
                                        subset = "training",
                                        shuffle = True,
                                        batch_size = 25,
                                        seed = 123)

validation_df = image_dataset_from_directory(path_for_data,
                                             image_size = (128, 128),
                                             validation_split = 0.35,
                                             subset = "validation",
                                             shuffle = True,
                                             batch_size = 25,
                                             seed = 123)

printy("Datasets were successfully loaded...", 'n')

if verbose is True:
    print(train_df, "\n", validation_df)

printy("There are {} images in the training dataset".format(len(train_df)), 'n')
printy("There are {} images in the validation dataset".format(len(validation_df)), 'n')

#Create Validation and test dataframes
validation_batches = tf.data.experimental.cardinality(validation_df)

if verbose is True:
    print(validation_batches,"\n")

test_df = validation_df.take(validation_batches // 5)

validation_df = validation_df.skip(validation_batches // 5)

if verbose is True:
    print(test_df, "\n",validation_df)

#Show batch of images
plt.figure(figsize = (20, 20))
for images, labels in train_df.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_df.class_names[labels[i]])
choice = input("Show image batch? (y/n)")
if choice == "y":
    plt.show()

#Prefetch
autotune = tf.data.AUTOTUNE
pf_train = train_df.prefetch(buffer_size = autotune)
pf_test = test_df.prefetch(buffer_size = autotune)
pf_val = validation_df.prefetch(buffer_size = autotune)

#Model Creation

data_augmentation = tf.keras.Sequential()
data_augmentation.add(layers.RandomRotation(0.3))
data_augmentation.add(layers.RandomFlip("horizontal_and_vertical"))

image_size = (128, 128)
image_shape = image_size + (3,)

preprocess_input = applications.resnet50.preprocess_input

base_model = applications.ResNet50(input_shape = image_shape, include_top = False, weights = 'imagenet')

base_model.trainable = False
choice = input("Show base model summary? (y/n)")
if choice == "y":
    base_model.summary()

# Add classification layers
class_num = len(train_df.class_names)
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(class_num, activation ='softmax')

# Chain layers
inputs = tf.keras.Input(shape = image_shape)
x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = base_model(x)
x = global_avg(x)
outputs = output_layer(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)

choice = input("Show model summary? (y/n)")
if choice == "y":
    model.summary()

# Plot model
choice = input("Plot model to .png file? (y/n)")
if choice == "y":
    plot_model(model, to_file = "model.png", show_shapes = True)

# Define learning rate schedule and optimiser
optimizer = optimizers.Adam(learning_rate = optimizers.schedules.CosineDecay(0.001, 500))
loss = losses.SparseCategoricalCrossentropy()

#Compile Model
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

#Train Model
history = model.fit(pf_train, validation_data =pf_val, epochs = 1)

#Finetune base model layers
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

#Compile after finetune
optimizer = optimizers.RMSprop(learning_rate = optimizers.schedules.CosineDecay(0.001, 500))
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

#Continue training model
ft_epoch = 1
n_epochs =+ ft_epoch
history_fine = model.fit(pf_train, validation_data=pf_val, epochs=n_epochs, initial_epoch=history.epoch[-1])

choice = input("Train again? (y/n)")
if choice == "y":
    ft_epoch = 1
    n_epochs = + ft_epoch
    history_final = model.fit(pf_train, validation_data=pf_val, epochs=n_epochs, initial_epoch=history_fine.epoch[-1])

#Test model
image_batch, label_batch = pf_test.as_numpy_iterator().next()
pred_labels = np.argmax(model.predict(image_batch), axis = 1)

#Print results
lab_and_pred = np.transpose(np.vstack((label_batch, pred_labels)))
print(lab_and_pred)

#Save model
save = input("Save Model? (y/n)")
if save == "y":
    model.save("saved_model/ARCM(rn50)")
else:
    print("Ending run...")

