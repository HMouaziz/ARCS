import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glob
import pathlib
import tensorflow as tf
import cv2, os, random
from termcolor import colored
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory, plot_model
from tensorflow.keras import layers, losses, optimizers, applications

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn import set_config
set_config(print_changed_only = False)

directory = r"H:\MyRepositories\ARCS\data\1\crop"
path_for_data = pathlib.Path(directory)


print(colored("Required libraries were succesfully imported...", color = "green", attrs = ["bold", "dark"]))

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

print(colored("The datasets were succesfully loaded...", color = "green", attrs = ["bold", "dark"]))

train_df, validation_df

print("There is {} images in the training dataset".format(len(train_df)))
print("There is {} images in the validation dataset".format(len(validation_df)))

validation_batches = tf.data.experimental.cardinality(validation_df)
validation_batches

test_df = validation_df.take(validation_batches // 5)

validation_df = validation_df.skip(validation_batches // 5)

test_df, validation_df

class_names = train_df.class_names

plt.figure(figsize = (20, 20))
for images, labels in train_df.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])


autotune = tf.data.AUTOTUNE
pf_train = train_df.prefetch(buffer_size = autotune)
pf_test = test_df.prefetch(buffer_size = autotune)
pf_val = validation_df.prefetch(buffer_size = autotune)

data_augmentation = tf.keras.Sequential()
data_augmentation.add(layers.RandomRotation(0.3))
data_augmentation.add(layers.RandomFlip("horizontal_and_vertical"))

image_size = (128, 128)
image_shape = image_size + (3,)

preprocess_input = applications.resnet50.preprocess_input

base_model = applications.ResNet50(input_shape = image_shape, include_top = False, weights = 'imagenet')

base_model.trainable = False
base_model.summary()

nclass = len(class_names)
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(nclass, activation = 'softmax')

inputs = tf.keras.Input(shape = image_shape)
x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = base_model(x)
x = global_avg(x)
outputs = output_layer(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.summary()

image_file = '/kaggle/working/model_plot.png'
plot_model(model, to_file = image_file, show_shapes = True)

optimizer = optimizers.Adam(learning_rate = optimizers.schedules.CosineDecay(0.001, 500))
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

history = model.fit(pf_train, validation_data = (pf_val), epochs = 1)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

optimizer = optimizers.RMSprop(learning_rate = optimizers.schedules.CosineDecay(0.001, 500))
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

ft_epoch = 1
n_epochs =+ ft_epoch

history_fine = model.fit(pf_train, validation_data = (pf_val), epochs = n_epochs, initial_epoch = history.epoch[-1])

image_batch, label_batch = pf_test.as_numpy_iterator().next()
pred_labels = np.argmax(model.predict(image_batch), axis = 1)

lab_and_pred = np.transpose(np.vstack((label_batch, pred_labels)))
print(lab_and_pred)

model.save()