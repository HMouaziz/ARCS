#Imports
from printy import printy
import functions
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

printy("Required libraries were successfully imported...",'n')

directory = './data/1/crop'

# Split data into dataframes
train_df, valid_df, test_df = functions.split_data(directory)

 # Get Generators
batch_size = 40
train_gen, valid_gen, test_gen = functions.create_model_data(train_df, valid_df, test_df, batch_size)

# Show image sample
functions.show_images(train_gen)

# show label count
functions.plot_label_count(train_df, 'train')