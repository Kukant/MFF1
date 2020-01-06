import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dot, Dropout
from keras.models import Model

from keras import datasets as dt
from keras.utils import plot_model

# source data are in an incompatible format - we need to construct pairs
# a batch generator would be better, but lets resort for simple static samples for the sake of simplicity
def create_pairs(x, y, size):
    pairs = []
    labels = []
    #select random indices
    a = np.random.choice(len(y), size)
    b = np.random.choice(len(y), size)
    x1 = x[a]
    x2 = x[b]
    y = [int(i) for i in y[a] == y[b]]

    return x1, x2, np.array(y)

# construct a siammese network (two towers) from a previous CNN model.
# Use some distance/similarity metric (L1, L2, cosine sim) and possibly an activation on top of it
# (or a simple tf.reduce_sum)
def get_Siammese_model(model, input_shape):
    first_input = Input(input_shape)
    second_input = Input(input_shape)

    # Generate feature vectors
    first_features = model(first_input)
    second_features = model(second_input)

    # cosine distance is computed using this Dot layer
    distance = Dot(1, normalize=True)([first_features, second_features])

    model_siam = Model(inputs=[first_input, second_input], outputs=distance)

    return model_siam

#build a simple model based on a 2D convolution
def get_CNN_model(input_shape):
    # use keras.layers.Conv2D interleaved with keras.layers.Maxpooling2D and with some keras.layers.Dense in the end
    # alternatively use only dense layers (you will have to Flatten() the inputs)
    # keras.Sequential is recommended
    # do not forget to set the input shape to the first layer
    kernel_size = 2

    model = Sequential([
        Conv2D(32, kernel_size, input_shape=input_shape, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, kernel_size, activation='relu'),
        MaxPooling2D(),
        Conv2D(128, kernel_size, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='sigmoid'),
    ])

    return model
