#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Lambda, Dot
from keras.models import Model

#load data
from keras import datasets as dt
from keras.optimizers import Adam
from keras.utils import plot_model

((x_train, y_train), (x_test, y_test)) = dt.mnist.load_data(path='mnist.npz')

print((x_test.shape, y_test.shape))

#reformat to fit with expected shapes for Conv2D
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_test.shape)


# if you want to make it a bit more challenging, remove most of the training examples from 1-2 classes, leave just 0-10 from each

#build a simple model based on a 2D convolution
def get_CNN_model(input_shape):
    # use keras.layers.Conv2D interleaved with keras.layers.Maxpooling2D and with some keras.layers.Dense in the end
    # alternatively use only dense layers (you will have to Flatten() the inputs)
    # keras.Sequential is recommended
    # do not forget to set the input shape to the first layer
    kernel_size = 3

    model = Sequential([
      Conv2D(32, kernel_size, input_shape=input_shape, activation='relu'),
      MaxPooling2D(),
      Conv2D(64, kernel_size, activation='relu'),
      MaxPooling2D(),
      Conv2D(128, kernel_size, activation='relu'),
      MaxPooling2D(),
      Flatten(),
      Dense(512, activation='softmax'),
    ])

    return model


model = get_CNN_model((28, 28, 1))
model.summary()


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

model_siam = get_Siammese_model(model, (28,28,1))
model_siam.summary()

# compile the model with Adam optimizer and e.g. binary cross entropy loss
# or define your own loss function, see e.g. here:
# you may add the metrics of your choice

plot_model(model_siam)
model_siam.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])


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


x1, x2, y = create_pairs(x_train, y_train, 300000)
x1_test, x2_test, y_pair_test = create_pairs(x_test, y_test, 10000)
ret = model_siam.fit([x1, x2], y, batch_size=128, epochs=1, validation_data=([x1_test, x2_test], y_pair_test))
model_siam.save("siam.model")

# 300000/300000 [==============================] - 162s 539us/step - loss: 0.1947 - acc: 0.9428 - val_loss: 0.0506 - val_acc: 0.9837

print(ret)
exit(0)

# real task: classification: get the class of the most similar item
# get K samples of each class, label with the max class with maximal similarity to the target

def get_samples(x, y):
    indices = [(y == i).nonzero()[0][0:10] for i in np.unique(y)]
    indices = np.array(indices).flatten()
    return (x[indices], y[indices])


# In[32]:


# a simple validation for a single sample. Extend this to get some overall results for the whole test set. 
# Return, e.g., accuracy
i=0
x0_sample, y0_sample = get_samples(x_train, y_train)
example = x_test[i]
x0_test = np.repeat(example[np.newaxis, :, :, :], len(y0_sample), axis=0)
pred = np.argmax(model_siam.predict([x0_test, x0_sample]))
(y_test[i], y0_sample[pred])


# ## Extensions
# 
# - Get the embeddings of individual data samples (use the original model)
# - Do some dimensionality reduction (PCA) and display the selected items based on their embeddings
# - What similarity metrics did you use & why? What are the consequences?
# - How does the individual convolutional filters work? Can you display their values for selected inputs?
# 

# In[ ]:




