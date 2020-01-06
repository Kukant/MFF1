
import numpy as np

from keras.datasets import fashion_mnist
from keras.utils import plot_model

from utils import get_CNN_model, get_Siammese_model, create_pairs

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

model = get_CNN_model((28, 28, 1))
model.summary()

model_siam = get_Siammese_model(model, (28, 28, 1))
model_siam.summary()

# compile the model with Adam optimizer and e.g. binary cross entropy loss
# or define your own loss function, see e.g. here:
# you may add the metrics of your choice

plot_model(model_siam)
model_siam.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])


x1, x2, y = create_pairs(x_train, y_train, 300000)
x1_test, x2_test, y_pair_test = create_pairs(x_test, y_test, 10000)
ret = model_siam.fit([x1, x2], y, batch_size=128, epochs=1, validation_data=([x1_test, x2_test], y_pair_test))
# I have experimented with the batch size, but the results were quite similar. However the smaller the batch size,
# the more time the learning took.

# i have also tried more epochs, however i only saw improvement in accuracy and not validation_accuracy, which
# clearly meant the NN is overfitting

# save both models
model_siam.save("models/siam.model")
model.save("models/cnn.model")

# the best result:
# 300000/300000 [==============================] - 148s 494us/step - loss: 0.1930 - acc: 0.9265 - val_loss: 0.1496 - val_acc: 0.9413





