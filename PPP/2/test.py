
import numpy as np

from keras import datasets as dt
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model


# get K samples of each class, label with the max class with maximal similarity to the target
def get_samples(x, y, K=1):
    indices = [(y == i).nonzero()[0][:K] for i in np.unique(y)]
    indices = np.array(indices).flatten()
    return x[indices], y[indices]


# a simple validation for a single sample. Extend this to get some overall results for the whole test set.
# Return, e.g., accuracy
def test_model(model, x, y, x_test, y_test, class_reprezentatives):
    x0_sample, y0_sample = get_samples(x, y, class_reprezentatives)

    classes_count = len(np.unique(y))
    confusion_matrix = np.zeros((classes_count, classes_count))
    correct = 0
    for i, example in enumerate(x_test):
        # create an array of example repetitions, of len(y)
        x0_test = np.repeat(example[np.newaxis, :, :, :], len(y0_sample), axis=0)
        # compare the examples array with x0_sample
        pred_list = model.predict([x0_test, x0_sample])
        # get the index of maximum similarity
        pred = np.argmax(pred_list)

        real_class = y_test[i]
        predicted_class = y0_sample[pred]
        confusion_matrix[real_class][predicted_class] += 1
        if real_class == predicted_class:
            correct += 1

    accuracy = correct / len(x_test)
    # normalize the matrix
    confusion_matrix /= confusion_matrix.max()
    return accuracy, confusion_matrix


def plot_confusion_matrix(confusion_matrix, title):
    grid_ticks = range(confusion_matrix.shape[0])
    im = plt.imshow(confusion_matrix)
    plt.colorbar(im)
    plt.title(title)
    plt.yticks(grid_ticks, grid_ticks)
    plt.xticks(grid_ticks, grid_ticks)
    plt.xlabel("Actual class")
    plt.ylabel("Predicted class")
    plt.savefig("{}.png".format(title.lower().replace(" ", "_")))
    plt.show()


class Dataset:
    def __init__(self, data, test_len=100):
        ((self.x, self.y), (self.x_test, self.y_test)) = data
        self.x = np.expand_dims(self.x, axis=3)
        self.x_test = np.expand_dims(self.x_test, axis=3)[:test_len]
        self.y_test = self.y_test[:test_len]


def main():
    # load saved model
    model_siam = load_model("models/siam.model")

    # load datasets
    fashion = Dataset(fashion_mnist.load_data(), 500)
    digits = Dataset(dt.mnist.load_data(path='mnist.npz'), 500)

    # combine data
    both_data = (
        (np.concatenate((fashion.x[:, :, :, 0], digits.x[:, :, :,  0])), np.concatenate((fashion.y, digits.y + 10))),
        (np.concatenate((fashion.x_test[:, :, :,  0], digits.x_test[:, :, :,  0])), np.concatenate((fashion.y_test, digits.y_test + 10)))
    )

    # combinated datasets dataset
    combo = Dataset(both_data, 1000)

    # test fashion first:
    accuracy, confusion_matrix = test_model(model_siam, combo.x, combo.y, fashion.x_test, fashion.y_test, 1)
    print("fashion accuracy: {}".format(accuracy))
    # fashion accuracy: 0.742
    plot_confusion_matrix(confusion_matrix, "Fashion confusion matrix")

    # test digits:
    accuracy, confusion_matrix = test_model(model_siam, combo.x, combo.y, digits.x_test, digits.y_test + 10, 1)
    print("digits accuracy: {}".format(accuracy))
    # fashion accuracy: 0.126
    plot_confusion_matrix(confusion_matrix, "Digits confusion matrix")

    # as we see, the digits accuracy is pretty bad. However we can try raising the number of representatives:

    # plot the relation between number of representatives and accuracy
    # use only first 100 for better time performance
    accuracies = []
    for representatives_cnt in range(1, 14, 2):
        acc, _ = test_model(model_siam, combo.x, combo.y, digits.x_test[:100], digits.y_test[:100] + 10, representatives_cnt)
        accuracies.append(acc)

    print(accuracies)
    # [0.18, 0.44, 0.36, 0.36, 0.42, 0.39, 0.41]
    plt.plot(range(1, 14, 2), accuracies)
    plt.xlabel("Number of representatives")
    plt.ylabel("Accuracy")
    plt.title("Representatives-accuracy relation")
    plt.savefig("representatives_accuracy_relation.png")
    plt.show()
    # as we can see, the accuracy rises significantly.

    # lets find out how similar are classes
    # all we need to do is to add the confusion matrix over its diagonal
    # meaning we are going to add numbers with indexes ie: (1,2), (2,1)
    _, cm = test_model(model_siam, combo.x, combo.y, combo.x_test, combo.y_test, 3)
    for i in range(cm.shape[0]):
        for j in range(i, cm.shape[1]):
            cm[i][j] += cm[j][i]
            cm[j][i] = 0

    cm /= cm.max()
    plot_confusion_matrix(cm, "Classes similarity")
    # there are few classes, that our network considers quite similar:
    # fashion: (0, 3) = (tshirt, dress), (2, 6) = (pullover, shirt),
    #          (4, 6) = (coat, shirt)
    # digits: (17, 13) = 7, 3
    # mix: (5, 15) = (sandal, 5)


if __name__ == '__main__':
    main()


