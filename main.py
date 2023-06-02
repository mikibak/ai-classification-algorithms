import numpy as np
import matplotlib.pyplot as plt

import svm
import time
import NeuralNetwork
from Confusion_matrix import ConfusionMatrix
from decision_tree import DecisionTree
from random_forest import RandomForest
from load_data import generate_data, load_titanic, load_reviews

def main():

    confMatrix = ConfusionMatrix()
    np.random.seed(123)
    data_sizes = [50, 100, 200, 500, 890]
    # data_sizes = [50, 100, 200, 500]
    train_accuracy_svm = []
    test_accuracy_svm = []
    test_accuracy_NN = []
    train_accuracy_NN = []
    time_SVM = []
    time_NN = []

    for data_size in data_sizes:
        train_data, test_data = load_titanic()
        # train_data, test_data = load_reviews(data_size)
        start = time.time()
        NN = NeuralNetwork.NeuraltNetwork()
        NN.train(train_data, test_data)
        end = time.time()
        time_NN.append(end - start)
        test_accuracy_NN.append(NN.evaluate(*test_data))
        train_accuracy_NN.append(NN.evaluate(*train_data))
        confMatrix.addNN(NN.getPredictions(test_data[0]), test_data[1])
        # dt = DecisionTree({"depth": 14})
        # dt.train(*train_data)
        # dt.evaluate(*train_data)
        # dt.evaluate(*test_data)

        start = time.time()
        SVM = svm.Svm()
        SVM.train(*train_data)
        end = time.time()
        time_SVM.append(end - start)
        confMatrix.addSvn(SVM.getPredictions(test_data[0]),test_data[1])
        train_accuracy_svm.append(SVM.evaluate(*train_data))
        test_accuracy_svm.append(SVM.evaluate(*test_data))

    # confMatrix.print()
    print("SVM")
    print(train_accuracy_svm)
    print(test_accuracy_svm)
    print("NN")
    print(train_accuracy_NN)
    print(test_accuracy_NN)

    plt.title("Train and test accuracy for SVM (Titanic survivors)")
    plt.xlabel('Data size (number of opinions)')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(bottom=0.25)
    plt.plot(data_sizes, train_accuracy_svm, label="Train accuracy")
    plt.plot(data_sizes, test_accuracy_svm, label="Test accuracy")
    plt.legend(loc='best')
    plt.show()

    plt.title("Train and test accuracy for NN (Titanic survivors)")
    plt.xlabel('Data size (number of opinions)')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(bottom=0.25)
    plt.plot(data_sizes, train_accuracy_NN, label="Train accuracy")
    plt.plot(data_sizes, test_accuracy_NN, label="Test accuracy")
    plt.legend(loc='best')
    plt.show()

    plt.title("Test accuracy for SVM and NN (Titanic survivors)")
    plt.xlabel('Data size (number of opinions)')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(bottom=0.25)
    plt.plot(data_sizes, test_accuracy_svm, label="Test accuracy SVM")
    plt.plot(data_sizes, test_accuracy_NN, label="Test accuracy NN")
    plt.legend(loc='best')
    plt.show()

    plt.title("Train accuracy for SVM and NN (Titanic survivors)")
    plt.xlabel('Data size (number of opinions)')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(bottom=0.25)
    plt.plot(data_sizes, train_accuracy_svm, label="Train accuracy SVM")
    plt.plot(data_sizes, train_accuracy_NN, label="Train accuracy NN")
    plt.legend(loc='best')
    plt.show()

    plt.title("Time of execution for SVM and NN (Titanic survivors)")
    plt.xlabel('Data size (number of opinions)')
    plt.ylabel('Time [s]')
    plt.subplots_adjust(bottom=0.25)
    plt.plot(data_sizes, time_SVM, label="Time of SVM")
    plt.plot(data_sizes, time_NN, label="Time of NN")
    plt.legend(loc='best')
    plt.show()

    # Gini index - miara nieczystości, chcemy to minimalizować
    # My mamy Gini gain - chcemy to maksymalizować
    # w korzeniu największy Gini gain

    # rf = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 100})
    # rf.train(*train_data)
    # rf.evaluate(*train_data)
    # rf.evaluate(*test_data)
    return

if __name__=="__main__":
    main()
    