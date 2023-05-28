import numpy as np
import matplotlib.pyplot as plt

import svm
from decision_tree import DecisionTree
from random_forest import RandomForest
from load_data import generate_data, load_titanic, load_reviews

def main():
    np.random.seed(123)
    # data_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    data_sizes = [2000]
    train_accuracy = []
    test_accuracy = []

    for data_size in data_sizes:
        # train_data, test_data = load_titanic()
        train_data, test_data = load_reviews(data_size)

        # dt = DecisionTree({"depth": 14})
        # dt.train(*train_data)
        # dt.evaluate(*train_data)
        # dt.evaluate(*test_data)

        SVM = svm.Svm()
        SVM.train(*train_data)
        train_accuracy.append(SVM.evaluate(*train_data))
        test_accuracy.append(SVM.evaluate(*test_data))

    print(train_accuracy)
    print(test_accuracy)

    plt.title("Train and test accuracy")
    plt.xlabel('Data size (number of opinions)')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(bottom=0.25)
    plt.plot(data_sizes, train_accuracy, label="Train accuracy")
    plt.plot(data_sizes, test_accuracy, label="Test accuracy")
    plt.legend(loc='best')
    plt.show()
    # Gini index - miara nieczystości, chcemy to minimalizować
    # My mamy Gini gain - chcemy to maksymalizować
    # w korzeniu największy Gini gain

    # rf = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 100})
    # rf.train(*train_data)
    # rf.evaluate(*train_data)
    # rf.evaluate(*test_data)

if __name__=="__main__":
    main()