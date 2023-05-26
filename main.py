import numpy as np

import svm
from decision_tree import DecisionTree
from random_forest import RandomForest
from load_data import generate_data, load_titanic, load_reviews

def main():
    np.random.seed(123)

    # train_data, test_data = load_titanic()
    train_data, test_data = load_reviews()

    # dt = DecisionTree({"depth": 14})
    # dt.train(*train_data)
    # dt.evaluate(*train_data)
    # dt.evaluate(*test_data)

    SVM = svm.Svm()
    SVM.train(*train_data)
    SVM.evaluate(*train_data)
    SVM.evaluate(*test_data)

    # Gini index - miara nieczystości, chcemy to minimalizować
    # My mamy Gini gain - chcemy to maksymalizować
    # w korzeniu największy Gini gain

    # rf = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 100})
    # rf.train(*train_data)
    # rf.evaluate(*train_data)
    # rf.evaluate(*test_data)

if __name__=="__main__":
    main()