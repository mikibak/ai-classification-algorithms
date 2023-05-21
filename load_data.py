import numpy as np
import pandas as pd

from bag_of_words import bag_of_words


def generate_data():
    TRAIN_SAMPLES = 100
    TEST_SAMPLES = 20
    FEATURE_DIM = 5

    X_train = np.random.rand(TRAIN_SAMPLES, FEATURE_DIM)
    y_train = np.random.binomial(1, 0.5, TRAIN_SAMPLES)
    X_test = np.random.rand(TEST_SAMPLES, FEATURE_DIM)
    y_test = np.random.binomial(1, 0.5, TEST_SAMPLES)
    return (X_train, y_train), (X_test, y_test)


def load_titanic():
    data = pd.read_csv("titanic.csv")
    data = data[["Pclass", "Fare", "Parch", "SibSp", "Age", "Sex", "Survived"]]
    data = data.dropna().reset_index(drop=True)
    data["Sex"] = [1 if sex == "female" else 0 for sex in data["Sex"]]
    test_idx = np.random.choice(range(data.shape[0]), round(0.2*data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)
    X_train = data_train.drop("Survived", axis=1).to_numpy()
    y_train = data_train["Survived"].to_numpy()
    X_test = data_test.drop("Survived", axis=1).to_numpy()
    y_test = data_test["Survived"].to_numpy()
    return (X_train, y_train), (X_test, y_test)


def load_reviews():
    data = pd.read_json("Magazine_Subscriptions.json",lines=True,nrows=1000)

    data = data[["overall", "verified", "reviewText"]]

    data = bag_of_words(data)

    data = data.dropna().reset_index(drop=True) # very important, drops rows containing NULL
    data["verified"] = [1 if verified else 0 for verified in data["verified"]]
    test_idx = np.random.choice(range(data.shape[0]), round(0.2*data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)
    X_train = data_train.drop("overall", axis=1).to_numpy()
    y_train = data_train["overall"].to_numpy()
    X_test = data_test.drop("overall", axis=1).to_numpy()
    y_test = data_test["overall"].to_numpy()
    return (X_train, y_train), (X_test, y_test)
