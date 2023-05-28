from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class Svm:

    def __init__(self):
        self.svm = SVC(kernel='linear', random_state=1, C=0.05)
    def train(self, X, y):
        # Training a SVM classifier using SVC class
        self.svm.fit(X, y)

    def evaluate(self, X, y):
        y_pred = self.svm.predict(X)
        print('Accuracy: %.3f' % accuracy_score(y, y_pred))
        return accuracy_score(y, y_pred)