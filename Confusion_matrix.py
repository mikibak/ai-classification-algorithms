import numpy as np
from matplotlib import pyplot as plt


class ConfusionMatrix:
    def __init__(self):
        self.svmConfucionMatrix = []
        self.NNConfusionMatrix = []

    def addSvn(self,svm_predicted,svm_true):
        self.svmConfucionMatrix.append((svm_predicted,svm_true))

    def addNN(self,NN_predicted,NN_true):
        self.NNConfusionMatrix.append((NN_predicted,NN_true))


    def print(self):
        iter=0
        mode = ["NN","SVM"]
        plt.figure(figsize=(10, 10))
        plt.title("confusion matrixes for NN and SVM")
        for algorythm in [self.NNConfusionMatrix,self.svmConfucionMatrix]:
            for i in range(len(algorythm)):
                plt.subplot(2, len(self.NNConfusionMatrix), 1+i+((len(self.NNConfusionMatrix))*iter))
                # plt.xticks([])
                # plt.yticks([])
                names = ["TP", "FP", "FN", "TN"]
                predicted, true = algorythm[i]
                values = [0, 0, 0, 0]
                for ii in range(len(predicted)):
                    values[predicted[ii] * 2 + true[ii]] = values[predicted[ii] * 2 + true[ii]] + 1
                plt.bar(names, values, tick_label=names, width=0.8, color=['green', 'red', 'red', 'green'])
                plt.gca().set_title(mode[iter]+" for test size: " + str(np.sum(values)))
            iter=iter+1
        plt.show()
