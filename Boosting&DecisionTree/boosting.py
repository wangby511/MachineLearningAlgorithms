import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs  # set of weak classifiers to be considered
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        self.betas = []  # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:

        _sum = 0
        for i in range(len(self.clfs_picked)):
            f = self.clfs_picked[i]
            _sum += np.multiply(np.array(f.predict(features)), self.betas[i])
        result = np.sign(_sum).tolist()
        return result


    '''
    Inputs:
    - features: the features of all test examples

    Returns:
    - the prediction (-1 or +1) for each example (in a list)
    '''
    ########################################################
    # TODO: implement "predict"
    ########################################################


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        N = np.array(features).shape[0]
        D = np.array(features).shape[1]
        w_t = np.ones(N) / N
        for t in range(self.T):
            minLoss = float('Inf')
            for clf in self.clfs:
                loss = -np.multiply(np.array(clf.predict(features)), np.array(labels))
                loss[loss < 0] = 0
                loss = np.sum(np.multiply(loss, w_t))
                if (loss < minLoss):
                    minLoss = loss
                    pickedClf = clf
            e_t = minLoss
            result = pickedClf.predict(features)
            b_t = 0.5 * np.log((1 - e_t) / e_t)
            for i in range(N):
                if (result[i] != labels[i]):
                    w_t[i] *= np.exp(b_t)
                else:
                    w_t[i] *= np.exp(-b_t)

            _normsum = np.sum(w_t)
            w_t = w_t / _normsum

            self.clfs_picked.append(pickedClf)
            self.betas.append(b_t)
        '''
        Inputs:
        - features: the features of all examples
        - labels: the label of all examples

        Require:
        - store what you learn in self.clfs_picked and self.betas
        '''

    ############################################################
    # TODO: implement "train"
    ############################################################


    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
