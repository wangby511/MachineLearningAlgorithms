import numpy as np
from typing import List
from classifier import Classifier


class DecisionStump(Classifier):
    def __init__(self, s: int, b: float, d: int):
        self.clf_name = "Decision_stump"
        self.s = s
        self.b = b
        self.d = d

    def train(self, features: List[List[float]], labels: List[int]):
        pass

    def predict(self, features: List[List[float]]) -> List[int]:
        result = []
        for feature in features:
            if (feature[self.d] > self.b):
                result.append(self.s)
            else:
                result.append(-self.s)
        return result
        '''
        Inputs:
        - features: the features of all test examples

        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''

        ##################################################
        # TODO: implement "predict"
        ##################################################
