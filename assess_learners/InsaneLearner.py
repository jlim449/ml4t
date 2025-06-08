import BagLearner as bl
import LinRegLearner as lrl
import numpy as np
class InsaneLearner:
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.models = []
    def author(self):
        return 'jlim449'
    def add_evidence(self, data_x, data_y):
        for i in range(20):
            bag = bl.BagLearner(lrl.LinRegLearner, bags = 20, kwargs = {}, verbose = self.verbose)
            bag.add_evidence(data_x, data_y)
            self.models.append(bag)
    def query(self, points):
        predictions = []
        for model in self.models:
            predictions.append(model.query(points))
        insane_pred =  np.mean(predictions, axis = 0)
        return insane_pred