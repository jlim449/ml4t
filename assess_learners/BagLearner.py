import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import random


class BagLearner(object):
    """
    This is a Decision Tree Learner
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner,  bags, boost = False, verbose=False, **kwargs):
        """
        Constructor method
        """
        self.params = kwargs
        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.models = []


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jlim449"  # replace tb34 with your Georgia Tech username



    def resample(self, data_x, data_y):
        samples = len(data_x)
        indices = np.random.choice(samples, size = samples, replace = True)
        resampled_x = data_x[indices]
        resampled_y = data_y[indices]

        return resampled_x, resampled_y


    def add_evidence(self, data_x, data_y):

        if 'kwargs' in self.params:
            params = {**self.params['kwargs']}
        else:
            params = {}

        bags = self.bags
        for bag in range(bags):
            sample_x, sample_y = self.resample(data_x, data_y)
            model = self.learner(**params)
            model.add_evidence(sample_x, sample_y)
            self.models.append(model)


    def query(self, points):
        # For each point in the test set
        all_pred = []
        for i in range(self.bags):

            pred = self.models[i].query(points)
            all_pred.append(pred)


        return np.mean(all_pred, axis = 0)


