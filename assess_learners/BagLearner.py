""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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


    def bagging(self, data_x, data_y):

        # bagging_models = []
        all_pred = []

        bags = self.bags
        for bag in range(bags):
            sample_x, sample_y = self.resample(data_x, data_y)

            if 'kwargs' in self.params:
                params = {**self.params['kwargs']}
            else:
                params = {}
            model = self.learner(**params)
            model.add_evidence(sample_x, sample_y)
            self.models.append(model)



    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        # For each point in the test set
        all_pred = []


        for i in range(self.bags):

            pred = self.models[i].query(points)
            all_pred.append(pred)


        return np.mean(all_pred, axis = 0)


if __name__ == "__main__":
    import sys
    import math
    import numpy as np
    import os


    dir = os.path.dirname(__file__)
    folder = 'Data'
    file = 'Istanbul.csv'
    file_path = os.path.join(dir, folder, file)

    with open(file_path) as inf:
        next(inf)  # Skip header row
        # Process each line: split by comma, take elements from the second onwards, then map to float
        data_rows = []
        for line in inf:
            parts = line.strip().split(",")
            # Skip the first column (date) and convert the rest to float
            data_rows.append(list(map(float, parts[1:])))
        data = np.array(data_rows)


    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # create a learner and train it
    bag = BagLearner(lrl.LinRegLearner, bags = 20, kwargs = {}, boost=False, verbose = False)  # create a DTLearner
    bag.bagging(train_x, train_y)
    pred_train = bag.query(train_x)

    rmse = math.sqrt(((train_y - pred_train) ** 2).sum() / train_y.shape[0])
    print(f"RMSE RT Training: {rmse}")


    pred_test = bag.query(test_x)
    rmse_test = math.sqrt(((test_y - pred_test) ** 2).sum() / test_y.shape[0])
    print(f"RMSE RT Test: {rmse_test}")

    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])

    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")





    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("Regression Training Data")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print("Regression Test Data")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


