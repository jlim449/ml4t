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


class Node():
    # reference: https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, mse = None, prediction=None, depth = None):
        """Initialize a node in the decision tree."""
        # self.root = None
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left  # Left child node
        self.right = right
        self.mse = mse
        self.prediction = prediction
        self.depth = depth
  		  	   		 	 	 			  		 			 	 	 		 		 	
class DTLearner(object):
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    def __init__(self, max_depth = None, verbose=False, min_samples_split=1):
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        Constructor method  		  	   		 	 	 			  		 			 	 	 		 		 	
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # pass
  		  	   		 	 	 			  		 			 	 	 		 		 	
    def author(self):  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
        :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        return "jlim449"  # replace tb34 with your Georgia Tech username


    def calculate_mse(self, data_y):

        """Calculate the Mean Squared Error (MSE) of the given data_y."""

        mean_y = np.mean(data_y)
        mse = np.mean((data_y - mean_y) ** 2)
        return mse


    def split(self, data_x, data_y, feature_index, threshold):

        left_mask = data_x[:, feature_index] <= threshold
        right_mask = data_x[:, feature_index] > threshold

        left_x = data_x[left_mask]
        left_y = data_y[left_mask]

        right_x = data_x[right_mask]
        right_y = data_y[right_mask]

        return left_x, right_x, left_y, right_y


    def find_best_split(self, data_x, data_y):
    #find the splits that will results in minimum MSE for each feature
        best_mse = float('inf')
        best_feature_index = None
        n_samples, features =  data_x.shape


        for feature_index in range(features):
            vals = data_x[:, feature_index]
            # unique_values = np.sort(unique_values)
            sort_idx = np.argsort(vals)


            sorted_x = vals[sort_idx]
            sorted_y = data_y[sort_idx]


            # split samples

            right_samples = n_samples
            right_sum = np.sum(sorted_y)
            right_sum_sq = np.sum(sorted_y ** sorted_y)

            left_samples = 0
            left_sum = 0
            left_sum_sq = 0



            if len(sorted_y) == 1:
                return {
                    'feature_index': feature_index,
                    'threshold': sorted_x[0],
                    'mse': self.calculate_mse(sorted_y)
                }




            for i in range(n_samples - 1):
                yi = float(sorted_y[i])


                left_samples += 1
                left_sum += yi
                left_sum_sq += yi ** 2


                right_samples -= 1
                right_sum -= yi
                right_sum_sq -= yi ** 2


                mse_left = 1 / left_samples * (left_sum_sq - (left_sum ** 2) / left_samples)
                mse_right = 1 / right_samples * (right_sum_sq - (right_sum ** 2) / right_samples)

                mse_split = (left_samples * mse_left + right_samples * mse_right) / (left_samples + right_samples)
                threshold = 0.5 * (sorted_x[i] + sorted_x[i + 1])


                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feature_index = feature_index
                    best_threshold = threshold


        return {
            'feature_index': best_feature_index,
            'threshold': best_threshold,
            'mse': best_mse
        }




    def build_tree(self, data_x, data_y, current_depth=0):

        if current_depth >= self.max_depth:
            return Node(feature_index=None, threshold=None, left=None, right=None, prediction=np.mean(data_y), depth = current_depth)

        if data_x.shape[0] == 1 or data_x.shape[0] < self.min_samples_split:
            return Node(feature_index=None, threshold=None, left=None, right=None, prediction=data_y[0] , depth = current_depth)


        optimal_movements = self.find_best_split(data_x, data_y)

        left_x, right_x, left_y, right_y = self.split(data_x, data_y,
            optimal_movements['feature_index'], optimal_movements['threshold'])


        # Recursively build subtrees
        left_subtree = self.build_tree(left_x, left_y, current_depth + 1)
        right_subtree = self.build_tree(right_x, right_y, current_depth + 1)

        return Node(
                    feature_index=optimal_movements['feature_index'],
                    threshold=optimal_movements['threshold'],
                    left=left_subtree,
                    right=right_subtree,
                    mse=optimal_movements['mse'],
                    prediction=np.mean(data_y),
                    depth=current_depth
                    )

    def add_evidence(self, data_x, data_y):  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        Add training data to learner  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
        :param data_x: A set of feature values used to train the learner  		  	   		 	 	 			  		 			 	 	 		 		 	
        :type data_x: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	 	 			  		 			 	 	 		 		 	
        :type data_y: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
        # slap on 1s column so linear regression finds a constant term
        new_data_x = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data_x[:, 0 : data_x.shape[1]] = data_x
  		  	   		 	 	 			  		 			 	 	 		 		 	
        # build and save the model  		  	   		 	 	 			  		 			 	 	 		 		 	
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(  		  	   		 	 	 			  		 			 	 	 		 		 	
            new_data_x, data_y, rcond=None  		  	   		 	 	 			  		 			 	 	 		 		 	
        )  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    def query(self, points):  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        Estimate a set of test points given the model we built. 			  		 			 	 	 		 		 	
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	 	 			  		 			 	 	 		 		 	
        :type points: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
        :return: The predicted result of the input data according to the trained model  		  	   		 	 	 			  		 			 	 	 		 		 	
        :rtype: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[  		  	   		 	 	 			  		 			 	 	 		 		 	
            -1  		  	   		 	 	 			  		 			 	 	 		 		 	
        ]  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    print("the secret clue is 'zzyzx'")
#  split the leaf into two parts
#  in the first root find the optimal point
#  where the rmse is minimized

    import numpy as np

    # Create simple test data
    np.random.seed(42)
    x = np.random.random((100, 2)) * 10
    y = x[:, 0] + x[:, 1] + np.random.normal(0, 0.1, 100)

    # Test the learner
    learner = DTLearner(    1, verbose=True, min_samples_split=2)  # Create a DTLearner with max depth of 5 and min_samples_split of 2
    learner.add_evidence(x, y)




    feature_idx = learner.find_best_split(x, y)
    a = learner.build_tree(x, y)
    a




    # Test predictions
    test_x = np.array([[5, 5], [1, 1], [9, 9]])
    predictions = learner.query(test_x)

