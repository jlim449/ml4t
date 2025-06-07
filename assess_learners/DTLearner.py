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

class DTLearner(object):
    """
    This is a Decision Tree Learner
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, verbose=False, leaf_size=1):
        """
        Constructor method
        """

        self.verbose = verbose
        self.leaf_size = leaf_size

        # pass

    def author(self):
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
        :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
        """
        return "jlim449"  # replace tb34 with your Georgia Tech username


    def calculate_corr(self, samples, data_y):

        """Calculate the Correlations Betwen X and Y"""
        try:
            cor = abs(np.corrcoef(samples, data_y)[0, 1])
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            print(samples)
            print(data_y)
            cor = 0
        return cor


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
        best_corr = 0
        best_feature_index = None
        n_samples, features =  data_x.shape


        if n_samples == 0:
            return None, None

        if len(np.unique(data_y)) == 1 or n_samples == 1:
            return {
                'feature_index': None,
                'threshold': None
            }
        # if all values in the target y is the same return None

        for feature_index in range(features):

            if len(np.unique(data_x[:, feature_index])) <= 1:
                continue
            corr = self.calculate_corr(data_x[:, feature_index], data_y)
            if corr > best_corr:
                best_corr = corr
                best_feature_index = feature_index


        split_val = np.median(data_x[:, best_feature_index])
        left_mask = data_x[:, best_feature_index] <= split_val
        right_mask = data_x[:, best_feature_index] > split_val

        if data_y[left_mask].size == 0 or data_y[right_mask].size == 0:
            return {
                'feature_index': -1,
                'threshold': np.mean(data_y)
            }

        return {
            'feature_index': best_feature_index,
            'threshold': split_val
        }


    def build_tree(self, data_x, data_y):


        """
        define numpy arrary for node

        self.feature_index = feature_index
        self.threshold = threshold
        self.right = right
        self.left = left
        self.correlation = correlation
        self.prediction = prediction
        self.depth = depth
        self.sample_size = sample_size
        """
        sample_size = data_x.shape[0]


        if data_x.shape[0] == 1 or data_x.shape[0] <= self.leaf_size or len(np.unique(data_y)) == 1:
            return np.array([[
                              -1, # feature index
                              -1, # treshold
                              -1, # left child
                              -1, # right child
                              np.mean(data_y), # prediction
                              ]]
                             )


        optimal_movements = self.find_best_split(data_x, data_y)

        if optimal_movements['feature_index'] == -1:
            return np.array([[-1, -1, -1, -1, optimal_movements['threshold']]]) # threshold as pred

        left_x, right_x, left_y, right_y = self.split(data_x, data_y,
            optimal_movements['feature_index'], optimal_movements['threshold'])

        # Recursively build subtrees
        left_subtree = self.build_tree(left_x, left_y)
        right_subtree = self.build_tree(right_x, right_y)

        featuer_index = optimal_movements['feature_index']
        threshold = optimal_movements['threshold']
        root = np.array([[
            featuer_index,  # feature index
            threshold,  # threshold
            1,
            left_subtree.shape[0] + 1,  # right child index relative offset from the root
            np.mean(data_y),  # prediction
        ]])

        tree = np.vstack((root, left_subtree, right_subtree))


        if self.verbose:
            print(f"Feature Index: {optimal_movements['feature_index']}, "
                  f"Threshold: {optimal_movements['threshold']}, "
                  f"Subtree size: {tree.shape[0]}, Sample Size: {sample_size},"
                  f"Leaf : {'Yes' if optimal_movements['feature_index'] == -1 else 'No'}"
                  )


        return tree

    def add_evidence(self, data_x, data_y):
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        Add training data to learner  		  	   		 	 	 			  		 			 	 	 		 		 	  		  	   		 	 	 			  		 			 	 	 		 		 	
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	 	 			  		 			 	 	 		 		 	
        :type data_y: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
        """
        # slap on 1s column so linear regression finds a constant term
        self.tree_train = self.build_tree(data_x, data_y)


    def traverse_numpy_recursively(self, point : np.ndarray, node_idx : int):
        curr_node = self.tree_train[node_idx]
        feature_idx = int(curr_node[0])

        # If  leaf node  return prediction
        if feature_idx == -1:
            return curr_node[4]

        # Otherwise, test the feature against threshold
        threshold = curr_node[1]
        if point[feature_idx] <= threshold:
            # Go to left child
            return self.traverse_numpy_recursively(point, node_idx + int(curr_node[2]))

        else:
            # Go to right child
            return self.traverse_numpy_recursively(point, node_idx + int(curr_node[3]))


    def query(self, points):
        """  		  	   		 	 	 			  		 			 	 	 		 		 	
        Estimate a set of test points given the model we built. 			  		 			 	 	 		 		 	
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	 	 			  		 			 	 	 		 		 	
        :type points: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
        :return: The predicted result of the input data according to the trained model  		  	   		 	 	 			  		 			 	 	 		 		 	
        :rtype: numpy.ndarray  		  	   		 	 	 			  		 			 	 	 		 		 	
        """
        if not hasattr(self, 'tree_train'):
            raise ValueError('Model Not Trained')

        # For each point in the test set

        return np.array([self.traverse_numpy_recursively(point, 0) for point in points])


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
    train_mean = np.mean(train_y)



    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # create a learner and train it
    dt_learner = DTLearner(verbose=True, leaf_size=10)  # create a DTLearner
    dt_learner.add_evidence(train_x, train_y)
    pred_train = dt_learner.query(train_x)

    rmse = math.sqrt(((train_y - pred_train) ** 2).sum() / train_y.shape[0])
    print(f"DT Training Accuracy: {rmse}")
    print(f"Normalized Training Accuracy: {rmse / train_mean}")


    pred_test = dt_learner.query(test_x)
    rmse_test = math.sqrt(((test_y - pred_test) ** 2).sum() / test_y.shape[0])
    print(f"DT Testing Accuracy: {rmse_test}")



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


