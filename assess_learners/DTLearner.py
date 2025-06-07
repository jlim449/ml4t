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
        if not hasattr(self, 'tree_train'):
            raise ValueError('Model Not Trained')

        # For each point in the test set
        return np.array([self.traverse_numpy_recursively(point, 0) for point in points])
