""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
  		  	   		 	 	 			  		 			 	 	 		 		 	
import math  		  	   		 	 	 			  		 			 	 	 		 		 	
import sys
import datetime
import random
import time
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
import DTLearner as dt
import RTLearner as rt
import matplotlib.pyplot as plt

  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":
    # file = 'Data/Istanbul.csv'
    # path = sys.argv[1]
    # inf = open(file)
    # if len(sys.argv) != 2:
    #     print("Usage: python testlearner.py <filename>")
    #     sys.exit(1)

    # inf = open(sys.argv[1])
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )

    # inf = open(sys.argv[1])

    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = []
    lines = inf.readlines()
    valid_cols = []
    first_lines = [lines[1]]

    # identify features which cannot be turned to float
    for line in first_lines:
        values = line.strip().split(',')
        for index, value in enumerate(values):
            try:
                float(value)
                valid_cols.append(index)
            except:
                pass

    # skip first header
    for line in lines[1:]:
        val = line.strip().split(",")
        valid_vals = []
        for col in valid_cols:
            valid_vals.append(float(val[col]))
        data.append(valid_vals)
    data = np.array(data)

    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # compute how much of the data is training and testing

    np.random.seed(900897987)
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows
    train_idx = np.random.choice(data.shape[0], size=train_rows, replace=False)


    # test idx
    mask = np.ones(data.shape[0], dtype=bool)
    mask[train_idx] = False
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # separate out training and testing data  		  	   		 	 	 			  		 			 	 	 		 		 	
    train_x = data[train_idx, 0:-1]
    train_y = data[train_idx, -1]
    test_x = data[mask, 0:-1]
    test_y = data[mask, -1]
  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"{test_x.shape}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"{test_y.shape}")  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # create a learner and train it  		  	   		 	 	 			  		 			 	 	 		 		 	
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner = dt.DTLearner(leaf_size=5, verbose=True) # create a decisiontree
    # learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {'leaf_size' : 5}, bags = 50, verbose = False)
    learner.add_evidence(train_x, train_y)  # train it  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(learner.author())  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # evaluate in sample  		  	   		 	 	 			  		 			 	 	 		 		 	
    pred_y = learner.query(train_x)  # get the predictions  		  	   		 	 	 			  		 			 	 	 		 		 	
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		 	 	 			  		 			 	 	 		 		 	
    print()  		  	   		 	 	 			  		 			 	 	 		 		 	
    print("In sample results")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"RMSE: {rmse}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    c = np.corrcoef(pred_y, y=train_y)  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"corr: {c[0,1]}")  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # evaluate out of sample  		  	   		 	 	 			  		 			 	 	 		 		 	
    pred_y = learner.query(test_x)  # get the predictions  		  	   		 	 	 			  		 			 	 	 		 		 	
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	 	 			  		 			 	 	 		 		 	
    print()  		  	   		 	 	 			  		 			 	 	 		 		 	
    print("Out of sample results")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"RMSE: {rmse}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    c = np.corrcoef(pred_y, y=test_y)  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"corr: {c[0,1]}")





    leaf_sizes = range(1, 101)

    train_rmses = []
    test_rmses = []

    for leaf_size in leaf_sizes:
        import matplotlib.pyplot as plt

        # Instantiate and train the DTLearner
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)

        # Calculate Training RMSE
        pred_y_train = learner.query(train_x)
        train_rmse = math.sqrt(((train_y - pred_y_train) ** 2).mean())
        train_rmses.append(train_rmse)

        # Calculate Testing RMSE
        pred_y_test = learner.query(test_x)
        test_rmse = math.sqrt(((test_y - pred_y_test) ** 2).mean())
        test_rmses.append(test_rmse)

    # 5. Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, train_rmses, label="Training RMSE", color="blue")
    plt.plot(leaf_sizes, test_rmses, label="Testing RMSE", color="red")
    plt.xticks(np.arange(0, 101, 10))
    plt.title("RMSE vs. Leaf Size")
    plt.xlabel("Leaf Size")
    plt.ylabel("Root Mean Squared Error")
    plt.legend()
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('Experiment1.png')
    plt.show()


    leaf_sizes = range(1, 101)

    train_rmses = []
    test_rmses = []

    for leaf_size in leaf_sizes:
        import matplotlib.pyplot as plt

        # Instantiate and train the BagLearner
        learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {'leaf_size' : leaf_size}, bags = 50, verbose = False)
        learner.add_evidence(train_x, train_y)

        # Calculate Training RMSE
        pred_y_train = learner.query(train_x)
        train_rmse = math.sqrt(((train_y - pred_y_train) ** 2).mean())
        train_rmses.append(train_rmse)

        # Calculate Testing RMSE
        pred_y_test = learner.query(test_x)
        test_rmse = math.sqrt(((test_y - pred_y_test) ** 2).mean())
        test_rmses.append(test_rmse)


    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, train_rmses, label="Training RMSE", color="blue")
    plt.plot(leaf_sizes, test_rmses, label="Testing RMSE", color="red")
    plt.xticks(np.arange(0, 101, 10))
    plt.title("Bagging RMSE vs. Leaf Size")
    plt.xlabel("Leaf Size")
    plt.ylabel("Root Mean Squared Error")
    plt.legend()
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('Experiment2.png')


    rt_r_squared = []
    dt_r_squared = []
    leaf_sizes = range(1, 101)


    def compute_r_squared(y_actual, y_predicted):
        # Calculate rss
        rss = np.sum((y_actual - y_predicted) ** 2)

        # Calculate the tss
        tss = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r_squared = 1 - (rss / tss)

        return r_squared


    for i in range(1, 101):
        dt_learner = dt.DTLearner(leaf_size=i, verbose=False)
        dt_learner.add_evidence(train_x, train_y)
        y_test_pred = dt_learner.query(test_x)
        r_squared = compute_r_squared(test_y, y_test_pred)
        dt_r_squared.append(r_squared)


        rt_learner = rt.RTLearner(leaf_size=i, verbose=False)
        rt_learner.add_evidence(train_x, train_y)
        y_test_pred_rt = rt_learner.query(test_x)
        r_squared_rt = compute_r_squared(test_y, y_test_pred_rt)
        rt_r_squared.append(r_squared_rt)


    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, rt_r_squared, label="Random Decision Tree Depth", color="blue")
    plt.plot(leaf_sizes, dt_r_squared, label="Decision Tree Depth", color="red")
    plt.xticks(np.arange(0, 101, 10))
    plt.title("Random Decision Tree Depth vs Decision R-Squared")
    plt.xlabel("Leaf Size")
    plt.ylabel("R-Squared")
    plt.legend()
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('Experiment3.png')




# training time

    rt_training = []
    dt_training = []
    leaf_sizes = range(1, 101)

    for i in leaf_sizes:
        dt_learner = dt.DTLearner(leaf_size=i, verbose=False)

        start = time.time()
        dt_learner.add_evidence(train_x, train_y)
        end = time.time()
        dt_duration = end - start
        dt_training.append(dt_duration)


        rt_learner = rt.RTLearner(leaf_size=i, verbose=False)
        start = time.time()
        rt_learner.add_evidence(train_x, train_y)
        end = time.time()
        rt_duration = end - start
        rt_training.append(rt_duration)

    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, rt_training, label="Random Decision Training Time", color="blue")
    plt.plot(leaf_sizes, dt_training, label="Decision Tree Training Time", color="red")
    plt.xticks(np.arange(0, 101, 10))
    plt.title("Random Decision Tree Training Time vs Decision Tree")
    plt.xlabel("Leaf Size")
    plt.ylabel("Training Time (in seconds)")
    plt.legend()
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('Experiment5.png')





