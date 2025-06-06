"""Assess a betting strategy.
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	                                		  		 			 	 	 		 		 	
def author():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return "jlim449"  # replace tb34 with your Georgia Tech username.
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def gtid():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return 900897987  # replace with your GT ID number  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def get_spin_result(win_prob):  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param win_prob: The probability of winning  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type win_prob: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The result of the spin.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    result = False  		  	   		 	 	 			  		 			 	 	 		 		 	
    if np.random.random() <= win_prob:  		  	   		 	 	 			  		 			 	 	 		 		 	
        result = True  		  	   		 	 	 			  		 			 	 	 		 		 	
    return result

def simulate_single_episode(experiment) -> np.ndarray:
    history = np.zeros(1001)
    episode_winning = 0
    bet_amount = 1
    win_prob = 18 / 38

    match experiment:

        #         experiment 1
        case 1:
            for i in range(1, 1001):

                if episode_winning == 80:
                    history[i] = episode_winning
                    continue

                won = get_spin_result(win_prob)
                if won:
                    episode_winning += bet_amount
                #     set bet to 1
                    bet_amount = 1
                else:
                    episode_winning -= bet_amount
                    bet_amount *= 2
                history[i] = episode_winning

        #         experiment 2

        case 2:
            bankroll = 256

            for i in range(1, 1001):

                bet_amount = min(bet_amount, bankroll)

                if episode_winning >= 80 or bankroll <= 0:
                    history[i] = episode_winning if episode_winning > 0 else -256
                    continue

                won = get_spin_result(win_prob)
                if won:
                    # handling special case when bet_amount is more than the epsiode_winning
                    # betting amount should mininum of epdisode_winning and bet_amount
                    bankroll += bet_amount
                    episode_winning += bet_amount
                #    reset bet to 1
                    bet_amount = 1
                else:
                    bankroll -= bet_amount
                    episode_winning -= bet_amount
                    bet_amount *= 2

                history[i] = episode_winning
    return history



def simulate_episode(episodes : int = 1, experiment_no : int = 1) -> np.ndarray:

    simulation = np.array([simulate_single_episode(experiment= experiment_no) for _ in range(episodes)])
    return simulation


def visualize_spins(sim_result : np.ndarray , max_spins : None, filename : str = None) -> None:
    """
    Visualize the spins of the roulette wheel
    """
    max_spins = max_spins if max_spins is not None else 1000
    x = np.arange(1, max_spins)
    plt.figure(figsize=(10, 5))
    for idx, row in enumerate(sim_result):
        plt.plot(x, row[1:max_spins], alpha=0.5, label=f"Episode {idx+1}")
    plt.xlabel("Spin")
    plt.ylabel("Winnings")
    plt.ylim(-256, 100)
    plt.title("Martingale Simulation")
    plt.legend()
    if filename:
       plt.savefig(f'{filename}.png')
    else:
        return None

def visualize_spins_stats(spin_stats : dict, max_spins : None, main_graph : str, filename : str  = None) -> None:
    """
    Visualize the spins of the roulette wheel
    """
    max_spins = max_spins if max_spins is not None else 1000
    x = np.arange(1, max_spins)
    plt.figure(figsize=(10, 5))


    # draw mean line
    mean_numpy = spin_stats.get('mean')
    std_numpy = spin_stats.get('std')
    median_numpy = spin_stats.get('median')

    if main_graph == 'mean':
        plt.plot(x, mean_numpy[1:max_spins], alpha=0.8, label=f"Mean", color = 'blue', linewidth=3)
        plt.plot(x, mean_numpy[1:max_spins] + std_numpy[1:max_spins], alpha=0.5, label=f"Mean + Std", linestyle='--', color = 'red', linewidth=1.5)
        plt.plot(x, mean_numpy[1:max_spins] - std_numpy[1:max_spins], alpha=0.5, label=f"Mean - Std", linestyle='--', color = 'orange', linewidth=1.5)
    elif main_graph == 'median':
        plt.plot(x, median_numpy[1:max_spins], alpha=0.8, label=f"Median", color = 'blue', linewidth=3)
        plt.plot(x, median_numpy[1:max_spins] + std_numpy[1:max_spins], alpha=0.5, label=f"Median + Std", linestyle='--', color = 'red', linewidth=1.5)
        plt.plot(x, median_numpy[1:max_spins] - std_numpy[1:max_spins], alpha=0.5, label=f"Median - Std", linestyle='--', color = 'orange', linewidth=1.5)


    plt.xlabel("Spin No.")
    plt.ylabel("Winnings")
    plt.ylim(-256, 100)
    plt.title("Martingale Simulation")
    plt.legend()
    if filename:
        plt.savefig(f'{filename}.png')
    else:
        return None

def compute_prop(simulation_result):

    total_epsisodes = simulation_result.shape[0]
    final_result = simulation_result[:, -1]
    wins = np.sum(final_result == 80)
    prob = wins / total_epsisodes

    return prob



def expected_value(simulation_result):

    total_epsisodes = simulation_result.shape[0]
    final_result = simulation_result[:, -1]
    unique_elements, counts = np.unique(final_result, return_counts=True)

    # prob = wins / total_epsisodes

    prob = dict(zip(unique_elements, counts / total_epsisodes))
    expected_value = sum(x * p for x, p in prob.items())

    return expected_value


#
# def propotion_value(simulation_result) -> np.ndarray:
#
#     num_episodes, num_spins = simulation_result.shape
#
#     result = []
#
#     for i in range(1, num_spins):  # Skip spin 0 (initial state)
#         values_at_spin = simulation_result[:, i]
#         unique_values, counts = np.unique(values_at_spin, return_counts=True)
#         distribution = dict(zip(unique_values, counts / num_episodes))
#         result.append(distribution)
#     return np.array(result, dtype=object)
#



def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Method to test your code  		  	   		 	 	 			  		 			 	 	 		 		 	
    """
    win_prob = 0.60  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments

    # experiment 1
    result = simulate_episode(episodes=3, experiment_no=1)
    prob = compute_prop(result)
    prob
    spin_mean = np.mean(result, axis = 0)
    spin_std = np.std(result, axis = 0)
    spin_median = np.median(result, axis=0)
    spin_stats = {
        'mean': spin_mean,
        'std': spin_std,
        'median': spin_median
    }
    visualize_spins(sim_result=result, max_spins=300, filename='figure1')
    visualize_spins_stats(spin_stats, max_spins=300, main_graph='mean', filename='figure2')
    visualize_spins_stats(spin_stats, max_spins=300, main_graph='median', filename='figure3')


    prob = compute_prop(result)
    print(prob)



    # experiment 2
    result = simulate_episode(episodes=1000, experiment_no=2)

    prob = compute_prop(result)
    print(prob)

    spin_mean = np.mean(result, axis = 0)
    spin_std = np.std(result, axis = 0)
    spin_median = np.median(result, axis=0)
    spin_stats = {
        'mean': spin_mean,
        'std': spin_std,
        'median': spin_median
    }


    visualize_spins_stats(spin_stats, max_spins=300, main_graph='mean', filename='figure4')
    visualize_spins_stats(spin_stats, max_spins=300, main_graph='median', filename='figure5')


  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
