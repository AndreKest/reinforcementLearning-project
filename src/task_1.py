"""
Author:     Andre Kestler
Date:       2023-11-17

Description:
    This file contains the implementation of the Monte Carlo Evaluation, SARSA and Q-Learning algorithms.
    The algorithms are used to solve the gridworld environment.

With as few episodes as possible
  (environment Random(size=12, water=0.3, mountain=0.0))

Plot your results as follows:
    As real cumulative reward of the current episode (y-axis) over episodes (x-axis), averaged over ten different environments


Functions:
    generate_episode(env, Q, max_steps, epsilon): Generate an episode following the epsilon-greedy policy
    select_action(env, Q, obs, epsilon): Select an action following the epsilon-greedy policy
    extract_policy(env, Q): Extract the policy from the Q-table
    epsilon_decay(episode, decay_rate): Decay epsilon every episode
    check_goal_reachable_one_policy(env, policy): Check if the goal state is reachable with the given policy
    check_goal_reachable_all_policies(env, lst_policies, title): Check if the goal state is reachable for all given policy
    
    plot_cummulative_reward(lst_reward, title): Plot the cummulative reward
    plot_epsilon_decay(max_episodes, decay_rate): Plot the epsilon decay
    
    monteCarlo_evaluation(env, max_episodes, max_steps, gamma, alpha, epsilon, initialisation, decay_rate, decay_interval): Monte Carlo Evaluation Algorithm
    sarsa(env, max_episodes, max_steps, gamma, alpha, epsilon, initialisation, decay_rate, decay_interval): SARSA Algorithm
    qLearning(env, max_episodes, max_steps, gamma, alpha, epsilon, initialisation, decay_rate, decay_interval): Q-Learning Algorithm

    
Parameters:
    method: The method to use (monteCarlo_evaluation, sarsa, qLearning)

    MAX_EPISODES: Maximum number of episodes
    MAX_STEPS: Maximum number of steps per episode
    EPSILON: The epsilon value for the epsilon-greedy policy
    GAMMA: The discount factor
    ALPHA: The learning rate
    INITIALIZATION: The initial value for the Q-table
    DECAY_RATE: The decay rate for the epsilon value
    DECAY_INTERVAL: The decay interval for the epsilon value

    
Best parameters (with Package Optuna - Hyperparameter Optimization (see: optimize_task_1.ipynb)):
    - Monte Carlo Evaluation:
    Optuna got me MAX_EPISODES = 100, but then the agent does not reaches the goal state in every run.
    So I increased the MAX_EPISODES to 250 but due to alot of noise I decreased alpha to 0.1  and now the agent reaches the goal state in every run.
    To find this out I increased the MAX_EPISODES to a high value (1000) then I saw that the plot converges at around 250 episodes. So I took 250 as MAX_EPISODES.
        MAX_EPISODES = 250
        MAX_STEPS = 1000
        EPSILON = 0.8
        GAMMA = 0.8
        ALPHA = 0.1
        INITIALIZATION = 47
        DECAY_RATE = 0.25
        DECAY_INTERVAL = 30
    
    - SARSA:
    Optuna got me MAX_EPISODES = 100, but then the agent does not reaches the goal state in every run.
    So I increased the MAX_EPISODES to 130 and now the agent reaches the goal state in every run.
    To find this out I increased the MAX_EPISODES to a high value (500) then I saw that the plot converges at around 130 episodes. So I took 130 as MAX_EPISODES.
        MAX_EPISODES = 130
        MAX_STEPS = 1000
        EPSILON = 0.8
        GAMMA = 0.7000000000000001
        ALPHA = 0.7000000000000001
        INITIALIZATION = 32
        DECAY_RATE = 2.25
        DECAY_INTERVAL = 61

    - Q-Learning:
    Optuna got me MAX_EPISODES = 100, but then the agent does not reaches the goal state in every run. 
    So I tried the second best parameter with MAX_EPISODES=103 and it worked. 
    The problem was that the agent does not reached the goal state in every run. So I increased the MAX_EPISODES to 120 and now the agent reaches the goal state in every run.
    To find this out I increased the MAX_EPISODES to a high value (500) then I saw that the plot converges at around 120 episodes. So I took 120 as MAX_EPISODES.
        MAX_EPISODES = 120
        MAX_STEPS = 1000
        GAMMA = 0.6
        ALPHA = 0.5
        EPSILON = 0.8
        INITIALIZATION = 16
        DECAY_RATE = 2.0
        DECAY_INTERVAL= 41


"""
from gridworld import *
from plot import *

import random

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------
MAX_RUNS = 10

# Random seed for reproducibility
random.seed(8)
np.random.seed(8)


# --------------------------------------------------------------------------------
# Choose method
method = "qLearning" # monteCarlo_evaluation, sarsa, qLearning

# Parameters
# MAX_EPISODES = 250
# MAX_STEPS = 1000
# EPSILON = 0.8
# GAMMA = 0.8
# ALPHA = 0.1
# INITIALIZATION = 47
# DECAY_RATE = 0.25
# DECAY_INTERVAL = 30

# MAX_EPISODES = 130
# MAX_STEPS = 1000
# EPSILON = 0.8
# GAMMA = 0.7000000000000001
# ALPHA = 0.7000000000000001
# INITIALIZATION = 32
# DECAY_RATE = 2.25
# DECAY_INTERVAL = 61

MAX_EPISODES = 120
MAX_STEPS = 1000
GAMMA = 0.6
ALPHA = 0.5
EPSILON = 0.8
INITIALIZATION = 16
DECAY_RATE = 2.0
DECAY_INTERVAL= 41

# --------------------------------------------------------------------------------
def generate_episode(env, Q, max_steps=100, epsilon=0.1):
    """ Generate an episode following the epsilon-greedy policy
    Args:
        env: the environment
        Q: the Q-table
        max_steps: maximum number of steps per episode
        epsilon: the epsilon value for the epsilon-greedy policy
    Returns:
        episodes: the generated episode
    """
    episodes = []
    obs = env.reset()

    # Select an action following the epsilon-greedy policy
    action = select_action(env, Q, obs, epsilon)

    # Create an episode with at most max_steps
    for step in range(max_steps):
        # Execute the action and observe the next state and reward
        next_obs, reward, done = env.step(action)
        # Select the next action following the epsilon-greedy policy
        next_action = select_action(env, Q, next_obs, epsilon)

        # Add the observed transition to the episode
        episodes.append((obs, action, reward, next_obs, next_action, done))

        # Update the current state and action
        obs = next_obs
        action = next_action

        # If the episode is done, stop the episode
        if done == True:
            break

    return episodes

def select_action(env, Q, obs, epsilon):
    """ Select an action following the epsilon-greedy policy
    Args:
        env: the environment
        Q: the Q-table
        obs: the current observation
        epsilon: the epsilon value for the epsilon-greedy policy
    Returns:
        the selected action
    """
    # Epsilon-greedy policy
    if random.random() < epsilon:
        return random.randint(0, env.num_actions()-1)
    else:
        return np.argmax(Q[obs])
    
def thomson_sampling_action(Q, pos, dist="beta"):
    """ Select an action following the thomson sampling policy
    Args:
        Q: the Q-table (num_states, num_actions)
        pos: the current position
        dist: the distribution to use for sampling
    Returns:
        the selected action
    """
    # Thomson sampling policy
    if dist == "beta":
        # Sample from beta distribution
        return np.argmax(np.random.beta(Q[pos], 1-Q[pos]))
    elif dist == "normal":
        # Sample from normal distribution
        return np.argmax(np.random.normal(Q[pos], 1-Q[pos]))
    elif dist == "uniform":
        # Sample from uniform distribution
        return np.argmax(np.random.uniform(Q[pos], 1-Q[pos]))
    else:
        return np.argmax(np.random.normal(Q[pos], 1))

def extract_policy(env, Q):
    """ Extract the policy from the Q-table
    Args:
        env: the environment
        Q: the Q-table
    Returns:
        the extracted policy
    """
    # Initialize policy with zero probability for each action
    policy = np.zeros((env.num_states(), env.num_actions()))

    # For each observation, select the action with the highest Q-value
    for obs in range(env.num_states()):
        policy[obs][np.argmax(Q[obs])] = 1

    return policy


# Epsilon schedules
def epsilon_decay(episode, decay_rate=0.5):
    """ Decay epsilon every episode
    Args:
        episode: the current episode
        max_episodes: the maximum number of episodes
        epsilon: the initial epsilon value
    Returns:
        the epsilon value for the current episode
    """
    return 1 / (episode + 1)**decay_rate
    

# Check Goal State
def check_goal_reachable_one_policy(env, policy):
    """ Check if the goal state is reachable with the given policy 
    Print "Goal state is reachable" if the goal state is reachable
    Print "Goal state is not reachable" if the goal state is not reachable

    Args:
        env: the environment
        policy: the policy
    Returns:
        True if the goal state is reachable
        False if the goal state is not reachable
    """
    obs = env.reset()
    # Do 1000 steps with the given policy
    for step in range(1000):
        # Take the action with the highest probability
        action = np.argmax(policy[obs])
        # action = np.random.choice(env.num_actions(), p=policy[obs])
        # Execute the action and observe the next state and reward
        next_obs, reward, done = env.step(action)
        # Update the current state
        obs = next_obs

        # If the goal state is reached, return True
        if done == True:
            return True
        
    # If the goal state is not reached after 1000 steps, return False
    return False

def check_goal_reachable_all_policies(env, lst_policies, title=""):
    """ Check if the goal state is reachable for all given policy 
    Print "Goal state is reachable" if the goal state is reachable
    Print "Goal state is not reachable" if the goal state is not reachable

    Args:
        env: the environment
        lst_policies: the list of policies
        title: the name of the algorithm
    Returns:
        list of indices of policies that is reachable
    """
    lst_reachable = []
    for policy in lst_policies:
        lst_reachable.append(check_goal_reachable_one_policy(env, policy))

    # save index of policies that is reachable
    lst_reachable_index = [i for i, x in enumerate(lst_reachable) if x == True]
    if len(lst_reachable_index) == 0:
        lst_reachable_index = [0]
        print(title + ": Goal state is not reachable with policies" + "\n")
        return lst_reachable_index[0]
    else:
        print(title + ": Goal state is reachable with " + str(lst_reachable_index) + " policies" + "\n")
        return lst_reachable_index[0]


# Plot functions
def plot_cummulative_reward(lst_reward, title=""):
    """ Plot the cumulative reward """
    # Calculate the mean of the rewards over all runs
    lst_reward = np.mean(lst_reward, axis=0)

    # Plot the cummulative reward
    plt.plot(range(len(lst_reward)), lst_reward)
    plt.title("Cumulative reward for " + title)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward")
    plt.show()


def plot_epsilon_decay(max_episodes=1000, decay_rate=0.5):
    """ Plot the epsilon decay """
    lst_epsilon = []
    for episode in range(max_episodes):
        lst_epsilon.append(epsilon_decay(episode, decay_rate=decay_rate))

    # Plot the epsilon decay
    plt.plot(range(len(lst_epsilon)), lst_epsilon)
    plt.title("Epsilon decay")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.show()

# --------------------------------------------------------------------------------
# Monte Carlo Evaluation
def monteCarlo_evaluation(env, max_episodes=1000, max_steps=100, gamma=0.8, alpha=0.1, epsilon=0.1, initialisation=0, decay_rate=0.5, decay_interval=100):
    """ Monte Carlo Evaluation Algorithm
    Args:
        env: the environment
        max_episodes: maximum number of episodes
        max_steps: maximum number of steps per episode
        gamma: the discount factor
        alpha: the learning rate
        epsilon: the epsilon value for the epsilon-greedy policy
        initialisation: the initial value for the Q-table
        decay_rate: the decay rate for the epsilon value

    Returns:
        Q: the Q-table
        policy: the policy
        lst_cummulative_reward: the cummulative reward for each episode
    """
    # Initialize q-values
    Q = np.ones((env.num_states(), env.num_actions())) * initialisation

    lst_cummulative_reward = []

    # Do max_episodes episodes
    for num_episode in range(max_episodes):
        # Decay epsilon every x episodes
        if num_episode % decay_interval == 0:
            epsilon = epsilon_decay(num_episode, decay_rate=decay_rate)

        # Create episode
        episode = generate_episode(env, Q, max_steps=max_steps, epsilon=epsilon)

        cummulative_reward = 0

        # Calculate G for each step in the episode and update Q (start at the end of the episode)
        G = 0
        for step in reversed(range(len(episode))):
            obs, action, reward, next_obs, next_action, done = episode[step]

            cummulative_reward += reward
            G = gamma * G + reward
            Q[obs][action] += alpha * (G - Q[obs][action])


        
        lst_cummulative_reward.append(cummulative_reward)
    
    # Extract policy from Q-table
    policy = extract_policy(env, Q)

    return Q, policy, lst_cummulative_reward


# --------------------------------------------------------------------------------
# SARSA (State-Action-Reward-State-Action)
def sarsa(env, max_episodes=1000, max_steps=100, gamma=0.8, alpha=0.1, epsilon=0.1, initialisation=0, decay_rate=0.5, decay_interval=100):
    """ SARSA Algorithm
    Args:
        env: the environment
        max_episodes: maximum number of episodes
        max_steps: maximum number of steps per episode
        gamma: the discount factor
        alpha: the learning rate
        epsilon: the epsilon value for the epsilon-greedy policy
        initialisation: the initial value for the Q-table
        decay_rate: the decay rate for the epsilon value

    Returns:
        Q: the Q-table
        policy: the policy
        lst_cummulative_reward: the cummulative reward for each episode
    """
    # Initialize q-values
    Q = np.ones((env.num_states(), env.num_actions())) * initialisation

    lst_cummulative_reward = []

    # Do max_episodes episodes
    for num_episode in range(max_episodes):
        obs = env.reset()

        cummulative_reward = 0
        
        # Select an action following the epsilon-greedy policy
        action = select_action(env, Q, obs, epsilon)

        # Decay epsilon every x episodes
        if num_episode % decay_interval == 0:
            epsilon = epsilon_decay(num_episode, decay_rate=decay_rate)
        
        # Do max_steps steps per episode
        for step in range(max_steps):
            # Execute the action and observe the next state and reward
            next_obs, reward, done = env.step(action)

            # Select the next action following the epsilon-greedy policy
            next_action = select_action(env, Q, next_obs, epsilon)

            cummulative_reward += reward

            # Update Q
            if done == True:
                Q[obs][action] += alpha * (reward - Q[obs][action])
                break
            else:
                Q[obs][action] += alpha * (reward + gamma * Q[next_obs][next_action] - Q[obs][action])
            
            # Update the current state and action
            obs = next_obs
            action = next_action
        
        lst_cummulative_reward.append(cummulative_reward)

        # Every 500 episodes plot the Q-table and the policy - Debugging purposes
        # if episode % 500 == 0:
        #     print("Episode: ", episode)
        #     # plot_q_table(env, Q)

    # Extract policy from Q-table
    policy = extract_policy(env, Q)
    
    return Q, policy, lst_cummulative_reward


# --------------------------------------------------------------------------------
# Q-Learning
def qLearning(env, max_episodes=1000, max_steps=100, gamma=0.8, alpha=0.5, epsilon=0.1, initialisation=0, decay_rate=0.5, decay_interval=100):
    """ Q-Learning Algorithm
    Args:
        env: the environment
        max_episodes: maximum number of episodes
        max_steps: maximum number of steps per episode
        gamma: the discount factor
        alpha: the learning rate
        epsilon: the epsilon value for the epsilon-greedy policy
        initialisation: the initial value for the Q-table
        decay_rate: the decay rate for the epsilon value
    
    Returns:
        Q: the Q-table
        policy: the policy
        lst_cummulative_reward: the cummulative reward for each episode
    """
    # Initialize q-values
    Q = np.ones((env.num_states(), env.num_actions())) * initialisation
    
    lst_cummulative_reward = []

    # Do max_episodes episodes
    for num_episode in range(max_episodes):
        obs = env.reset()
        
        cummulative_reward = 0

        # Select an action following the epsilon-greedy policy
        action = select_action(env, Q, obs, epsilon)

        # Decay epsilon every x episodes
        if num_episode % decay_interval == 0:
            epsilon = epsilon_decay(num_episode, decay_rate=decay_rate)

        # Do max_steps steps per episode
        for step in range(max_steps):
            # Execute the action and observe the next state and reward
            next_obs, reward, done = env.step(action)
            # Select the next action following the epsilon-greedy policy
            next_action = select_action(env, Q, next_obs, epsilon)

            cummulative_reward += reward

            # Update Q
            if done == True:
                Q[obs][action] += alpha * (reward - Q[obs][action])
                break
            else:
                Q[obs][action] += alpha * (reward + gamma * max(Q[next_obs]) - Q[obs][action])
            
            # Update the current state and action
            obs = next_obs
            action = next_action
        
        lst_cummulative_reward.append(cummulative_reward)

        # Every 100 episodes plot the Q-table and the policy - Debugging purposes
        # if episode % 500 == 0:
        #     print("Episode: ", episode)
        #     # plot_q_table(env, Q)
    
    # Extract policy from Q-table
    policy = extract_policy(env, Q)
    
    return Q, policy, lst_cummulative_reward


# --------------------------------------------------------------------------------
    
if __name__ == "__main__":
    # create environment
    # Solve environment with as few episodes as possible
    env = Random(size=12, water=0.3, mountain=0.0)

    # plot_epsilon_decay(max_episodes=MAX_EPISODES, decay_rate=DECAY_RATE)
    
    # --------------------------------------------------------------------------------
    if method == "monteCarlo_evaluation":
        # Iterative Monte Carlo Evaluation
        lst_Q_table = []
        lst_policy = []
        lst_reward = []
        for i in range(MAX_RUNS):
            Q_table, policy, reward = monteCarlo_evaluation(env, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, initialisation=INITIALIZATION, decay_rate=DECAY_RATE, decay_interval=DECAY_INTERVAL)
            lst_Q_table.append(Q_table)
            lst_policy.append(policy)
            lst_reward.append(reward)

        
        # Check if the goal state is reachable with the every policy
        index = check_goal_reachable_all_policies(env, lst_policy, "Iterative Monte Carlo Evaluation")

        # Plot the cummulative reward
        plot_cummulative_reward(lst_reward, title="Iterative Monte Carlo Evaluation")

        # Example Q-table plot
        plot_q_table(env, lst_Q_table[index], lst_policy[index])


    # --------------------------------------------------------------------------------
    # SARSA
    elif method == "sarsa":
        lst_Q_table = []
        lst_policy = []
        lst_reward = []
        for i in range(MAX_RUNS):
            Q_table, policy, reward = sarsa(env, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, initialisation=INITIALIZATION, decay_rate=DECAY_RATE, decay_interval=DECAY_INTERVAL)
            lst_Q_table.append(Q_table)
            lst_policy.append(policy)
            lst_reward.append(reward)

        # Check if the goal state is reachable with the every policy
        index = check_goal_reachable_all_policies(env, lst_policy, "SARSA")

        # Plot the cummulative reward
        plot_cummulative_reward(lst_reward, title="SARSA")

        # Example Q-table plot
        plot_q_table(env, lst_Q_table[index], lst_policy[index])


    # --------------------------------------------------------------------------------
    # Q-Learning
    elif method == "qLearning":
        lst_Q_table = []
        lst_policy = []
        lst_reward = []
        for i in range(MAX_RUNS):
            Q_table, policy, reward = qLearning(env, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, initialisation=INITIALIZATION, decay_rate=DECAY_RATE, decay_interval=DECAY_INTERVAL)
            lst_Q_table.append(Q_table)
            lst_policy.append(policy)
            lst_reward.append(reward)

        # Check if the goal state is reachable with the every policy
        index = check_goal_reachable_all_policies(env, lst_policy, "Q-Learning")

        # Plot the cummulative reward
        plot_cummulative_reward(lst_reward, title="Q-Learning")

        # Example Q-table plot
        plot_q_table(env, lst_Q_table[index], lst_policy[index])
    else:
        print("Wrong method name!")
        print("Please choose between: monteCarlo_evaluation, sarsa, qLearning!")