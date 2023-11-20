# Reinforcement Learning Project 

Name:         André Kestler  
Date:         17.11.2023  
Course:       Master KI  
Lecture:      Deep Reinforcement Learning  
Professor:    Prof. Dr.-Ing. Thomas Nierhoff  

Description:  
    This file contains the implementation of the Monte Carlo Evaluation, SARSA and Q-Learning algorithms.  
    The algorithms are used to solve the gridworld environment.

Task 1  
With as few episodes as possible  
- (environment Random(size=12, water=0.3, mountain=0.0))  
- Plot your results as follows:  
    As real cumulative reward of the current episode (y-axis) over episodes (x-axis), averaged over ten different environments  

Task 2  
With as few steps as possible  
- (environment Random(size=12, water=0.0, mountain=0.3))  
- Plot your results as follows:  
    As real cumulative reward of all episodes so far (y-axis) over number of steps (x-axis), averaged over ten different environments
<br><br>
--------------------------------------------------------------------------------  
### Folder structure (TASK1)  
.  
├── plot                     &emsp;# The figures of the tasks (plot as png)  
&emsp;├── plot                    &emsp;# Plot for task 1  
&emsp;&emsp;├── mc_evaluation          &emsp;# cumulative reward plot and example q_table   for mc evaluation  
&emsp;&emsp;├── qlearning              &emsp;# cumulative reward plot and example q_table   for qlearning  
&emsp;&emsp;└── sarsa                  &emsp;# cumulative reward plot and example q_table for sarsa  
&emsp;├── plot                    &emsp;# Plot for task 2  
&emsp;&emsp;├── mc_evaluation          &emsp;# cumulative reward plot and example q_table for mc evaluation  
&emsp;&emsp;├── qlearning              &emsp;# cumulative reward plot and example q_table for qlearning  
&emsp;&emsp;└── sarsa                  &emsp;# cumulative reward plot and example q_table for sarsa  
├── src                      &emsp;# Python code  
&emsp;├── Task1_Optuna_Result     &emsp;# DataFrames (csv) for the results of Optuna optimizing Task 1  
&emsp;├── Task2_Optuna_Result     &emsp;# DataFrames (csv) for the results of Optuna optimizing Task 2  
&emsp;├── config.py               &emsp;# Config file  
&emsp;├── gridworld.py            &emsp;# Gridworld (Environment) file  
&emsp;├── optimize_task_1.ipynb   &emsp;# Optimize hyperparameters for task 1  
&emsp;├── optimize_task_2.ipynb   &emsp;# Optimize hyperparameters for task 2  
&emsp;├── plot.py                 &emsp;# Plot file for Q-Table  
&emsp;├── task_1.py               &emsp;# Implemented Algorithm for task 1  
&emsp;└── task_2.py               &emsp;# Implemented Algorithm for task 2  
├── presentation.pptx        &emsp;# Presentation file in pptx format  
├── presentation.pdf         &emsp;# Presentation file in pdf format  
├── readme.txt               &emsp;# Readme file  
├── requirements.txt         &emsp;# Requirements file for pip packages  
└── task1.pdf                &emsp;# Task descripion  
<br><br>
--------------------------------------------------------------------------------
### Task definition:
Solve the gridworld environment Random in file gridworld.py
1. with as few episodes as possible
  (environment Random(size=12, water=0.3, mountain=0.0))
2. with as few steps as possible
  (environment Random(size=12, water=0.0, mountain=0.3))

- Plot your results as follows:
    - for 1. as real cumulative reward of the current episode (y-axis) over episodes (x-axis), averaged over ten different environments
    - for 2. as real cumulative reward of all episodes so far (y-axis) over number of steps (x-axis), averaged over ten different environments
<br><br>
--------------------------------------------------------------------------------  
### Version

- Python:      3.10.8  
- numpy:       1.20.3  
- matplotlib:  3.4.3  
<br><br>
--------------------------------------------------------------------------------
### Install packages  
- Install Python  
- Go to the folder TASK1   
- In console: pip install -r requirements.txt  
<br><br>
--------------------------------------------------------------------------------
### Start script  
  
#### Task 1:  
  Setup parameters (beginning of the script task_1.py)  
- method: The method to use (monteCarlo_evaluation, sarsa, qLearning)  
    <br>
- MAX_EPISODES: Maximum number of episodes  
- MAX_STEPS: Maximum number of steps per episode  
- EPSILON: The epsilon value for the epsilon-greedy policy  
- GAMMA: The discount factor  
- ALPHA: The learning rate  
- INITIALIZATION: The initial value for the Q-table  
- DECAY_RATE: The decay rate for the epsilon value  
- DECAY_INTERVAL: The decay interval for the epsilon value  
<br><br>

- cd TASK1/            &emsp;# Go to folder TASK1/  
- cd src/              &emsp;# Go to folder src/ in TASK1/  
- python3 task_1.py    &emsp;# Start the algorithm with the specified hyperparameters   and method to get results


#### Task 2:
  Setup parameters (beginning of the script task_2.py)  
- method: The method to use (monteCarlo_evaluation, sarsa, qLearning)

- MAX_EPISODES: Maximum number of episodes
- MAX_STEPS: Maximum number of steps per episode
- EPSILON: The epsilon value for the epsilon-greedy policy
- GAMMA: The discount factor
- ALPHA: The learning rate
- INITIALIZATION: The initial value for the Q-table
- DECAY_RATE: The decay rate for the epsilon value
- DECAY_INTERVAL: The decay interval for the epsilon value

<br><br>
- cd TASK1/            &emsp;#  Go to folder TASK1/
- cd src/              &emsp;# Go to folder src/ in TASK1/
- python3 task_2.py    &emsp;# Start the algorithm with the specified hyperparameters and method to get results


--------------------------------------------------------------------------------
### Evaluation

#### Implemented
  - epsilon-greedy policy
  - thomson-sampling (not used due to no effort)
  - epsilon-decay over time t
  - decay epsilon just every nth episode
  - optimistic initialization


#### Task 1:  
  - Optimized with Optuna package (optimize_task_1.ipynb)
  - With Optuna's best parameters, the agent did not manage to reach the goal in all 10 runs
  - I manually tweaked the MAX_EPISODES parameter in a way that
      1. Took the best_params from Optuna
         ---> Agent did not reach the goal state in every run
      2. Increased MAX_EPISODES to a high value (around 500-800)
      3. Looked at the cumulative_plot (cumulative reward of all episodes so far over episodes)
      4. The plot converged arount some EPISODES to a cumulative reward around 100
      5. Took this episode as a new MAX_EPISODE
  - Run the task_1.py with the specific parameter for each method (monteCarlo_evaluation, sarsa, qlearning)
  - Save the cumulative_plot and a example Q-Table for each method
  --> Cumulative-Plot and Q-Table with Policy Plot is in the respective TASK1/task_1/ folder or in the presentation.pptx/presentation.pdf


#### Task 2:  
  - Optimized with Optuna package (optimize_task_2.ipynb)
  - I have changed the MAX_EPISODES parameter from every method down to (mc: 50, sarsa: 80, qlearning: 50) because the agent can also reach the goal in every run with a lower episode
  - Run the task_2.py with the specific parameter for each method (monteCarlo_evaluation, sarsa, qlearning)
  - Save the plot (cumulative reward over steps) and a example Q-Table for each method
  --> Cumulative-Plot and Q-Table with Policy Plot is in the respective TASK1/task_2/ folder or in the presentation.pptx/presentation.pdf