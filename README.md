# Actor-Critic 

The implementation was adapted from the paper "Addressing Function Approximation Error in Actor-Critic Methods" [1], using specifically a Twin Delayed Deep Deterministic Policy Gradient method. The papers main focus is on minimising errors due to overestimation that stem from function approximation. 

## Clipped Double Q-learning [1]

The algorithm builds on Double Q-learning by taking the minimum value between a pair of critics to limit overestimation. 

Below is the algorithm of this method from the paper 

![1572711292070](/home/fofo/.config/Typora/typora-user-images/1572711292070.png)

In Double Q-learning, the greedy update is disentangled from the value function by maintaining two separate value estimates, each of which is used to update the other. If the value estimates are independent, they can be used to make unbiased estimates of the actions selected using the opposite value estimate. In an actor-critic setting an analogous update uses the current policy rather than the target policy in the learning target. 

TD3 maintains a pair of critics along with a single actor. For each time step, the pair of critics are updated towards the minimum target value of the actions selected by the target policy $y=r+\gamma \min _{i=1,2} Q_{\theta_{i}^{\prime}}\left(s^{\prime}, \pi_{\phi^{\prime}}\left(s^{\prime}\right)+\epsilon\right)$



# Implementation

Training was carried out on Google Colab using a GPU run-time, making periodic recordings of the models and results every 100 episodes. The models were then run locally to produce video recordings of the evaluation. 

The model was trained for 2000 episodes. 

The implementation consists of 3 files. Besides the main.py, the RB.py describes the replay buffer and TD3.py describes the policy.

## Main.py 

The main driver file can be broadly divided into 3 parts: Setup, Gym Run, Record.

Hyperparameters were chosen according to the source paper. 

## RB.py

This is a simple container to easily store state, action and reward sequences for replay. Values are stored in numpy arrays and are retrieved via the sample method as PyTorch tensors. 

## TD3.py 

The actor network / policy is a 3 layer fully connected / dense network  with ReLu actions in the hidden layers and a tanh activation for the output layer. This selects for a max action dependant on both the state input and a previous max_action input. 

The critic networks / policies are divided into two separate architectures, Q1 & Q2. Both are 3 layer fully connected networks with ReLu actions in the hidden layers and a single output node. With inputs of both the state and action the forward pass returns a single result from both the Q1 and Q2 simultaneously. 

The TD3 object is where the Actor-Critic method is described. This consists of an actor and 2 critics.  Using samples from the replay buffer to train actions are selected from the current policy with noise added. The output state and action are used to compute the Q targets using the target critic. The lower of the Q1 and Q2 targets is selected. This is then added to previous reward and discounted. 

Current Q1 and Q2 values are then computed using the second critic. Using the current Q1 & Q 2 values and the previously computed Q target, a critic loss is computed as a mean squared error. The second critic is then optimised using backpropergation.  

Delayed policy updates are then computed using actor losses. The actor loss is computed as the negative mean of the critic Q1 output. The actor is then optimized using backpropergation. The parameters of the frozen target models are then updated. 



# Reference

1. Fujimoto, S., van Hoof, H. and Meger, D., 2018. Addressing function approximation error in actor-critic methods. *arXiv preprint arXiv:1802.09477*. 
