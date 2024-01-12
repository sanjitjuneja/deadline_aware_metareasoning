# Deadline-Aware Metareasoning PPO

This project leverages Proximal Policy Optimization (PPO) to train an AI model in a custom reinforcement learning environment based on deadline aware metareasoning. The core challenge involves managing multiple symbolic plans with a strict deadline constraint, requiring the AI model to make strategic decisions to optimize plan execution under time pressure.

## Project File Structure

- `environment.py`: Implements a custom gym environment simulating multiple symbolic plans with a deadline constraint.
- `network.py`: Defines the neural network architecture used in the PPO model.
- `ppo.py`: Contains the implementation of the Proximal Policy Optimization algorithm, crucial for training the AI model.
- `main.py`: The main script responsible for training and testing the model.
- `eval_policy.py`: Used for evaluating the performance of the trained model against baseline policies and various configurations.

## Install & Usage

### Setting Up the Environment:
- Clone the repository.
- Set up a Python virtual environment.
- Install the required packages using `pip install -r requirements.txt`.
- Create a .env file for configuring torch device settings. (Optional, defaults to "cpu" if not specified)
  - TORCH_DEVICE: Set to "cpu" or "cuda" depending on your device.

### Running the Program:
- Review `main.py` for the training and testing logic.
  - The custom environment is initialized to simulate 3 plans with a deadline of 10 timesteps. (this is what the pre-trained model was trained on)
- Testing:
  - Run `python main.py --test --actor_model='ppo_actor.pth` to test the model against a random policy.
- Training:
  - Run `python main.py --train --actor_model='ppo_actor.pth' --critic_model='ppo_critic.pth'` to continue training a previously saved model.
  - Run `python main.py --train` to train the model from scratch.


## Performance Evaluation

During testing (aka. `eval_policy.py`), the actor model is evaluated against a random policy also running in the same environment. The random policy serves as a baseline, typically achieving a success rate inversely proportional to the number of plans when tested on enough iterations. Aka 1/3 success rate for 3 plans, 1/4 for 4 plans, etc.
The trained model, on the other hand, is able to achieve a 100% success rate (tested on 10,000s of iterations) in the current environment setup with 3 plans and a deadline of 10 timesteps.
When testing against a more complex environment by reducing the deadline to 6, the success rate drops to around 98%.
All of this is able to reproduced by simply running the testing script (`python main.py --test --actor_model='ppo_actor.pth'`) and changing the environment parameters in `main.py`.


## Custom Environment

The custom environment is implemented using the OpenAI Gym framework. The environment is initialized with the following parameters:
- Number of Plans
- Deadline

Here are descriptions of each of the main elements/components of the environment (comments also included in source file):

- State Representation: `(ct, pti, eti, ri)`
  - `ct`: Current timestep
  - `pti`: Cumulative planning time for plan i
  - `eti`: Cumulative execution time for plan i
  - `ri`: Index of the current action for plan i
- Action Space
  - Discrete space with `n` actions, where `n` is the number of plans
  - Each action corresponds to the index of the plan to execute next
  - Example: `n = 3` => `action_space = [0, 1, 2]`
- Observation Space
  - Box space with `4n` dimensions, where `n` is the number of plans
  - Each plan has 4 dimensions: `ct`, `pti`, `eti`, `ri`
- Planning & Execution Time Distributions
  - The planning time for each symbolic plan is determined using a normal distribution, which is continuous. The implementation within the `step` function specifically uses the CDF of this distribution through the utilization of the Percent Point Function (PPF) method, which returns the value of the inverse CDF at a given probability.
  - The execution time, the implementation uses a Poisson distribution, which is a discrete probability distribution. The implementation within the `step` function uses the rvs or random variates method, which returns a random sample from the distribution. This method effectively utilizes the PMF to provide a probable execution time.
- Step Function & Transition Dynamics
  - The step function takes in an action and returns the next state, reward, and done flag
    - Increment `ct` by 1
    - Updates the cumulative planning `pti` and execution `eti` times for the chosen plan based onm the respective distributions
    - Changes the index of the current action (`rt`) for the chosen plan
- Reward Calculation
  - Completion Reward
    - Reward returned is 1 (this is the max reward possible)
  - Progress-Based Reward
    - Receives a proportional reward based on the number of plans completed and the efficiency in terms of time usage
  - Penalty for Exceeding Deadline
    - Reward returned is 0 (this is the min reward possible)


## Neural Network Architecture

- The model uses a feedforward neural network with two hidden layers. Each hidden layer is followed by a ReLU activation function. The output layer is a linear layer.
- Layer configurations are tailored to process the state information effectively and output a probability distribution over the action space.
- The actor and critic networks share the same architecture, with the actor network outputting a probability distribution over the action space and the critic network outputting a value estimate for the current state.

## Hyperparameter Selection

The following hyperparameters have been chosen with the following initializations and can be configured in `main.py`:

- `timesteps_per_batch`: 1024
  - A value of 1024 strikes a balance between having enough data to accurately estimate the advantage at each step and keeping the training iterations computationally manageable.
- `max_timesteps_per_episode`: 100
  - The maximum number of timesteps allowed per episode. This is set to 100 to ensure that the agent has enough time to complete all plans.
- `gamma`: 0.99
  - Set to encourage the agent to plan strategically for the long term while still acknowledging immediate outcomes.
- `n_updates_per_iteration`: 10
  - The number of times the model is updated per iteration. This is set to 10 to ensure that the model is updated enough to learn from the collected data.
- `lr`: 2e-4
  - The learning rate for the Adam optimizer is set to 2e-4, which is small enough to prevent overshooting optimal policies but large enough to ensure reasonable convergence times. 
- `clip`: 0.2
  - The clipping parameter is set to 0.2 to ensure that the model is stable by limiting the ratio of new and old policies. 0.2 prevents the policy from changing too drastically in a single update. 


## Proposed Environment Enhancements

- Complexities and Urgencies for Each Plan: Introduce varying levels of difficulty and priority for each plan, adding depth to the decision-making process.
- Expanded Action Space: Include additional actions like reallocating resources between plans or adjusting the execution order.
- Refined Reward Function: Develop a more sophisticated reward mechanism to provide nuanced feedback based on the agent's performance. (Not limited to a space between 0 and 1)

## References 

1. Original sources and inspiration for the [PPO algorithm](https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py) and other boilerplate code used for this project
2. [PPO-Clip Algo](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)
3. Spinning Up in Deep RL by OpenAI: [PPO](https://spinningup.openai.com/en/latest/spinningup/spinningup.html)