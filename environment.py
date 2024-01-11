import gymnasium as gym
import numpy as np
import scipy.stats as stats


class DeadlineAwareMetaReasoningEnv(gym.Env):
    """
    Custom Gym Environment for Deadline-Aware Metareasoning
    """

    def __init__(self, num_symbolic_plans, deadline):
        """
        Initializes the environment with the following parameters:
        :param num_symbolic_plans: Number of symbolic plans to simulate (int)
        :param deadline: Deadline for the episode (int)
        """
        super(DeadlineAwareMetaReasoningEnv, self).__init__()
        self.num_symbolic_plans = num_symbolic_plans
        self.deadline = deadline

        # Initialize the planning and execution time distributions for each plan
        self._init_distributions()

        # Actions represented by a discrete set where each action corresponds to choosing a plan to work on
        self.action_space = gym.spaces.Discrete(self.num_symbolic_plans)

        # Observations of states are represented by a Box space
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]),  # Lower bounds for ct, pti, eti, ri
            high=np.array([self.deadline, np.inf, np.inf, self.num_symbolic_plans]),  # Upper bounds
            dtype=np.float32
        )

        # Initializes starting state (ct, pti, eti, ri)
        self.state = self.reset()

    def step(self, action):
        """
        Takes a step in the environment by choosing an action to work on
        :param action: Selected symbolic plan to work on (int)
        :return: New state (list), reward (float), done (bool)
        """
        # Get the planning and execution time for the chosen plan to work on (aka. action)
        planning_time_distribution = self.planning_time_distributions[action][0]
        planning_time = planning_time_distribution.ppf(np.random.rand())
        execution_time_distribution = self.execution_time_distributions[action][0]
        execution_time = execution_time_distribution.rvs()

        # Update State (Timestep, Accumulated Planning Time, Accumulated Execution Time, Latest Symbolic Action)
        self.state[0] += 1
        self.state[1] += planning_time
        self.state[2] += execution_time
        self.state[3] = action

        # Calculate reward and check if the episode is done
        reward = self.calculate_reward()
        done = True if reward in [0, 1] else False

        # Return the new state, reward, and done
        return self.state, reward, done

    def reset(self):
        """
        Resets the environment to the initial state
        :return: New state (list)
        """
        # Reset the state variables to their initial values
        self.state = [0, 0.0, 0.0, 0]  # Resetting ct, pti, eti, ri
        return self.state

    def render(self, mode='console'):
        """
        Renders the current state of the environment
        :param mode: Rendering mode, 'console' or human' (str)
        :return: None
        """
        if mode == 'human':
            pass
        else:
            print("-" * 50)
            print(f"Current State: ")
            print("Timestep: ", self.state[0])
            print("Accumulated Planning Time: ", self.state[1])
            print("Accumulated Execution Time: ", self.state[2])
            print("Last Symbolic Action: ", self.state[3])
            print("-" * 50)

    def calculate_reward(self):
        """
        Helper Function: Calculates the reward for the current state
        :return: Reward (float)
        """
        ct, pti, eti, ri = self.state
        if ri == self.num_symbolic_plans and ct + eti <= self.deadline:
            return 1  # Reward for completing all plans within the deadline
        elif ct > self.deadline:
            return 0 # Failure terminal state
        else:
            # Reward for making progress towards completing all plans
            return 0.1 * (1 - (pti + eti) / self.deadline) + 0.9 * (1 - (pti + eti) / (self.deadline * self.num_symbolic_plans))

    def _init_distributions(self):
        """
        Helper Function: Initializes the planning and execution time distributions for each plan.
        :return: None
        """
        self.planning_time_distributions = []
        self.execution_time_distributions = []
        for _ in range(self.num_symbolic_plans):
            plan_planning_times = []
            plan_execution_times = []
            for _ in range(self.max_actions_per_plan):
                mean, std = np.random.uniform(0.5, 2), np.random.uniform(0.1, 0.5)
                plan_planning_times.append(stats.norm(mean, std))
                lambda_param = np.random.uniform(1, 4)
                plan_execution_times.append(stats.poisson(lambda_param))
            self.planning_time_distributions.append(plan_planning_times)
            self.execution_time_distributions.append(plan_execution_times)
