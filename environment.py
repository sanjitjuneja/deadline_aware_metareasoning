import gymnasium as gym
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


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
        planning_time_distribution = self.planning_time_distributions[action]
        planning_time = planning_time_distribution.ppf(np.random.rand())
        execution_time_distribution = self.execution_time_distributions[action]
        execution_time = execution_time_distribution.rvs()

        # Update State (Timestep, Accumulated Planning Time, Accumulated Execution Time, Latest Symbolic Action)
        self.state[0] += 1
        self.state[1] += planning_time
        self.state[2] += execution_time
        self.state[3] = action

        # Calculate reward and check if the episode is done
        reward = self.calculate_reward()
        done = True if reward == 0 or reward == 1 else False

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
        if mode == 'console':
            # Console rendering logic
            print("-" * 50)
            print(f"Current State: ")
            print("Timestep: ", self.state[0])
            print("Accumulated Planning Time: ", self.state[1])
            print("Accumulated Execution Time: ", self.state[2])
            print("Last Symbolic Action: ", self.state[3])
            print("-" * 50)

        elif mode == 'plot':
            pass
            # ct, pti, eti, ri = self.state
            #
            # # Clearing previous figure and setting up new plots
            # plt.clf()
            #
            # # Subplot for the Current Timestep vs Deadline
            # plt.subplot(2, 2, 1)
            # plt.bar(['Current Timestep', 'Deadline'], [ct, self.deadline], color=['blue', 'red'])
            # plt.title('Current Timestep vs Deadline')
            #
            # # Subplot for Accumulated Planning and Execution Time
            # plt.subplot(2, 2, 2)
            # plt.bar(['Planning Time', 'Execution Time'], [pti, eti], color=['green', 'orange'])
            # plt.title('Accumulated Times')
            #
            # # Subplot for Current Plan Index
            # plt.subplot(2, 2, 3)
            # plt.bar('Current Plan Index', ri, color='purple')
            # plt.title('Current Plan Index')
            # plt.xticks([])  # No x-ticks
            #
            # # Subplot for Progress (ct relative to Deadline)
            # plt.subplot(2, 2, 4)
            # progress = ct / self.deadline
            # plt.bar('Progress', progress, color='cyan')
            # plt.title('Progress')
            # plt.xticks([])  # No x-ticks
            #
            # plt.tight_layout()
            # plt.pause(0.1)  # Pause to update the plots
            #
            # input("Press Enter to continue...")  # Wait for user input to proceed

        else:
            raise NotImplementedError

    def calculate_reward(self):
        ct, pti, eti, ri = self.state

        # Reward for completing all plans within the deadline
        if ri == self.num_symbolic_plans - 1 and ct + eti <= self.deadline:
            return 1  # Maximum reward for full success

        # No reward if the deadline is exceeded
        if ct > self.deadline:
            return 0

        # Calculate proportional reward based on progress and time efficiency
        # Ensure these values contribute less than 1 when combined
        progress = ri / (self.num_symbolic_plans - 1)
        time_efficiency = max(0, 1 - (ct + eti) / self.deadline)

        # Balance the reward to be less than 1 for partial success
        reward = 0.5 * progress + 0.5 * time_efficiency

        # Ensure the reward is strictly less than 1 for incomplete tasks
        return min(reward, 0.99)

    def _init_distributions(self):
        """
        Helper Function: Initializes the planning and execution time distributions for each plan
        :return: None
        """
        self.planning_time_distributions = []
        self.execution_time_distributions = []

        for _ in range(self.num_symbolic_plans):
            # For simplicity, using a fixed distribution for planning and execution times per plan
            mean_planning_time, std_dev_planning_time = 1.0, 0.5  # Example values
            planning_distribution = stats.norm(mean_planning_time, std_dev_planning_time)
            self.planning_time_distributions.append(planning_distribution)

            lambda_execution_time = 2.0  # Example value for Poisson distribution
            execution_distribution = stats.poisson(lambda_execution_time)
            self.execution_time_distributions.append(execution_distribution)
