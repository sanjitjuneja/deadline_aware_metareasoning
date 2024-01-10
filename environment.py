import gymnasium as gym
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class DeadlineAwareMetaReasoningEnv(gym.Env):
    """
    Custom Environment for Deadline-Aware Metareasoning.
    This simulates a set of symbolic plans, each with a sequence of actions, under a deadline constraint.
    """
    metadata = {'render.modes': ['human', 'console']}

    def __init__(self, num_plans, max_actions_per_plan, deadline):
        super(DeadlineAwareMetaReasoningEnv, self).__init__()
        self.num_plans = num_plans
        self.max_actions_per_plan = max_actions_per_plan
        self.deadline = deadline

        self._init_distributions()

        self.action_space = gym.spaces.Discrete(self.num_plans)
        self.observation_space = gym.spaces.Box(
            low=np.array([0] + [0]*self.num_plans + [0]*self.num_plans),
            high=np.array([self.deadline] + [float('inf')]*self.num_plans + [self.max_actions_per_plan]*self.num_plans),
            dtype=np.float32
        )

        # Initialize history tracking
        self.action_history = []
        self.state_history = []
        self.episode_count = 0
        self.latest_action = -1

        self.fig, self.ax = plt.subplots()

        self.state = self.reset()

    def _init_distributions(self):
        self.planning_time_distributions = []
        self.execution_time_distributions = []
        for _ in range(self.num_plans):
            plan_planning_times = []
            plan_execution_times = []
            for _ in range(self.max_actions_per_plan):
                mean, std = np.random.uniform(0.5, 2), np.random.uniform(0.1, 0.5)
                plan_planning_times.append(stats.norm(mean, std))
                lambda_param = np.random.uniform(1, 4)
                plan_execution_times.append(stats.poisson(lambda_param))
            self.planning_time_distributions.append(plan_planning_times)
            self.execution_time_distributions.append(plan_execution_times)

    def step(self, action):
        self.state[0] += 1

        planning_time_distribution = self.planning_time_distributions[action][0]
        execution_time_distribution = self.execution_time_distributions[action][0]

        # Generate planning time using inverse transform sampling
        planning_time = planning_time_distribution.ppf(np.random.rand())

        # Select execution time from Poisson distribution
        execution_time = execution_time_distribution.rvs()

        self.state[1 + action] += planning_time
        self.state[1 + self.num_plans + action] += execution_time

        # Check for completion of individual tasks and all tasks
        task_completed = self.state[1 + self.num_plans + action] >= self.max_actions_per_plan
        all_tasks_completed = all(
            [self.state[1 + self.num_plans + i] >= self.max_actions_per_plan for i in range(self.num_plans)])

        reward = 0

        # Reward/Penalty Logic
        if all_tasks_completed and self.state[0] <= self.deadline:
            # Big bonus for completing all tasks within the deadline
            reward += 20 + (self.deadline - self.state[0]) * 0.5  # Scaled bonus based on remaining time
        elif self.state[0] > self.deadline:
            # Heavy penalty for exceeding deadline
            reward -= 15
        else:
            # Small penalty for each timestep to encourage efficiency
            reward -= 0.1

        if task_completed:
            # Incremental reward for each task completed, scaled by speed
            time_taken = self.state[1 + action]  # Example: time taken for the completed task
            reward += max(5 - time_taken * 0.1, 1)  # Scaled reward for task completion

        done = all_tasks_completed or self.state[0] > self.deadline

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0] + [0.0] * self.num_plans + [0] * self.num_plans)
        self.episode_count += 1
        self.action_history = []
        self.state_history = []
        return self.state

    def render(self, mode='console'):
        if mode == 'console':
            # Console rendering logic
            pass
            # print(f"Current State: {self.state}")

        elif mode == 'human':
            if self.episode_count == 0:
                # Setup the plot for the first episode
                plt.ion()
                self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8))  # Two subplots for state and actions
                plt.show()

            self.update_graph(self.cumulative_rewards_graph, new_reward_data)
            self.update_graph(self.actions_graph, new_action_data)

            # Update history
            self.action_history.append(self.latest_action)
            self.state_history.append(self.state.copy())

            # Clear previous content
            self.ax.clear()
            self.ax.clear()

            # Plot state history
            state_history_array = np.array(self.state_history)
            for i in range(state_history_array.shape[1]):
                self.ax.plot(state_history_array[:, i], label=f'State {i}')

            # Plot action history
            self.ax.stem(self.action_history)
            self.ax.set_ylim(0, self.num_plans)
            self.ax.set_ylabel('Actions')
            self.ax.set_xlabel('Timestep')

            # Add legends and titles
            self.ax.legend(loc='upper left')
            self.ax.set_title('State Over Time')
            self.ax.set_title('Action History')

            # Draw and pause
            self.fig.canvas.draw()
            plt.pause(0.001)

            if self.check_for_pause():
                self.pause_rendering()

            if self.should_save_graphs():
                self.save_graphs()

    def close(self):
        pass