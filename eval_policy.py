import numpy as np


def _log_summary(ep_len, ep_ret, ep_num, total_reward, num_successful_episodes, policy_name):
    """
    Prints to stdout what we've logged so far in the most recent episode.
    :param ep_len: episodic length
    :param ep_ret: episodic return
    :param ep_num: episode number
    :return: None
    """

    # Round decimal places for better logging messages
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print(f"--------------------- Episode #{ep_num} ({policy_name}) ---------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"--------------------------------------------------------------------------", flush=True)
    print(f"Number of Episodes: {ep_num + 1}", flush=True)
    print(f"Number of Successes: {num_successful_episodes}", flush=True)
    print(f"Success Rate: {round(num_successful_episodes / (ep_num + 1), 2)}", flush=True)
    print(f"Average Reward: {round(total_reward / (ep_num + 1), 2)}", flush=True)
    print(f"--------------------------------------------------------------------------", flush=True)
    print(flush=True)


def ppo_policy_rollout(policy, env, render):
    """
    Returns a generator to roll out each episode given a trained policy and environment to test on.
    :param policy: the trained policy to test
    :param env: the environment to evaluate the policy on
    :param render: specifies whether to render or not
    :return: generator object rollout which returns the latest ep_len and ep_ret on each iteration
    """

    while True:
        obs = env.reset()
        done = False
        t = 0  # Number of timesteps so far
        ep_len = 0  # Episodic length
        ep_ret = 0  # Episodic return

        while not done:
            t += 1

            if render:
                env.render()

            # Query deterministic action from policy
            action_probs = policy(obs).detach().cpu().numpy()
            action = np.argmax(action_probs)  # Select the action with the highest probability

            obs, rew, done = env.step(action)
            ep_ret += rew

        ep_len = t
        yield ep_len, ep_ret/ep_len


def random_policy_rollout(env, render):
    """
    Rollout episodes with a random policy.
    :param env: the environment to evaluate the policy on
    :param render: specifies whether to render or not
    :return: generator object rollout which returns the latest ep_len and ep_ret on each iteration
    """
    while True:
        obs = env.reset()
        done = False
        t = 0
        ep_len = 0
        ep_ret = 0

        while not done:
            t += 1

            if render:
                env.render()

            # Random action
            action = env.action_space.sample()

            obs, rew, done = env.step(action)
            ep_ret += rew

        ep_len = t
        yield ep_len, ep_ret/ep_len


def eval_policy(policy, env, render=False):
    """
    Evaluate both PPO policy and a random policy.
    """
    total_reward_ppo = 0
    total_reward_random = 0
    num_successes_ppo = 0
    num_successes_random = 0

    # Rollout with PPO policy and random policy
    for ep_num, ((ep_len_ppo, ep_ret_ppo), (ep_len_random, ep_ret_random)) in enumerate(zip(ppo_policy_rollout(policy, env, render), random_policy_rollout(env, render))):
        total_reward_ppo += ep_ret_ppo
        total_reward_random += ep_ret_random
        if ep_ret_ppo == 1:
            num_successes_ppo += 1
        if ep_ret_random == 1:
            num_successes_random += 1

        _log_summary(ep_len_ppo, ep_ret_ppo, ep_num, total_reward_ppo, num_successes_ppo, "PPO Policy")
        _log_summary(ep_len_random, ep_ret_random, ep_num, total_reward_random, num_successes_random, "Random Policy")
