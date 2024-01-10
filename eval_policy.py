import numpy as np


def _log_summary(ep_len, ep_ret, ep_num, total_reward):
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
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(f"Number of Episodes: {ep_num + 1}", flush=True)
    print(f"Average Reward: {round(total_reward / (ep_num + 1), 2)}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy, env, render):
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

            obs, rew, done, _ = env.step(action)
            ep_ret += rew

        ep_len = t
        yield ep_len, ep_ret


def eval_policy(policy, env, render=False):
    """
    Main function to evaluate our policy with. Will run forever until the process is killed.
    :param policy: the trained policy to test (actor model)
    :param env: the environment to test the policy on
    :param render: whether we should render our episodes, False by default
    :return: None
    """
    total_reward = 0
    # Rollout with the policy and environment, look at rollout's function description
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
        total_reward += ep_ret
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num, total_reward=total_reward)
