import os

import sys
import torch

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy
from environment import DeadlineAwareMetaReasoningEnv


def train(env, hyperparameters, actor_model=None, critic_model=None):
    """
    Trains the model.
    :param env: the environment to train the policy on
    :param hyperparameters: a dict of hyperparameters to use, defined in main
    :param actor_model: the actor model to load in if we want to continue training
    :param critic_model: the critic model to load in if we want to continue training
    :return: None
    """

    print(f"Training", flush=True)

    # Create a model for PPO
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # Loads in existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
    elif actor_model == '' and critic_model == '':
        print(f"Training from scratch.", flush=True)
    else:
        print(f"Error: Either specify both actor/critic models or none at all.")
        sys.exit(0)

    # Train the PPO model with specified total timesteps
    model.learn(total_timesteps=200_000_000)


def test(env, actor_model):
    """
    Tests the model.
    :param env: the environment to test the policy on
    :param actor_model: the actor model to load in
    :return: None
    """

    print(f"Testing {actor_model}", flush=True)

    # Exit if actor model is not specified
    if actor_model == '':
        print(f"Didn't specify model file. Exiting", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Build policy from our NN defined in network
    if os.environ.get('TORCH_DEVICE'):
        device = torch.device(os.environ.get('TORCH_DEVICE'))
    else:
        device = torch.device('cpu')
    policy = FeedForwardNN(obs_dim, act_dim).to(device)

    # Load in actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))
    policy.to(device)

    # Evaluate our policy with separate model, eval_policy
    eval_policy(policy=policy, env=env, render=True)


def main(args):
    """
    The main function to run.
    :param args: the arguments passed in from the command line
    :return: None
    """

    hyperparameters = {
        'timesteps_per_batch': 1024,
        'max_timesteps_per_episode': 100,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 2e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    # Creates the environment that will be running
    env = DeadlineAwareMetaReasoningEnv(num_plans=3, max_actions_per_plan=5, deadline=10)

    # Train or test, specified through passed arguments
    if args.mode == 'train':
        train(env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env, actor_model=args.actor_model)


if __name__ == '__main__':
    args = get_args()
    main(args)
