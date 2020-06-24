import functools

import dm_env
import gym
import sonnet as snt
import tensorflow as tf
import trfl
from acme import wrappers
from acme.tf import networks


def R2D2AtariActorNetwork(num_actions: int, epsilon: tf.Variable):
  network = networks.R2D2AtariNetwork(num_actions)
  return snt.DeepRNN([
    network,
    lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
  ])


def make_r2d2_atari_environment(
    task_and_level: str = 'PongNoFrameskip-v4',
    evaluation: bool = False
) -> dm_env.Environment:
  env = gym.make(task_and_level, full_action_space=True)
  
  max_episode_len = 108_000 if evaluation else 50_000
  
  return wrappers.wrap_all(env, [
    wrappers.GymAtariAdapter,
    functools.partial(
      wrappers.AtariWrapper,
      to_float=True,
      max_episode_len=max_episode_len,
      zero_discount_on_life_loss=True,
    ),
    wrappers.SinglePrecisionWrapper,
    wrappers.ObservationActionRewardWrapper
  ])
