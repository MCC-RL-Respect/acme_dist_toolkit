import functools

import bsuite
import dm_env
import gym
from acme import wrappers
# from dm_control import suite
from collections import namedtuple


def make_gym_environment(
    task_name: str = 'MountainCarContinuous-v0'
) -> dm_env.Environment:
  """Creates an OpenAI Gym environment."""
  
  # Load the gym environment.
  environment = gym.make(task_name)
  
  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  
  return environment


# def make_control_suite_environment(
#     domain_name: str = 'cartpole',
#     task_name: str = 'balance'
# ) -> dm_env.Environment:
#   """Creates a control suite environment."""
#   environment = suite.load(domain_name, task_name)
#   environment = wrappers.SinglePrecisionWrapper(environment)
#   return environment


def make_bsuite_environment(
    bsuite_id: str = 'deep_sea/0',
    results_dir: str = '/tmp/bsuite',
    overwrite: bool = False
) -> dm_env.Environment:
  raw_environment = bsuite.load_and_record_to_csv(
    bsuite_id=bsuite_id,
    results_dir=results_dir,
    overwrite=overwrite,
  )
  return wrappers.SinglePrecisionWrapper(raw_environment)


def make_dqn_atari_environment(
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


create_env_fns = {
  'gym': make_gym_environment,
  # 'control_suite': make_control_suite_environment,
  'bsuite': make_bsuite_environment,
  'atari': make_dqn_atari_environment,
  'r2d2_atari': make_r2d2_atari_environment
}
