import abc
import copy
import itertools
import time

import acme
import dm_env
import ray
import reverb
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tree
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers


class RemoteActor(object):
  def __init__(
      self,
      actor_id,
      environment_module,
      environment_fn_name,
      environment_kwargs,
      network_module,
      network_fn_name,
      network_kwargs,
      adder_module,
      adder_fn_name,
      adder_kwargs,
      replay_server_address,
      variable_server_name,
      variable_server_address,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):
    # Counter and Logger
    self._actor_id = actor_id
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
      f'actor_{actor_id}')
    
    # Create the environment
    self._environment = getattr(environment_module, environment_fn_name)(
      **environment_kwargs)
    env_spec = acme.make_environment_spec(self._environment)
    
    # Create actor's network
    self._network = getattr(network_module, network_fn_name)(**network_kwargs)
    tf2_utils.create_variables(self._network, [env_spec.observations])
    
    self._variables = tree.flatten(self._network.variables)
    self._policy = tf.function(self._network)
    
    # The adder is used to insert observations into replay.
    self._adder = getattr(adder_module, adder_fn_name)(
      reverb.Client(replay_server_address),
      **adder_kwargs
    )
    
    variable_client = reverb.TFClient(variable_server_address)
    self._variable_dataset = variable_client.dataset(
      table=variable_server_name,
      dtypes=[tf.float32 for _ in self._variables],
      shapes=[v.shape for v in self._variables]
    )
  
  def update(self):
    """Copies the new variables to the old ones."""
    for x in self._variable_dataset.take(1):
      new_variables = x.data
      if len(self._variables) != len(new_variables):
        raise ValueError('Length mismatch between old variables and new.')
      
      for new, old in zip(new_variables, self._variables):
        old.assign(new)
  
  def run(self, update_period: int = 400, num_episodes: int = None):
    num_steps = 0
    iterator = range(num_episodes) if num_episodes else itertools.count()
    for _ in iterator:
      # Reset any counts and start the environment.
      start_time = time.time()
      episode_steps = 0
      episode_return = 0
      timestep = self._environment.reset()
      
      # Make the first observation.
      self._observe_first(timestep)
      
      # Run an episode.
      while not timestep.last():
        # Generate an action from the agent's policy and step the environment.
        action = self._select_action(timestep.observation)
        timestep = self._environment.step(action)
        
        # Have the agent observe the timestep
        self._observe(action, next_timestep=timestep)
        
        # Book-keeping.
        episode_steps += 1
        episode_return += timestep.reward
        num_steps += 1
        if num_steps % update_period == 0:
          self.update()
        #   print(self._get_probs(test_x))
      
      # Record counts.
      counts = self._counter.increment(episodes=1, steps=episode_steps)
      
      # Collect the results and combine with counts.
      steps_per_second = episode_steps / (time.time() - start_time)
      result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
      }
      result.update(counts)
      
      # Log the given results.
      self._logger.write(result)
  
  @abc.abstractmethod
  def _observe_first(self, timestep: dm_env.TimeStep):
    """Make a first observation from the environment.

    Note that this need not be an initial state, it is merely beginning the
    recording of a trajectory.

    Args:
      timestep: first timestep.
    """
  
  @abc.abstractmethod
  def _observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    """Make an observation of timestep data from the environment.

    Args:
      action: action taken in the environment.
      next_timestep: timestep produced by the environment given the action.
    """
  
  @abc.abstractmethod
  def _select_action(self, observation: types.NestedArray) -> types.NestedArray:
    """Samples from the policy and returns an action."""


@ray.remote(num_cpus=1)
class RemoteFeedForwardActor(RemoteActor):
  """A feed-forward actor.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """
  
  def __init__(
      self,
      actor_id,
      environment_module,
      environment_fn_name,
      environment_kwargs,
      network_module,
      network_fn_name,
      network_kwargs,
      adder_module,
      adder_fn_name,
      adder_kwargs,
      replay_server_address,
      variable_server_name,
      variable_server_address,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):
    super().__init__(
      actor_id=actor_id,
      environment_module=environment_module,
      environment_fn_name=environment_fn_name,
      environment_kwargs=environment_kwargs,
      network_module=network_module,
      network_fn_name=network_fn_name,
      network_kwargs=network_kwargs,
      adder_module=adder_module,
      adder_fn_name=adder_fn_name,
      adder_kwargs=adder_kwargs,
      replay_server_address=replay_server_address,
      variable_server_name=variable_server_name,
      variable_server_address=variable_server_address,
      counter=counter,
      logger=logger,
    )
  
  def _observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)
  
  def _observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if self._adder:
      self._adder.add(action, next_timestep)
  
  def _select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)
    
    # Forward the policy network.
    policy_output = self._policy(batched_obs)
    
    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output
    
    policy_output = tree.map_structure(maybe_sample, policy_output)
    
    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)
    
    return action


@ray.remote(num_cpus=1)
class RemoteRecurrentActor(RemoteActor):
  """A recurrent actor.

  An actor based on a recurrent policy which takes non-batched observations and
  outputs non-batched actions, and keeps track of the recurrent state inside. It
  also allows adding experiences to replay and updating the weights from the
  policy on the learner.
  """
  
  def __init__(
      self,
      actor_id: int,
      environment_module,
      environment_fn_name: str,
      environment_kwargs: dict,
      network_module,
      network_fn_name: str,
      network_kwargs: dict,
      adder_module,
      adder_fn_name: str,
      adder_kwargs: dict,
      replay_server_address: str,
      variable_server_name,
      variable_server_address,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):
    super().__init__(
      actor_id=actor_id,
      environment_module=environment_module,
      environment_fn_name=environment_fn_name,
      environment_kwargs=environment_kwargs,
      network_module=network_module,
      network_fn_name=network_fn_name,
      network_kwargs=network_kwargs,
      adder_module=adder_module,
      adder_fn_name=adder_fn_name,
      adder_kwargs=adder_kwargs,
      replay_server_address=replay_server_address,
      variable_server_name=variable_server_name,
      variable_server_address=variable_server_address,
      counter=counter,
      logger=logger,
    )
    self._policy = tf.function(self._network.__call__)
    self._state = None
    self._prev_state = None
  
  def _observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)
      
      # Set the state to None so that we re-initialize at the next policy call.
    self._state = None
  
  def _observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return
    
    numpy_state = tf2_utils.to_numpy_squeeze(self._prev_state)
    self._adder.add(action, next_timestep, extras=(numpy_state,))
  
  def _select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)
    
    # Initialize the RNN state if necessary.
    if self._state is None:
      self._state = self._network.initial_state(1)
    
    # Forward.
    policy_output, new_state = self._policy(batched_obs, self._state)
    
    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output
    
    policy_output = tree.map_structure(maybe_sample, policy_output)
    
    self._prev_state = self._state
    self._state = new_state
    
    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)
    
    return action


@ray.remote(num_cpus=1)
class RemoteIMPALAActor(RemoteActor):
  def __init__(
      self,
      actor_id: int,
      environment_module,
      environment_fn_name: str,
      environment_kwargs: dict,
      network_module,
      network_fn_name: str,
      network_kwargs: dict,
      adder_module,
      adder_fn_name: str,
      adder_kwargs: dict,
      replay_server_address: str,
      variable_server_name,
      variable_server_address,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):
    super().__init__(
      actor_id=actor_id,
      environment_module=environment_module,
      environment_fn_name=environment_fn_name,
      environment_kwargs=environment_kwargs,
      network_module=network_module,
      network_fn_name=network_fn_name,
      network_kwargs=network_kwargs,
      adder_module=adder_module,
      adder_fn_name=adder_fn_name,
      adder_kwargs=adder_kwargs,
      replay_server_address=replay_server_address,
      variable_server_name=variable_server_name,
      variable_server_address=variable_server_address,
      counter=counter,
      logger=logger,
    )
    self._policy = tf.function(self._network.__call__)
    self._state = None
    self._prev_state = None
    self._prev_logits = None
  
  def _select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)
    
    if self._state is None:
      self._state = self._network.initial_state(1)
    
    # Forward.
    (logits, _), new_state = self._policy(batched_obs, self._state)
    
    self._prev_logits = logits
    self._prev_state = self._state
    self._state = new_state
    
    action = tfd.Categorical(logits).sample()
    action = tf2_utils.to_numpy_squeeze(action)
    
    return action
  
  def _observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)
    
    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None
  
  def _observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return
    
    extras = {'logits': self._prev_logits, 'core_state': self._prev_state}
    extras = tf2_utils.to_numpy_squeeze(extras)
    self._adder.add(action, next_timestep, extras)
