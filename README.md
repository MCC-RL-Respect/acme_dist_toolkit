A toolkit to extend acme to distributed version
Currently, I only implement the APEX-DQN and R2D2
### How I did that?
I use the rpc tool `ray` to build remote actors and remote variable clients, you can extend acme's algorithms with several lines. 
### How to use it?
I provided the steps to run R2D2 with acme_dist_toolkit and acme
1. Install acme, ray
1. Install acme_dist_toolkit using pip or put the folder in your project directly
2. Create a directory `r2d2`
3. Create a new python file name `user_modules.py` in the folder
4. Write your environment's wrapper, for example, `make_dqn_atari_environment()`
```python
import sonnet as snt
import trfl
import tensorflow as tf
from acme.tf import networks
def R2D2AtariActorNetwork(num_actions: int, epsilon: tf.Variable):
  network = networks.R2D2AtariNetwork(num_actions)
  return snt.DeepRNN([
    network,
    lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
  ])
```
5. Write your network's wrapper
```python
import functools
import dm_env
import gym
from acme import wrappers

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
```
6. Create a new python file name `train_r2d2.py` in the folder and do the following things:
    1. create remote actors
    2. create acme agent
    3. create a local variable server and remote variable clients to sharing parameters of the network
    4. run remote actors and the learner of the acme agent
```python
import acme
import ray
import reverb
import tensorflow as tf
import user_modules
from acme.agents.tf import r2d2
from acme.tf import networks
from acme.tf import utils as tf2_utils

from acme_dist_toolkit.remote_actors import RemoteRecurrentActor
from acme_dist_toolkit.remote_variable_client import RemoteVariableClient
from acme_dist_toolkit.remote_environments import create_env_fns

# Set gpu config
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(device=gpu, enable=True)
tf.config.set_visible_devices(devices=gpus[-1], device_type='GPU')
ray.init()

# Create network
env = create_env_fns['r2d2_atari']('PongNoFrameskip-v4')
env_spec = acme.make_environment_spec(env)
network = networks.R2D2AtariNetwork(env_spec.actions.num_values)
tf2_utils.create_variables(network, [env_spec.observations])

# Create a variable replay buffer for sharing parameters
# between the learner and the actor
variable_server = reverb.Server(tables=[
  reverb.Table(
    name='variable_server',
    sampler=reverb.selectors.Fifo(),
    remover=reverb.selectors.Fifo(),
    max_size=5,
    rate_limiter=reverb.rate_limiters.MinSize(1)),
])
variable_server_address = f'localhost:{variable_server.port}'
variable_client = RemoteVariableClient.remote('variable_server',
                                              variable_server_address)

agent = r2d2.R2D2(env_spec, network, burn_in_length=40, trace_length=39,
                  replay_period=40, batch_size=64, target_update_period=2500)

replay_server_address = agent._actor._adder._client.server_address
variable_client.add.remote(agent._learner._variables)

remote_processes = []
epsilon = 0.4
alpha = 7
num_actors = 8
for i in range(num_actors):
  actor_epsilon = pow(epsilon, 1 + i / (num_actors - 1) * alpha)
  remote_actor = RemoteRecurrentActor.remote(
    actor_id=i,
    environment_module=user_modules,
    environment_fn_name='make_r2d2_atari_environment',
    environment_kwargs={'task_and_level': 'PongNoFrameskip-v4'},
    network_module=user_modules,
    network_fn_name='R2D2AtariActorNetwork',
    network_kwargs={'num_actions': env_spec.actions.num_values,
                    'epsilon': tf.Variable(actor_epsilon)},
    adder_module=acme.adders.reverb.sequence,
    adder_fn_name='SequenceAdder',
    adder_kwargs={'period': 40, 'sequence_length': 80},
    replay_server_address=replay_server_address,
    variable_server_name='variable_server',
    variable_server_address=variable_server_address
  )
  remote_processes.append(remote_actor.run.remote(400))

agent._learner.step()
print("--------------------------Start Training--------------------------")
while True:
  for _ in range(3):
    agent._learner.step()
  variable_client.add.remote(agent._learner._variables)
```

### Limitations
The remote actors built by ray can not use gpus with limiting the gpu memory. 
Please see more detials in ![https://github.com/ray-project/ray/issues/6633](https://github.com/ray-project/ray/issues/6633)