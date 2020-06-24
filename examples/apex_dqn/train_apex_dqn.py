import acme
import ray
import reverb
import tensorflow as tf
import user_modules
from acme.agents.tf import dqn
from acme.tf import networks
from acme.tf import utils as tf2_utils

from acme_dist_toolkit.remote_actors import RemoteFeedForwardActor
from acme_dist_toolkit.remote_environments import create_env_fns
from acme_dist_toolkit.remote_variable_client import RemoteVariableClient

ray.init()

# Set gpu config
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(device=gpu, enable=True)
tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')

# Create network
env = create_env_fns['atari']('PongNoFrameskip-v4')
env_spec = acme.make_environment_spec(env)
network = networks.DQNAtariNetwork(env_spec.actions.num_values)
tf2_utils.create_variables(network, [env_spec.observations])

# Create a variable replay buffer for sharing parameters
# between the learner and the actor
variable_server = reverb.Server(tables=[
  reverb.Table(
    name='variable_server',
    sampler=reverb.selectors.Fifo(),
    remover=reverb.selectors.Fifo(),
    max_size=20,
    rate_limiter=reverb.rate_limiters.MinSize(1)),
])
variable_server_address = f'localhost:{variable_server.port}'
variable_client = RemoteVariableClient.remote('variable_server',
                                              variable_server_address)
agent = dqn.DQN(env_spec, network,
                batch_size=512, prefetch_size=16)
replay_server_address = agent._actor._adder._client.server_address
variable_client.add.remote(agent._learner._variables)

remote_processes = []
epsilon = 0.4
alpha = 7
num_actors = 8
for i in range(num_actors):
  actor_epsilon = pow(epsilon, 1 + i / (num_actors - 1) * alpha)
  remote_actor = RemoteFeedForwardActor.remote(
    actor_id=i + 1,
    environment_module=user_modules,
    environment_fn_name='make_dqn_atari_environment',
    environment_kwargs={'task_and_level': 'PongNoFrameskip-v4'},
    network_module=user_modules,
    network_fn_name='DQNAtariActorNetwork',
    network_kwargs={'num_actions': env_spec.actions.num_values,
                    'epsilon': tf.Variable(actor_epsilon)},
    adder_module=acme.adders.reverb.transition,
    adder_fn_name='NStepTransitionAdder',
    adder_kwargs={'n_step': 1, 'discount': 0.99},
    replay_server_address=replay_server_address,
    variable_server_name='variable_server',
    variable_server_address=variable_server_address
  )
  remote_processes.append(remote_actor.run.remote(400))

agent._learner.step()
print("--------------------------Start Training--------------------------")
while True:
  for _ in range(18):
    agent._learner.step()
  variable_client.add.remote(agent._learner._variables)
