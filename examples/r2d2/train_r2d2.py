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
