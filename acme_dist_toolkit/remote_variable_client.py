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


@ray.remote
class RemoteVariableClient:
  def __init__(self, variable_server_name, variable_server_address):
    self._variable_server_name = variable_server_name
    self._variable_client = reverb.Client(variable_server_address),
  
  def add(self, variables):
    np_variables = [
      tf2_utils.to_numpy(v) for v in variables
    ]
    self._variable_client[0].insert(np_variables,
                                 priorities={self._variable_server_name: 1.0})
