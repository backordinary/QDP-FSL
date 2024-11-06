# https://github.com/seunomonije/quantum-programming-api/blob/b2d45cdbf13b8e4d3917d9bea6317898da71aa33/qapi/system_configuration/simulated_backends/qiskit/QiskitSimulator.py
from ..Simulator import Simulator
from ..exceptions import *

from qiskit import Aer, execute

class QiskitSimulator(Simulator):
  def __init__(self, backend='qasm_simulator'):
    print(f'Active simulator: {backend}')

    try:
      self.backend = Aer.get_backend(backend)
    except:
      raise InvalidBackendError('Invalid backend name entered')
    