# https://github.com/abhishekagarwal2301/isl/blob/5e40f3cd0d36af1ce0a2a09b3a5a22a6c015eb7d/isl/examples/simple_isl_example.py
import logging

from qiskit import QuantumCircuit

import isl.utils.circuit_operations as co
from isl.recompilers import ISLRecompiler

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("qiskit").setLevel(logging.WARNING)

# Create circuit creating a random initial state
qc = co.create_random_initial_state_circuit(4)

isl_recompiler = ISLRecompiler(qc)

result = isl_recompiler.recompile()
approx_circuit = result["circuit"]
print(f"Overlap between circuits is {result['overlap']}")
print(f'{"-"*32}')
print(f'{"-"*10}OLD  CIRCUIT{"-"*10}')
print(f'{"-"*32}')
print(qc)
print(f'{"-"*32}')
print(f'{"-"*10}ISL  CIRCUIT{"-"*10}')
print(f'{"-"*32}')
print(approx_circuit)
