# https://github.com/jdanielescanez/quantum-solver/blob/d8357c0c04e097ad52ec99d953c65eaeb6476c7f/src/b92/sender.py
#!/usr/bin/env python3

# Author: Daniel Escanez-Exposito

from b92.participant import Participant
from qiskit import QuantumCircuit

## The Sender entity in the B92 implementation
## @see https://qiskit.org/textbook/ch-algorithms/quantum-key-distribution.html
class Sender(Participant):
  ## Constructor
  def __init__(self, name='', original_bits_size=0):
    super().__init__(name, original_bits_size)
    
  ## Encode the message (values) using a quantum circuit
  def encode_quantum_message(self):
    encoded_message = []
    for i in range(len(self.values)):
      qc = QuantumCircuit(1, 1)
      if self.values[i] == 1:
        qc.h(0)
      encoded_message.append(qc)
    return encoded_message