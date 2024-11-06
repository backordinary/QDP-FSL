# https://github.com/IndigoChild88/Quantum-Computing/blob/501d1507b5d2a8ce26f3b9ca6e7ed97b9a4bb235/Qskit/First_quantum.py
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 01:00:24 2019

@author: acn00
"""

import sys
import qiskit
import logging
from qiskit import QuantumProgram
# Main sub
def main():
  # create a  program
  qp = QuantumProgram()
  # create 1 qubit
  quantum_r = qp.create_quantum_register("qr", 1)
  print("It works")
   # create 1 classical register
  classical_r = qp.create_classical_register("cr", 1)
    # create a circuit
  qp.create_circuit("Circuit", [quantum_r], [classical_r])
  # get the circuit by name
  circuit = qp.get_circuit('Circuit')
  # enable logging
  qp.enable_logs(logging.DEBUG);
  # Pauli X gate to qubit 1 in the Quantum Register "qr"
  circuit.x(quantum_r[0])
  # measure gate from qubit 0 to classical bit 0
  circuit.measure(quantum_r[0], classical_r[0])
  # backend simulator
  backend = 'local_qasm_simulator'
  # Group of circuits to execute
  circuits = ['Circuit']
  # Compile your program
  qobj = qp.compile(circuits, backend)
  # run in simulator
  result = qp.run(qobj, timeout=240)
  # Show result counts
  print (str(result.get_counts('Circuit')))
  
  
  
if __name__ ==  '__main__':
  main()