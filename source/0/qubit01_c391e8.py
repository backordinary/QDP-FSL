# https://github.com/geophf/logic-programming/blob/fd80570d6f347730cf475c1fd119ab01642a8c15/prolog/quantum/python/qubit01.py
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram
# %matplotlib inline ... only for working in a Jupyter notebook

# create a circuit

def build_circuit():
   circ = QuantumCircuit(3)

   # initialize the circuit and add some gates

   circ.h(0)        # add a Hadamard (H) gate
   circ.cx(0, 1)    # add a CNOT (cx) gate
   circ.cx(0, 2)

   # circ.draw('mpl')    # ... NOT WORKING ??????

   print(circ)           # gets you:

   '''
     ┌───┐          
q_0: ┤ H ├──■────■──
     └───┘┌─┴─┐  │  
q_1: ─────┤ X ├──┼──
          └───┘┌─┴─┐
q_2: ──────────┤ X ├
               └───┘
'''

   return circ

# generic simulator

def simulate_generic(simulator, circuit):
   backend = Aer.get_backend(simulator)

   job = execute(circuit, backend)

   result = job.result()
   return result

# statevector approach

def simulate_sv(circuit):
   result = simulate_generic('statevector_simulator',circuit)
   outputstate = result.get_statevector(circuit, decimals=3)
   print('output vector is',outputstate)

# unitary approach

def simulate_unit(circuit):
   result = simulate_generic('unitary_simulator',circuit)
   unis = result.get_unitary(circuit, decimals=3)
   print('Unitary simulator result run', unis)

# measurement

def measure(circuit):
   meas = QuantumCircuit(3, 3)
   meas.barrier(range(3))
   meas.measure(range(3), range(3))

   qc = circuit + meas
   print('Quantum circuit')
   print(qc)
   return qc

def simulate_qasm(circuit):
   qc = measure(circuit)
   result = simulate_generic('qasm_simulator', qc)
   counts = result.get_counts(qc)
   print(counts)
   # plot_histogram(counts)  # use inside a Jupyter notebook

def main(argv):
   circuit = build_circuit()
   simulate_sv(circuit)
   simulate_unit(circuit)
   simulate_qasm(circuit)

if __name__ == '__main__':
   main([])
