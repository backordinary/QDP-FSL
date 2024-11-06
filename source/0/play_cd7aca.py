# https://github.com/Stephen-Campbell-UTD/NM_Project_Quantum_Computing/blob/676e0dfe7f04dfe36e687bb95b5479f752f14218/Qiskit/play.py
#%%

from typing import Iterable
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.compiler.transpiler import transpile
from qiskit.visualization import plot_bloch_multivector
from qclib import qft
import warnings

# from tutorials.qft_playground import qft
warnings.filterwarnings("ignore")
#%%

def get_binary_number_representation(qc : QuantumCircuit,start =0 ,stop=-1,  *args, **kwargs):
  sim = Aer.get_backend("aer_simulator")
  qc_to_plot =qc.copy()
  qc_to_plot.save_statevector()
  statevector = sim.run(transpile(qc_to_plot)).result().get_statevector()

  indices = np.where(statevector == 1)
  assert(len(indices) == 1)
  num =  indices[0][0]
  # convert to binary -> truncate 0b -> reverse to qubit order 
  # -> slice start,stop -> reverse to num order -> cast to int
  # print(num, bin(num))
  stop = qc.num_qubits if stop == -1 else stop
  # print(bin(num)[2:][::-1][start:stop])#[::-1])
  # print(start,stop)
  num = int(bin(num)[2:][::-1][start:stop][::-1],base=2)
  return num

def not_qubits_from_num(qc,num : int, indices : Iterable = None):
  indices = range(qc.num_qubits) if indices is None else indices
  indices = list(indices)
  word_size = len(indices)
  num_as_bits = bin(num)[2:].zfill(word_size)[::-1]
  for i in range(word_size):
    should_not = num_as_bits[i] == '1'
    if should_not:
      qc.x(indices[i])

#%%
qc = QuantumCircuit(8)
not_qubits_from_num(qc,2,range(2,9))
# get_binary_number_representation(qc,2,9)
get_binary_number_representation(qc)


# %%
qft(3).draw()
# %%
