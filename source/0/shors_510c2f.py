# https://github.com/Stephen-Campbell-UTD/NM_Project_Quantum_Computing/blob/676e0dfe7f04dfe36e687bb95b5479f752f14218/Qiskit/shors.py
#%%
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.visualization.counts_visualization import plot_histogram
from num_theory import modular_inverse
from qclib import n_bit_controlled_modular_multiplier
from qiskit import Aer, QuantumCircuit
from numpy import pi
import math
# %%

"""
#Conventions
0th index qubit is MSB



qpe needs to know 3 things
1.
 t : int :number of "counting" qubits to use 
  For facotring, shor recomends  n^2< 2^t <2n^2 where n is the size of the integer we are factoring

2.
psi : QuantumCircuit;  the eigenstate of the unitary operator
should be |1>
represented by at least the number of bits required to represent n, the integer
we are factoring

3.
U : func(power)->QuantumCircuit; function that generates a Quantum Circuit that
will apply the controlled unitary operation with 0 as control bit 
"""

#%%




def findNumBitsRequired(n) -> int:
  return math.ceil(2*math.log2(n))

def generate_shor_eigenstate(n):
  """
  0th index qubit is MSB
  Generate the number 1 in n qubits
  """
  qc = QuantumCircuit(n)
  qc.x(n-1) 
  return qc


def shors_unitary_generator(a,N):
  """
  U |y> = |a*y (mod N)>

  2.
  """

  def shors_unitary(a: int):
    """
    """
    pass

  return shors_unitary
#%%
def shors(word_size: int, modulus: int, multiplier: int):
  counting_bits = 2*word_size
  unitary_length =n_bit_controlled_modular_multiplier(word_size,modulus,multiplier).num_qubits
  total_qubits = counting_bits + unitary_length
  multiplier_input_size = word_size
  multiplier_input_indices = list(range(counting_bits,counting_bits+multiplier_input_size))
  multiplier_output_size = word_size+1
  multiplier_output_start = counting_bits + multiplier_input_size+word_size
  multiplier_output_indices = list(range(multiplier_output_start, multiplier_output_start + multiplier_output_size))

  multiplier_control_index = counting_bits
  multiplier_indices = list(range(counting_bits+1, total_qubits))


  main = QuantumCircuit(total_qubits)
  counting_indices = list(range(counting_bits))
  for counting_bit_index in range(counting_bits):
    exp_multiplier = multiplier**(2**counting_bit_index)
    exp_multiplier_mod_inverse = modular_inverse(exp_multiplier,modulus)
    main.append(n_bit_controlled_modular_multiplier(word_size,modulus,exp_multiplier),qargs=[counting_bit_index,*multiplier_indices]) 
    for word_bit_index in range(word_size):
      main.cswap(counting_indices[counting_bit_index],multiplier_input_indices[word_bit_index], multiplier_output_indices[word_bit_index] )
    unmult = n_bit_controlled_modular_multiplier(word_size,modulus,exp_multiplier_mod_inverse).inverse()
    main.append(unmult,qargs=[counting_bit_index,*multiplier_indices])
  return main

# %%
shors(3,5,4).draw()

# %%
