# https://github.com/ylunaliu/time-crystal/blob/0bb1c8e6fe4f2cd99f63493bff457917a7b633e2/generate_random_para.py
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector
from math import sqrt, pi
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
from create_quantum_circuit import create_quantum_circuit
#define constant
num_bits = 5
num_round = 50
g = 0.6
qubit_to_measure = 3

# Get random initial state
number_of_qubit_1 = np.random.randint(0,num_bits,size=1)
array_qubit1_pos = list(dict.fromkeys(np.random.randint(0,num_bits,size = number_of_qubit_1)))

# Get random interaction for random interaction
random_j = np.random.uniform(-1.5*np.pi, -0.5*np.pi, size = num_bits-1)

# Get random field parameter
random_field = np.random.uniform(-np.pi,np.pi, size = num_bits)
data = []
for i in range(num_round):
    counts = create_quantum_circuit(i+1, g, num_bits, array_qubit1_pos, random_j, random_field)
    keys = list(counts)
    expectation_value_z_qubit = []
    for key in keys:
        bit_status = int(key[qubit_to_measure-1])
        expectation_value_z = bit_status*int(counts[key])
        # print(expectation_value_z)
        expectation_value_z_qubit.append(expectation_value_z)
        # print(expectation_value_z_qubit)

    data.append(sum(expectation_value_z_qubit)/1024)


x = np.arange(0,num_round)
print(x)
print(data)
plt.title("g = {}".format(g) )
plt.xlabel("t")
plt.ylabel("qubit {} <Z>".format(qubit_to_measure))
plt.plot(x, data, '-bo')
plt.show()
