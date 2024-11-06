# https://github.com/ashishar/quantum_inner_product/blob/537dffb8a553b16436483bdaf59828eec09e0c4f/quantum_inner_product.py
# A quantum subroutine for speeding up HYPMIX
# by Ashish K S Arya, ashishk1@hotmail.com

# A quantum subroutine for computationally complex inner product calculation is proposed.
# This is an implementation of the proposed method in Qiskit. In order to run this you must have an account
# on IBM Quantum platform quantum-computing.ibm.com, thereafter you can run from inside IBM Quantum Lab. It's free.

# For this specific code the input vectors x,y (with norms < 1) assumed to be sent from the classical routine


# TO USE THE QUANTUM INNER PRODUCT IMPORT/LOAD THIS FILE AND THEN CALL quantum_inner_product(vector1,vector2).
# THIS FUNCTION RETURNS THE INNER PRODUCT OF vector1 AND vector2

# Import necessary libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, assemble, transpile
from qiskit import execute, BasicAer
from qiskit.visualization import plot_state_qsphere, plot_histogram

from itertools import zip_longest
import numpy as np


# The vector sent by classical routine are the inputs to the quantum sub routine.
# However since can only load normalized vectors into a quantum circuit we need to
# introduce a dummy variable to make the norm of x and y equalt to 1.

def add_dummy_variables(vector1, vector2):
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    dummy_variable1 = np.sqrt(1 - norm1 ** 2)
    dummy_variable2 = np.sqrt(1 - norm2 ** 2)

    vector1 = np.append(vector1, dummy_variable1)
    vector2 = np.append(vector2, dummy_variable2)

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    return vector1, vector2


# Let's define the function to build the quantum vectors to load the inputs into quantum circuit
# number of qubits required is ceil of log2 (length of vector)
# Further if length of vector in not a power of two then we pad
# the input vectors with zeros to feed it into the quantum circuit,
# padding with zeros does not change the norm.

# Since any quantum state will have 2^n possibilities, if our vector is less than 2^n for a suitable n we will pad it with zeros

def make_quantum_vector(vector1d, vector2d):
    length_vector = len(vector1d)

    # is it 2^n
    n = np.log2(length_vector)

    num_qubits = int(np.ceil(n))
    # print("number of qubits: ",num_qubits)

    # To embed the given vectors vector1 and vector2 we create two random vectors using Statevector class of size num_qubits,
    # multiply them with zero and then add the values of vector1 and vector2 on these, padding will automatically come in this way

    dummy_vector1 = [0] * 2 ** num_qubits

    # Lets call the vector after padding as quantum_vector1 and quantum_vector2
    quantum_vector1 = [x + y for x, y in zip_longest(dummy_vector1, vector1d, fillvalue=0)]
    quantum_vector2 = [x + y for x, y in zip_longest(dummy_vector1, vector2d, fillvalue=0)]

    return quantum_vector1, quantum_vector2, num_qubits


# Now we have two proper quantum vectors ,let's define the function to create
# a quantum circuit to upload these quantum vectors
# We need to initialize a quantum circuit of num_qubits with the quantum vector 1

def initializer_circuit(quantum_vector1, quantum_vector2, num_qubits):
    qc_quantum_vector1 = QuantumCircuit(num_qubits, name='vector1')
    qc_quantum_vector1.initialize(quantum_vector1)
    qc_quantum_vector1 = transpile(qc_quantum_vector1, basis_gates=['u', 'cx'])
    qc_quantum_vector1 = qc_quantum_vector1.control(1, ctrl_state='0')

    # We need to initialize a quantum circuit of num_qubits with the quantum vector 2
    qc_quantum_vector2 = QuantumCircuit(num_qubits, name='vector2')
    qc_quantum_vector2.initialize(quantum_vector2)
    qc_quantum_vector2 = transpile(qc_quantum_vector2, basis_gates=['u', 'cx'])
    qc_quantum_vector2 = qc_quantum_vector2.control(1, ctrl_state='1')

    return qc_quantum_vector1, qc_quantum_vector2


# Lets define a function to generate the main quantum circuit with the qubit_0 as control qubit
# this qubit will be measured to get the estimate inner product] and add the initializer circuits to this.

def get_main_circuit(qc_quantum_vector1, qc_quantum_vector2, num_qubits):
    inner_product_circuit = QuantumCircuit(num_qubits + 1, 1, name='inner product circuit')

    # We need to get the control qubit [first qubit] in super position

    inner_product_circuit.h(0)
    inner_product_circuit.append(qc_quantum_vector1, inner_product_circuit.qubits)
    inner_product_circuit.append(qc_quantum_vector2, inner_product_circuit.qubits)

    # We need to get the control qubit [first qubit] in super position [again] after adding the ininiatization circuits
    inner_product_circuit.h(0)

    # We need to add the measurement to first_qubit
    inner_product_circuit.measure([0], 0)

    # inner_product_circuit = inner_product_circuit.decompose()
    # inner_product_circuit.draw('mpl')
    return inner_product_circuit


# lets conduct measurement of the first qubit and get the probabilities

def measure_step(inner_product_circuit, repetitions):
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(inner_product_circuit, backend, shots=repetitions)
    result = job.result()
    counts = result.get_counts(inner_product_circuit)
    return counts


# Now we have defined all the required functions, lets call them

def quantum_inner_product(vector1, vector2):
    vector1d, vector2d = add_dummy_variables(vector1, vector2)
    q_vector1, q_vector2, num_qubits = make_quantum_vector(vector1d, vector2d)
    qc_vector1, qc_vector2 = initializer_circuit(q_vector1, q_vector2, num_qubits)
    inner_product_circuit = get_main_circuit(qc_vector1, qc_vector2, num_qubits)
    repetitions = 20000
    counts = measure_step(inner_product_circuit, repetitions)
    p = counts['0'] / (counts['0'] + counts['1'])

    estimated_inner_product_with_dummy = 2 * p - 1

    estimated_inner_product_without_dummy = estimated_inner_product_with_dummy - vector1d[-1] * vector2d[-1]
    # estimated_inner_product_without_dummy is the answer we need
    return estimated_inner_product_without_dummy


# Sample call
# vector1=[0.38461261, 0.08546947, 0.35612279, 0.09021777, 0.24216349, 0.2469118,
#          0.38936091, 0.11395929, 0.25640841, 0.05223134, 0.15669403, 0.30863975,
#          0.43209565, 0.16144233]
# vector2=[0.36723377, 0.01224113, 0.29786739, 0.22442064, 0.02040188, 0.21625989,
#          0.07344675, 0.34275152, 0.40395715, 0.30194777, 0.26522439, 0.01224113,
#          0.16321501, 0.38763565]

# print(vector1,"\n",vector2)
# estimated_inner_product_without_dummy=quantum_inner_product(vector1,vector2)
# print('\nEstimated_inner_product ',estimated_inner_product_without_dummy,'numpy inner product: ', np.inner(vector1,vector2))
