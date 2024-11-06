# https://github.com/GaiaBolognini/Zero-Noise-Extrapolation/blob/2e4c2a4e6e416e638c1d27c3af20ef092173e566/Projects/Coding/Grover.py
#Implementation of the Grover algorithm for 3 qubits
#Single solution to the search problem: |111>

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, execute
from qiskit import IBMQ, Aer
from qiskit.providers.aer.noise import NoiseModel
import numpy as np
import pandas as pd

#importing the file where the folding functions are defined
import Folding as Folding

prova = 8
n_solutions = 1
solutions = ['111']
N = 8
qubits = 3
df = pd.DataFrame()


###########################################
#DEFINING AN EXECUTOR FUNCTION
def executor(circuit, sim, shots=10 ** 5, noise_model= None):
    """
    Executes the input circuit and returns the noisy expectation value <A>.
    Args: circuit = circuit to compute the exp value
          sim = simulator where to act
          shots = number of trials
          noise_model = noise to be used
    return: expectation value
    """
    # Append measurements
    circuit_to_run = circuit.copy()
    circuit_to_run.measure_all()

    # Run and get counts
    #print(f"Executing circuit with {len(circuit)} gates using {shots} shots.")
    job = execute(circuit_to_run, backend=sim, shots=shots, noise_model=noise_model, optimization_level = 0).result()
    counts = job.get_counts()

    # Compute expectation value of the observable
    exp_value = counts[solutions[0]] / shots
    return exp_value


######################################
#DEFINING THE CIRCUIT

from qiskit.circuit.library import ZGate
#defining the oracle, solutions already known
def oracle_gate(circuit):
    circuit.append(ZGate().control(2), [2,1,0])

    return circuit

from qiskit.circuit.library import MCMT
#defining the Grover gate
def grover_gate(circuit):
    # Hadamard
    for i in range(qubits):
        circuit.h(i)

    # To add a phase at all the solutions except |0>, we create a set of
    # transformations that only |0> won't trigger
    # from qiskit template
    circuit.z([0, 1, 2])

    circuit.cz(2, 1)
    circuit.cz(1, 0)
    circuit.cz(0, 2)
    circuit.append(ZGate().control(2), [2, 1, 0])

    #Hadamard again
    for i in range(qubits):
        circuit.h(i)

    circuit.barrier()
    return circuit

#constructiong the Grover
#applying the Hadamard
circuit = QuantumCircuit(qubits)

for i in range(qubits):
    circuit.h(i)


#defining the number of times we need to apply the Grover's circuit (defined as pi/4 *sqrt(search/solutions))
num_times = int(np.pi/4 *np.sqrt(N/n_solutions))

for i in range(num_times):
    oracle_gate(circuit)
    grover_gate(circuit)

print(circuit)
print('Circuit Depth: ', circuit.depth())
print('Circuit size: ', circuit.size())


'''
print('GATE FOLDING')
circuit_folded_gate = Folding.gate_folding(circuit, scaling = 3, way = 'Random')
print(circuit_folded_gate)
print('Circuit Depth: ', circuit_folded_gate.decompose().decompose().depth())
print('Circuit gate size: ', circuit_folded_gate.size())


#CIRCUIT FOLDING
print('CIRCUIT FOLDING')
circuit_folded_circuit = Folding.circuit_folding(circuit, scaling = 3)
print(circuit_folded_circuit)
print('Circuit Depth: ', circuit_folded_circuit.depth())
print('Circuit circuit size: ', circuit_folded_circuit.size())
'''

#####################################################
#EXECUTE WITH NOISELESS BACKEND
# Set the number of shots
shots = 5*10 ** 5

# Initialize ideal backend (classical noiseless simulator)
ideal_backend = qiskit.Aer.get_backend('qasm_simulator')
ideal_value = executor(circuit, ideal_backend, shots)
print('Ideal_value: ', ideal_value)


###################################################
#EXECUTE WITH NOISY BACKEND
# Select a noisy backend
provider = IBMQ.load_account()

backend = provider.get_backend('ibmq_lima')
noise_model = NoiseModel.from_backend(backend)
simulator = Aer.get_backend('qasm_simulator')
#noisy_value = executor(circuit_folded_circuit, simulator, shots, noise_model=noise_model)
#print('Noisy_value: ', noisy_value)


#definiscilo con la risoluzione che può essere data al tuo sistema in base a d
scales = np.linspace(1, 8, 15)
i = 0

for scale in scales:
    print(scale)
    circuit_folded_gate_left = Folding.gate_folding(circuit, scaling=scale, way='Left')
    value_gate_left = executor(circuit_folded_gate_left, simulator, shots=shots, noise_model=noise_model)

    circuit_folded_gate_right = Folding.gate_folding(circuit, scaling=scale, way='Right')
    value_gate_right = executor(circuit_folded_gate_right, simulator, shots=shots, noise_model=noise_model)

    circuit_folded_gate_random = Folding.gate_folding(circuit, scaling=scale, way='Random')
    value_gate_random = executor(circuit_folded_gate_random, simulator, shots=shots, noise_model=noise_model)

    circuit_folded_circuit = Folding.circuit_folding(circuit, scaling=scale)
    value_circuit = executor(circuit_folded_circuit, simulator, shots=shots, noise_model=noise_model)

    new_row = pd.Series(data={"Ideal":ideal_value,"Scale": scale, "Gate Left": value_gate_left, "Gate Right": value_gate_right,
                              "Gate Random": value_gate_random, "Circuit": value_circuit},
                        name='{}'.format(i))
    i += 1

    df = df.append(new_row, ignore_index=False)

df.to_csv('Results\Grover_{}.csv'.format(prova))
