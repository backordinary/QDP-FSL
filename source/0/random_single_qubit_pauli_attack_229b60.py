# https://github.com/catproof/Authenticity-Integrity-and-Replay-Protection-in-Quantum-Data-Communications-and-Networking/blob/a677ef719bd886980c0989f3ce5601f274e00e55/random_single_qubit_pauli_attack.py
#This code correspondes to Figure 4.B in "Authenticity, Integrity and Replay Protection in Quantum DataCommunications and Networking"
#Please cite our paper when using this code
#This code automatically makes use of the Gottesman-Knill theorem via the Qiskit API
#https://en.wikipedia.org/wiki/Gottesman%E2%80%93Knill_theorem

import time
import random
from qiskit.quantum_info import StabilizerState, Pauli
import numpy as np
from qiskit import quantum_info

#returns a random pauli composed of a specified maximum number of non-identity single qubit paulis
#'potential_non_identity_paulis' can be specified to choose which qubit positions the single qubit 
#paulis can act on
def random_pauli(size_of_pauli, num_potential_non_identity_paulis = 0, potential_non_identity_paulis = [-1]):
    if num_potential_non_identity_paulis != 0 and potential_non_identity_paulis[0] == -1:
        num_certain_identity_paulis = size_of_pauli - num_potential_non_identity_paulis
        potential_non_identity_paulis = np.array([0] * num_certain_identity_paulis + [1] * num_potential_non_identity_paulis)
        np.random.shuffle(potential_non_identity_paulis)

    if num_potential_non_identity_paulis == 0 and potential_non_identity_paulis[0] == -1:
        potential_non_identity_paulis = [1] * size_of_pauli
    
    indexed_paulis = ["I", "X", "Y", "Z"]
    is_identity_pauli = True
    while is_identity_pauli:
        pauli_string = ""
        if potential_non_identity_paulis[0] != -1:
            for i in range(size_of_pauli):
                pauli_index = random.randint(0, 3)
                if potential_non_identity_paulis[i] == 1:
                    pauli_string = pauli_string + indexed_paulis[pauli_index]
                else:
                    pauli_string = pauli_string + "I"
                if pauli_string[-1] != "I":
                    is_identity_pauli = False
        else:
            for i in range(size_of_pauli):
                pauli_index = random.randint(0, 3)
                pauli_string = pauli_string + indexed_paulis[pauli_index]
                if pauli_string[-1] != "I":
                    is_identity_pauli = False
                
    return Pauli(pauli_string)

def measure_stabilizer_state(stabilizer_state, total_qubits, num_authenticator_bits):
    measurement_result, resulting_state = stabilizer_state.measure(list(range(num_authenticator_bits)))
    if '1' in str(measurement_result):
        return 1
    return 0

start = time.time()


#num_trials = 2
#parameters used in the paper
#num_data_qubits_list = [16, 32, 64]
#num_signature_qubits_list = [2, 4, 8]

num_trials = 100

num_data_qubits_list = [2, 4, 6]
num_signature_qubits_list = [1, 2, 3]
max_total_qubits = max(num_data_qubits_list) + max(num_signature_qubits_list)
num_qubits_to_attack = 1

num_detections_array = np.full([max_total_qubits - 1, max_total_qubits - 1], float("nan"))
for num_data_qubits in num_data_qubits_list:
    for num_signature_qubits in num_signature_qubits_list:
        num_detections = 0
        num_qubits = num_data_qubits + num_signature_qubits
        print("number of data qubits: " + str(num_data_qubits))
        print("number for signature qubits: " + str(num_signature_qubits))
        for i in range(num_trials):
            #initial_state = QuantumCircuit(num_qubits)
            initial_state = StabilizerState(quantum_info.random_clifford(num_qubits))

            encryption_operation = quantum_info.random_clifford(num_qubits)

            adversary_attack = random_pauli(num_qubits,num_qubits_to_attack)

            encrypted_state = initial_state.evolve(encryption_operation)
            attacked_state = encrypted_state.evolve(adversary_attack)
            decrypted_state = attacked_state.evolve(encryption_operation.adjoint())

            if measure_stabilizer_state(decrypted_state, num_qubits, num_signature_qubits):
                num_detections = num_detections + 1
        num_detections_array[num_data_qubits-1,num_signature_qubits-1] = num_detections
        print(str(num_detections/num_trials * 100) + "% of attacks detected\n")
filename = "num_detections_array_for_" + str(num_trials) + "_num_trials_Pauli(" + str(num_qubits_to_attack) + ")"
np.save(filename,num_detections_array)

stop = time.time()

print("Time to complete: " + str(stop - start) + " seconds")