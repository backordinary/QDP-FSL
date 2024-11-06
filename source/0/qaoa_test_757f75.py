# https://github.com/oscarlindbul/Sequence-Alignment/blob/fb3ad94e190692ec9e5cc0b16385fae63ff5d5a2/QAOA/QAOA_test.py
import warnings
warnings.filterwarnings("ignore")
import time

from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor

import numpy as np
from qiskit_aqua import QuantumInstance
from qiskit_aqua.algorithms import QAOA, VQE, ExactEigensolver
from qiskit_aqua.components.optimizers import SPSA, COBYLA
from qiskit_aqua.components.variational_forms import RY
from qiskit_aqua.components.initial_states import Custom as CustomState
from qiskit import BasicAer as Aer
import MSA_column

# load IBM credentials
IBMQ.load_accounts()

backend = "statevector_simulator"
shots = 1024
method = input("Quantum method? ")

sequences = ["CTA", "T"]
costs = [-10, 10, 10]

matchings = MSA_column.get_match_matrix(sequences, costs)

sizes = [len(sequences[i]) for i in range(len(sequences))]
inserts = 0
Hamiltonian_penalty = 100
Hamilt, shift, rev_inds = MSA_column.get_MSA_qubitops(sizes, matchings, gap_pen=costs[2], extra_inserts=inserts, coeffs=[1, Hamiltonian_penalty])
seq_inds = {}
for i in range(len(rev_inds)):
    seq_inds[rev_inds[i]] = i

print("Solving problem with {} qubits".format(Hamilt.num_qubits))

exact_solver = ExactEigensolver(Hamilt)
solution = exact_solver.run()
# exact solution
positions = MSA_column.sample_most_likely(solution["wavefunction"][0], rev_inds)
exact_alignment = MSA_column.get_alignment_string(sequences, inserts, positions)
print("Exact solution: Energy=", solution["eigvals"][0] + shift)
for s in exact_alignment:
    print(s)
print()
# QAOA
if method == "QAOA":
    # initial_state = np.zeros(2**Hamilt.num_qubits)
    # state_bit_arr = np.zeros(Hamilt.num_qubits)
    # for s in range(len(sequences)):
    #     for n in range(len(sequences[s])):
    #         state_bit_arr[seq_inds[(s,n,n)]] = 1
    # print(state_bit_arr)
    # initial_state[int(np.sum(state_bit_arr*np.power(2, np.arange(Hamilt.num_qubits,dtype=np.int32))))] = 1
    # initial_state = CustomState(Hamilt.num_qubits, state_vector=initial_state)

    opt = COBYLA()
    p = 1
    angles = [0,0]
    sol_found = False
    data_file = "QAOA_run2"
    seq_array = np.array(sequences)
    cost_array = np.array(costs)
    coeff_arr = np.array([1, Hamiltonian_penalty])
    insert_vals = np.array(inserts)

    times = []
    energies = []
    angle_arr = []
    states = []
    while not sol_found:
        print("Starting iteration p =", p)
        solver = QAOA(Hamilt, opt, p, initial_point=angles)
        simulator = Aer.get_backend(backend)#IBMQ.get_backend(backend)
        instance = QuantumInstance(backend=simulator, shots=shots)
        start = time.time()
        result = solver.run(instance)
        run_time = time.time() - start

        state = result["eigvecs"][0]
        energy = result["eigvals"][0] + shift
        angles = list(result["opt_params"])
        states.append(state)
        energies.append(energy)
        angle_arr.append(angles)
        times.append(run_time)

        positions = MSA_column.sample_most_likely(state, rev_inds)
        for (key, value) in positions.items():
            print(key, value)

        alignment = MSA_column.get_alignment_string(sequences, inserts, positions)
        print("QAOA solution: Energy=", energy, "params=", angles, "runtime = ", run_time, "seconds")
        sol_found = True
        for (s1,s2) in zip(alignment, exact_alignment):
            if not s1 == s2:
                sol_found = False
                break
        if not sol_found:
            angles.extend([0, 0])
            p += 1
        np.savez(data_file, seqs=seq_array, costs=cost_array, coeffs=coeff_arr, inserts=insert_vals, times=np.array(times), states=np.array(states), angles=np.array(angle_arr), energies=np.array(energies))
    for s in alignment:
        print(s)


# VQE
if method == "VQE":
    opt = SPSA(max_trials=100)
    trial = RY(Hamilt.num_qubits, depth=10, entanglement='linear')
    solver = VQE(Hamilt, trial, opt, "paulis")
    simulator = IBMQ.get_backend(backend)
    instance = QuantumInstance(backend=simulator, shots=1024)
    print("Starting job")
    result = solver.run(instance)
    print("Job finished?")

    state = result["eigvecs"][0]
    energy = result["eigvals"][0] + shift

    positions = MSA_column.sample_most_likely(state, rev_inds)
    for (key, value) in positions.items():
        print(key, value)

    alignment = MSA_column.get_alignment_string(sequences, inserts, positions)
    print("VQE solution: Energy=", energy)
    for s in alignment:
        print(s)
