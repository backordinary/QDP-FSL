# https://github.com/gonyano/ta-readouterr-mtg/blob/deb0b29ce8e94985c43e9ad65c6d9b437bad4de6/src/top.py
import sys
import time
import argparse
import pickle
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import gmres
from qiskit import QuantumCircuit, QuantumRegister, Aer, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.test.mock import FakeProvider

parser = argparse.ArgumentParser(description="")
parser.add_argument("-s", "--shots", help="number of shots for experiment", type=int)
args = parser.parse_args()

qubits = 12
D = 3 # desired Hamming distance
cal_shots = 8192
exp_shots = args.shots
reps = 100
pickle_path = f'../outputs/s{exp_shots}.pickle'

aer_sim = Aer.get_backend('aer_simulator')
device_backend = FakeProvider().get_backend('fake_mumbai')
device_sim = AerSimulator.from_backend(device_backend)

####################
# calibration
qr = QuantumRegister(qubits)
mit_pattern = [[i] for i in range(qubits)]
meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')

t_qc = transpile(meas_calibs, device_sim)
qobj = assemble(t_qc, shots=cal_shots)
cal_results = device_sim.run(qobj, shots=cal_shots).result()

meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
S = np.array(meas_fitter.cal_matrices)
print(S.shape)
####################
# tensored assignment matrix
A = S[0]
for i in range(qubits-1):
    A = np.kron(A, S[i+1])

####################
# experiments
qc = QuantumCircuit(qubits,qubits)
for i in range(qubits):
    qc.x(i)
    qc.h(i)
for i in range(int(qubits/2)):
    qc.ch(6-i,5-i)
for i in range(int(qubits/2)-1):
    qc.ch(6+i,7+i)
qc.measure([i for i in range(qubits)], [i for i in range(qubits)])

t_qc = transpile(qc, aer_sim)
sim_statevector = Aer.get_backend('statevector_simulator')
ideal_results = sim_statevector.run(t_qc).result()
Cideal = ideal_results.get_statevector(t_qc); Cideal = abs(Cideal) ** 2

exp_noisy_list = []; exp_mitigated_inv_list = []; exp_mitigated_m3_list = []
time_inv_list = []; time_m3_list = []
for num in range(reps):
    t_qc = transpile(qc, device_sim)
    results = device_sim.run(t_qc, shots=exp_shots).result()
    noisy_counts = results.get_counts(0)
    Cnoisy = np.zeros(2**qubits)
    for string, value in noisy_counts.items():
        Cnoisy[int(string[::-1],2)] = value
    Cnoisy /= exp_shots

    # mitigation - inverse A
    start = time.time()
    Ainv = la.inv(A)
    Cmitigated_inv = np.dot(Ainv, Cnoisy)
    inv_time = time.time() - start; time_inv_list.append(inv_time)

    # mitigation - M3
    Cnoisy_dense = Cnoisy[Cnoisy!=0]
    original_indices = np.array([i for i in range(Cnoisy.shape[0])]); original_indices = original_indices[Cnoisy!=0]
    A_tilde = A[Cnoisy!=0,:]; A_tilde = A_tilde[:,Cnoisy!=0]
    for i in range(A_tilde.shape[0]):
        for j in range(A_tilde.shape[1]):
            if bin(original_indices[i] ^ original_indices[j]).count('1') <= D:
                continue
            else:
                A_tilde[i,j] = 0
    for j in range(A_tilde.shape[1]):
        A_tilde[:,j] /= sum(A_tilde[:,j])
    P_inv = np.zeros(A_tilde.shape)
    for i in range(P_inv.shape[0]):
        P_inv[i,i] = 1/A_tilde[i,i]
    start = time.time()
    Cmitigated_dense_m3, exitCode = gmres(P_inv @ A_tilde, P_inv @ Cnoisy_dense)
    m3_time = time.time() - start; time_m3_list.append(m3_time)

    Cmitigated_m3 = np.zeros(Cnoisy.shape)
    Cmitigated_m3[original_indices] = Cmitigated_dense_m3

    # expectation value: -0.446 * Z_0
    O = np.ones(2**qubits)
    O[int((2**qubits)/2):] *= -1

    exp_ideal = -0.446 * np.dot(O, Cideal)
    exp_noisy = -0.446 * np.dot(O, Cnoisy); exp_noisy_list.append(exp_noisy)
    exp_mitigated_inv = -0.446 * np.dot(O, Cmitigated_inv); exp_mitigated_inv_list.append(exp_mitigated_inv)
    exp_mitigated_m3 = -0.446 * np.dot(O, Cmitigated_m3); exp_mitigated_m3_list.append(exp_mitigated_m3)

data = [time_inv_list, time_m3_list, exp_noisy_list, exp_mitigated_inv_list, exp_mitigated_m3_list]
with open(pickle_path,'wb') as f:
    pickle.dump(data, f)