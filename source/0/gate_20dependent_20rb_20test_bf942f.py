# https://github.com/ace314/RB_2-qubit/blob/939d0156368b145606d93c1b34b61f911b449b0b/RB_1q/gate%20dependent%20RB%20test.py
import matplotlib.pyplot as plt
import multiprocessing as mp
import copy
from scipy.optimize import curve_fit
from qiskit import quantum_info
from lib.oneqrb import *


# noisy_clifford_list : Gate dependent noisy Clifford
def RB_single_sequence(l, rho_initial, noisy_clifford_list):
    np.random.seed()
    cliff_seq = np.random.choice(24, l[-1], replace=True)

    seq = []
    for i in range(len(cliff_seq)):
        m = noisy_clifford_list[cliff_seq[i]]
        seq.append(m)

    f = np.zeros(len(l))
    rho = copy.deepcopy(rho_initial)

    for i in range(len(seq)):
        rho = seq[i] @ rho @ seq[i].conj().T
        if i+1 in l:
            inv = get_seq_inverse(cliff_seq[:(i+1)])
            rho_inversed = inv @ rho @ inv.conj().T
            fidelity = abs(np.trace(rho_initial @ rho_inversed))
            j = l.index(i+1)
            f[j] += fidelity
    return f


# Fitting function
def func(x, p, A, B):
    return A * p ** x + B


delta1 = 0.9
delta2 = 0.1
dephasing_unitary1 = np.cos(delta1) * I_1q - 1j * np.sin(delta1) * Z_1q
dephasing_unitary2 = np.cos(delta2) * I_1q - 1j * np.sin(delta2) * Z_1q

noise_unitary = []

for i in range(12):
    noise_unitary.append(dephasing_unitary1)
for i in range(12):
    noise_unitary.append(dephasing_unitary2)


# for i in range(24):
#     noise_unitary.append(dephasing_unitary2)

F_hamiltonian = []
for i in range(len(noise_unitary)):
    ch = quantum_info.Operator(get_perfect_cliff(i) @ noise_unitary[i] @ get_perfect_cliff(i).conj().T)
    F_ave = quantum_info.average_gate_fidelity(ch)
    F_hamiltonian.append(F_ave)

Clifford_list = []
for i in range(len(noise_unitary)):
    Clifford_list.append(get_perfect_cliff(i) @ noise_unitary[i])

L = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
rho_0 = np.array([[1, 0],
                  [0, 0]])
rep = 200

if __name__ == '__main__':
    result_list = []

    def log_result(result):
        result_list.append(result)

    pool = mp.Pool()
    for re in range(rep):
        pool.apply_async(RB_single_sequence, args=(L, rho_0, Clifford_list), callback=log_result)
    pool.close()
    pool.join()
    F = sum(result_list) / rep

    popt, pcov = curve_fit(func, L, F, p0=[1, 1, 1], bounds=(0, 1), maxfev=5000)
    F_Clifford = (popt[0] + 1) / 2
    print("Theoretical Hamiltonain noise average gate fidelity: ", np.mean(F_hamiltonian))
    print("Experimental Hamiltonain noise average gate fidelity: ", F_Clifford)


