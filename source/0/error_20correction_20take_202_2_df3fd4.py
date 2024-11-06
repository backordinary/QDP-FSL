# https://github.com/zademi/QI_project_ichverzweifel/blob/702ea83f9bdd2edd727c1c10e2b43f4327b210bb/Error%20Correction%20Take%202.2.py
from qiskit import IBMQ, QuantumCircuit, execute,  Aer, assemble, transpile, QuantumRegister, ClassicalRegister
from qiskit.result import marginal_counts
from qiskit.providers.ibmq.job import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_vector
import matplotlib.pyplot as plt

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
s_provider = 'ibmq_bogota'
backend = provider.get_backend(s_provider)

# Initialization
n_rounds = 10
plural = False
if n_rounds > 1:
    plural = True
qr = QuantumRegister(5)
cr_data = ClassicalRegister(3)
cr_ancilla_first_round = ClassicalRegister(2)
cr_ancilla_second_round = ClassicalRegister(2)
cr_ancilla_third_round = ClassicalRegister(2)
cr_ancilla_fourth_round = ClassicalRegister(2)
cr_ancilla_fifth_round = ClassicalRegister(2)
cr_ancilla_sixth_round = ClassicalRegister(2)
cr_ancilla_seventh_round = ClassicalRegister(2)
cr_ancilla_eighth_round = ClassicalRegister(2)
cr_ancilla_ninth_round = ClassicalRegister(2)
cr_ancilla_tenth_round = ClassicalRegister(2)
qc_simp = QuantumCircuit(qr, cr_data, cr_ancilla_first_round, cr_ancilla_second_round, cr_ancilla_third_round,
                         cr_ancilla_fourth_round, cr_ancilla_fifth_round, cr_ancilla_sixth_round,
                         cr_ancilla_seventh_round, cr_ancilla_eighth_round, cr_ancilla_ninth_round,
                         cr_ancilla_tenth_round)
qc_simp.h(2)
qc_simp.cx([2, 2, 1, 3, 0, 4], [1, 3, 0, 4, 1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# First round of error detection
# n = 2
# x = 0
# qc_simp.x(2)
# while n <= n_rounds:
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_first_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])
    # x += 2
    # n += 1

# Second round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_second_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Third round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_third_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Fourth round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_fourth_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Fifth round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_fifth_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Sixth round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_sixth_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Seventh round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_seventh_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Eighth round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_eighth_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Ninth round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_ninth_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Tenth round of error detection
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_tenth_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Measurement of the Data Qubits
qc_simp.measure([0, 2, 4], cr_data)

# Plotting of the circuit
# qc_simp.draw('mpl')
# plt.title('The circuit for {} {round} of error correction'.format(n_rounds, round='rounds' if plural else 'round'))
# plt.show()

# Job Executing
simp_job = execute(qc_simp, backend=backend, shots=50, memory=True)
print(simp_job.job_id())
job_monitor(simp_job)

# All the results
data_memory = simp_job.result().get_memory()
print("GHZ memory: ", data_memory)


# Dividing the data qubit measurements and the ancilla qubit measurements into two lists
ancilla_list = []
data_list = []
for measurement in data_memory:
    ancilla_list.append(measurement[:n_rounds * 3])
    data_list.append(measurement[n_rounds * 3:])

print(ancilla_list)
print(data_list)

#plot histogram for GHZ before correction

plt.title('GHZ state', fontsize=20)
plt.ylabel('probabilities', fontsize=15)
plt.hist(data_list, bins=8, rwidth=0.7, normedcolor = 'blue', edgecolor = 'black')
plt.show()

# Performing correction on the data qubits using the ancilla measurements
y = 0
while y < len(data_list):
    i = 0
    while i < n_rounds:
        if (i + 1) >= n_rounds:
            if ancilla_list[y][3 * i:] == '00':
                None
            if ancilla_list[y][3 * i:] == '01':
                if data_list[y][0] == '0':
                    data_list[y] = '1' + data_list[y][1] + data_list[y][2]
                elif data_list[y][0] == '1':
                    data_list[y] = '0' + data_list[y][1] + data_list[y][2]
            if ancilla_list[y][3 * i:] == '11':
                if data_list[y][1] == '0':
                    data_list[y] = data_list[y][0] + '1' + data_list[y][2]
                elif data_list[y][1] == '1':
                    data_list[y] = data_list[y][0] + '0' + data_list[y][2]
            if ancilla_list[y][3 * i:] == '10':
                if data_list[y][2] == '0':
                    data_list[y] = data_list[y][0] + data_list[y][1] + '1'
                elif data_list[y][2] == '1':
                    data_list[y] = data_list[y][0] + data_list[y][1] + '0'
            i += 1
            break
        if ancilla_list[y][3 * i:3 * i + 2] == ancilla_list[y][3 * i + 3:3 * i + 5]:
            i += 2
            continue
        if ancilla_list[y][3 * i:3 * i + 2] == '01' or ancilla_list[y][3 * i + 3:3 * i + 5] == '01':
            if data_list[y][0] == '0':
                data_list[y] = '1' + data_list[y][1] + data_list[y][2]
            elif data_list[y][0] == '1':
                data_list[y] = '0' + data_list[y][1] + data_list[y][2]
        if ancilla_list[y][3 * i:3 * i + 2] == '11' or ancilla_list[y][3 * i + 3:3 * i + 5] == '11':
            if data_list[y][1] == '0':
                data_list[y] = data_list[y][0] + '1' + data_list[y][2]
            elif data_list[y][1] == '1':
                data_list[y] = data_list[y][0] + '0' + data_list[y][2]
        if ancilla_list[y][3 * i:3 * i + 2] == '10' or ancilla_list[y][3 * i + 3:3 * i + 5] == '10':
            if data_list[y][2] == '0':
                data_list[y] = data_list[y][0] + data_list[y][1] + '1'
            elif data_list[y][2] == '1':
                data_list[y] = data_list[y][0] + data_list[y][1] + '0'
        i += 2
    y += 1


# Printing the state of the data qubits after correction
print(data_list)

new_data = data_list

#plot histogram GHZ
plt.title('GHZ state after correction', fontsize=20)
plt.ylabel('probabilities', fontsize=15)
plt.hist(new_data, bins=8, rwidth=0.7, color = 'blue', edgecolor = 'black')
plt.show()







