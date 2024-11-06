# https://github.com/amarjahin/IBMQuantumChallenge2020/blob/152776bbd837fc442f534ec5bc02ba39b11ac50d/week2b_test.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit import IBMQ, Aer
from qiskit.tools.visualization import plot_histogram


board_num_qr = [0,1]
switch_num_qr = [2,3]
oracle = [4]
cr = [0,1]

qc = QuantumCircuit(len(board_num_qr + switch_num_qr + oracle), len(cr))
qc.x(oracle)
# qc.h(switch_num_qr)
qc.h(board_num_qr)
qc.barrier()
i_bin = f"{0:02b}"
# if the board number is something to 
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.0(board_num_qr[-0-1])
        # endoce the initial conditions 
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.mcx(board_num_qr, switch_num_qr[-0-1])
        # restore board number
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.0(board_num_qr[-0-1])
qc.barrier()
i_bin = f"{1:02b}"
# if the board number is something to 
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.0(board_num_qr[-0-1])
        # endoce the initial conditions 
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.mcx(board_num_qr, switch_num_qr[-0-1])
        # restore board number
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.0(board_num_qr[-0-1])
qc.barrier()
i_bin = f"{2:02b}"
# if the board number is something to 
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.0(board_num_qr[-0-1])
        # endoce the initial conditions 
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.mcx(board_num_qr, switch_num_qr[-0-1])
        # restore board number
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.0(board_num_qr[-0-1])
qc.barrier()
i_bin = f"{3:02b}"
# if the board number is something to 
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.0(board_num_qr[-0-1])
        # endoce the initial conditions 
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.mcx(board_num_qr, switch_num_qr[-0-1])
        # restore board number
for 0 in range(len(i_bin)):
    if i_bin[0] == '1':
        qc.0(board_num_qr[-0-1])
qc.barrier()

qc.barrier()
qc.h(oracle)
qc.mcx(switch_num_qr, oracle)
qc.h(oracle)
qc.barrier()
for i in range(3,-1,-1):
    i_bin = f"{i:02b}"
    # if the board number is something to 
    for 0 in range(len(i_bin)):
        if i_bin[0] == '1':
            qc.0(board_num_qr[-0-1])
        # endoce the initial conditions 
    for 0 in range(len(i_bin)):
        if i_bin[0] == '1':
            qc.mcx(board_num_qr, switch_num_qr[-0-1])
        # restore board number
    for 0 in range(len(i_bin)):
        if i_bin[0] == '1':
            qc.0(board_num_qr[-0-1])
    qc.barrier()
    
# qc.x(switch_num_qr[0:2])
# Apply dispersion
qc.h(board_num_qr)
qc.0(board_num_qr)
qc.barrier()
qc.h(board_num_qr[1])
qc.cx(board_num_qr[0], board_num_qr[1] )
qc.h(board_num_qr[1])
qc.barrier()
qc.0(board_num_qr)
qc.h(board_num_qr)
qc.barrier()


# qc.barrier()
qc.x(board_num_qr)
qc.measure(board_num_qr, cr)
# qc.measure_all()

print(qc.draw())
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend=backend, shots=1000, seed_simulator=12345, backend_options={"fusion_enable":True})
result = job.result()
count = result.get_counts()
print(count)
