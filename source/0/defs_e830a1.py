# https://github.com/ayankc/qosf_21/blob/a50bf7596c7a0362ca285645c919018f0e1b3e25/defs.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from IPython.display import Image
from qiskit.visualization import plot_histogram

import numpy as np

def circuit_proc_func(index, index_qubit, dec_val, store_qubit_len, out_index):
    print('index: {}, dec_val: {}, qubit_len: {}'.format(index, dec_val, store_qubit_len))
    backend = Aer.backends(name='qasm_simulator')[0]
    """Preparation of index q-register"""
    qr_index = QuantumRegister(index_qubit)
    """Preparation of stored q-state against each q-index"""
    qr = QuantumRegister(store_qubit_len)
    """Preparation of condition register to be operated"""
    qr_cond = QuantumRegister(store_qubit_len - 1)
    """out condition register to be measured"""
    qr_out = QuantumRegister(1)
    """classical registers for the measurement"""
    cr = ClassicalRegister(index_qubit + 1)
    qc = QuantumCircuit(qr_index, qr, qr_cond, qr_out, cr)

    count = 0
    """converting each input decimal value to binary for stored q-state preparation"""
    while dec_val != 0:
        if (np.bitwise_and(dec_val, 1)) == 1:
            qc.x(qr[count])
        count += 1
        dec_val = np.right_shift(dec_val, 1)

    count = 0
    """main logic"""
    while count < (store_qubit_len - 1):
        qc.cx(qr[count+1], qr[count])
        count += 1

    count = 0
    """put and then look for all condition registers"""
    while count < (store_qubit_len - 1):
        qc.cx(qr[count], qr_cond[count])
        count += 1

    """flip out registers so that if condition is met '0' will be the readout"""
    qc.mct(qr_cond, qr_out)
    qc.x(qr_out)

    qc.measure(qr_out, cr[index_qubit])
    count = 0
    """if out register is 0 then measure all the index registers"""
    while count < index_qubit:
        qc.measure(qr_index[count], cr[count])
        count += 1

    job = execute(qc, backend)
    cur_result = job.result()

    outcome = 0
    """store only the index q-registers when the condition is met"""
    for key in cur_result.get_counts():
        outcome = int(key)
        if outcome < np.left_shift(1, index_qubit):
            out_index.append(index)
    """Sample q-circuit"""
    print(qc.draw(output='text'))
