# https://github.com/lbadams2/csc_qc/blob/c9894fa587a331e44583717341013abd3a5d6875/hw3/phase.py
'''
Liam Adams - lbadams2
'''
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_histogram
from qiskit import IBMQ
from math import pi
import matplotlib.pyplot as plt

IBMQ.save_account('')

def create_circuits_reuse():
    q_in = QuantumRegister(2)  # input
    c = ClassicalRegister(2)
    
    qc_cx = QuantumCircuit(q_in, c)
    qc_cx.h(q_in[0])
    qc_cx.x(q_in[1])
    qc_cx.cx(q_in[0], q_in[1])
    qc_cx.measure(q_in[0], c[0])
    qc_cx.measure(q_in[1], c[1])

    qc_cu = QuantumCircuit(q_in, c)
    qc_cu.h(q_in[0])
    qc_cu.x(q_in[1])
    qc_cu.cu1(-pi/2, q_in[0], q_in[1])
    qc_cu.measure(q_in[0], c[0])
    qc_cu.measure(q_in[1], c[1])
    
    return qc_cx, qc_cu


def create_circuits(add_h):
    q_in = QuantumRegister(2)  # input
    qc_cx = QuantumCircuit(q_in)
    qc_cx.h(q_in[0])
    qc_cx.x(q_in[1])
    if add_h:
        qc_cx.h(q_in[1])
    qc_cx.cx(q_in[0], q_in[1])

    q_in = QuantumRegister(2)  # input
    qc_cu = QuantumCircuit(q_in)
    qc_cu.h(q_in[0])
    qc_cu.x(q_in[1])
    if add_h:
        qc_cu.h(q_in[1])
    qc_cu.cu1(-pi/2, q_in[0], q_in[1])

    q_in = QuantumRegister(2)  # input
    c = ClassicalRegister(2)
    qc_cx_m = QuantumCircuit(q_in, c)
    qc_cx_m.h(q_in[0])
    qc_cx_m.x(q_in[1])
    if add_h:
        qc_cx_m.h(q_in[1])
    qc_cx_m.cx(q_in[0], q_in[1])
    qc_cx_m.measure(q_in[0], c[0])
    qc_cx_m.measure(q_in[1], c[1])

    q_in = QuantumRegister(2)  # input
    c = ClassicalRegister(2)
    qc_cu_m = QuantumCircuit(q_in, c)
    qc_cu_m.h(q_in[0])
    qc_cu_m.x(q_in[1])
    if add_h:
        qc_cu_m.h(q_in[1])
    qc_cu_m.cu1(-pi/2, q_in[0], q_in[1])
    qc_cu_m.measure(q_in[0], c[0])
    qc_cu_m.measure(q_in[1], c[1])
    
    return qc_cx, qc_cu, qc_cx_m, qc_cu_m


def run_reuse_regs():
    qc_cx, qc_cu = create_circuits_reuse()
    backend = Aer.get_backend('statevector_simulator')
    
    job = execute(qc_cx, backend)  # shots default = 1024
    result = job.result()
    outputstate = result.get_statevector(qc_cx, decimals=3)
    print(outputstate)
    print('\n\n')
    
    job = execute(qc_cu, backend)  # shots default = 1024
    result = job.result()
    outputstate = result.get_statevector(qc_cu, decimals=3)
    print(outputstate)
    print('\n\n')

    backend = Aer.get_backend('qasm_simulator')
    
    job = execute(qc_cx, backend, shots=500)  # shots default = 1024
    result = job.result()
    print(result.get_counts())
    print('\n\n')

    job = execute(qc_cu, backend, shots=500)  # shots default = 1024
    result = job.result()
    print(result.get_counts())


def run(add_h):
    qc_cx_state, qc_cu_state, qc_cx_m, qc_cu_m = create_circuits(add_h)
    backend = Aer.get_backend('statevector_simulator')
    
    job = execute(qc_cx_state, backend)  # shots default = 1024
    result = job.result()
    outputstate = result.get_statevector(qc_cx_state, decimals=3)
    print(outputstate)
    print('\n\n')
    
    job = execute(qc_cu_state, backend)  # shots default = 1024
    result = job.result()
    outputstate = result.get_statevector(qc_cu_state, decimals=3)
    print(outputstate)
    print('\n\n')

    backend = Aer.get_backend('qasm_simulator')
    
    job = execute(qc_cx_m, backend, shots=500)  # shots default = 1024
    result = job.result()
    print(result.get_counts())
    print('\n\n')

    job = execute(qc_cu_m, backend, shots=500)  # shots default = 1024
    result = job.result()
    print(result.get_counts())
    print('\n\n')

    
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmq_ourense')
    
    job = execute(qc_cx_m, backend, shots=500)  # shots default = 1024
    result = job.result()
    print(result.get_counts())
    print('\n\n')

    job = execute(qc_cu_m, backend, shots=500)  # shots default = 1024
    result = job.result()
    print(result.get_counts())

    qc_cx_m.draw(output='mpl')
    qc_cu_m.draw(output='mpl')
    plt.show()


if __name__ == '__main__':
    #run_reuse_regs()
    add_h = True
    run(add_h)