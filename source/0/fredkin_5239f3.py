# https://github.com/lbadams2/csc_qc/blob/c9894fa587a331e44583717341013abd3a5d6875/hw3/fredkin.py
'''
Liam Adams - lbadams2
'''
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_histogram
from math import pi
import matplotlib.pyplot as plt


# 1/2 in exponent rotates pi/2, 1/4 in exponent rotates pi/4
def fredkin(qc, control, targets):
    qc.rz(pi/4, control[0])
    
    qc.cx(targets[1], targets[0])
    qc.ry(-pi/2, targets[1])
    
    qc.rz(pi/4, targets[0])
    qc.rz(pi/4, targets[1])

    qc.cx(control[0], targets[0])
    qc.cx(targets[0], targets[1])

    qc.rz(-pi/4, targets[0])
    qc.rz(pi/4, targets[1])

    qc.cx(control[0], targets[0])
    qc.cx(targets[0], targets[1])
    qc.cx(control[0], targets[0])

    qc.rz(-pi/4, targets[1])
    qc.cx(targets[0], targets[1])

    qc.rx(pi/2, targets[0])
    qc.rz(-pi/4, targets[1])

    qc.cx(control[0], targets[0])
    qc.cx(targets[0], targets[1])

    qc.rz(pi/2, targets[1])
    qc.rx(pi/2, targets[0])
    qc.rx(-pi/2, targets[1])

    return qc
    

def create_circuit(control, targets, classical, vals):
    qc = QuantumCircuit(control,targets,classical)
    if vals[0]: # control val
        qc.x(control[0])
    if vals[1]: # target 0 val
        qc.x(targets[0])
    if vals[2]: # target 1 val
        qc.x(targets[1])

    qc = fredkin(qc, control, targets)
    qc.measure(control[0],classical[0])
    qc.measure(targets[0],classical[1])
    qc.measure(targets[1],classical[2])
    return qc


def create_circuits():
    control = QuantumRegister(1)  # input
    targets = QuantumRegister(2) # output
    c = ClassicalRegister(3)    
    backend = Aer.get_backend('qasm_simulator')

    first_qc = None
    bin_string = bin(0)[2:].zfill(3)
    vals = [int(b) for b in bin_string]
    qc = create_circuit(control, targets, c, vals)
    if 0 == 0:
        first_qc = qc
    job = execute(qc, backend, shots=100)
    result = job.result()
    print(result.get_counts())
    bin_string = bin(1)[2:].zfill(3)
    vals = [int(b) for b in bin_string]
    qc = create_circuit(control, targets, c, vals)
    if 1 == 0:
        first_qc = qc
    job = execute(qc, backend, shots=100)
    result = job.result()
    print(result.get_counts())
    bin_string = bin(2)[2:].zfill(3)
    vals = [int(b) for b in bin_string]
    qc = create_circuit(control, targets, c, vals)
    if 2 == 0:
        first_qc = qc
    job = execute(qc, backend, shots=100)
    result = job.result()
    print(result.get_counts())
    bin_string = bin(3)[2:].zfill(3)
    vals = [int(b) for b in bin_string]
    qc = create_circuit(control, targets, c, vals)
    if 3 == 0:
        first_qc = qc
    job = execute(qc, backend, shots=100)
    result = job.result()
    print(result.get_counts())
    bin_string = bin(4)[2:].zfill(3)
    vals = [int(b) for b in bin_string]
    qc = create_circuit(control, targets, c, vals)
    if 4 == 0:
        first_qc = qc
    job = execute(qc, backend, shots=100)
    result = job.result()
    print(result.get_counts())
    bin_string = bin(5)[2:].zfill(3)
    vals = [int(b) for b in bin_string]
    qc = create_circuit(control, targets, c, vals)
    if 5 == 0:
        first_qc = qc
    job = execute(qc, backend, shots=100)
    result = job.result()
    print(result.get_counts())
    bin_string = bin(6)[2:].zfill(3)
    vals = [int(b) for b in bin_string]
    qc = create_circuit(control, targets, c, vals)
    if 6 == 0:
        first_qc = qc
    job = execute(qc, backend, shots=100)
    result = job.result()
    print(result.get_counts())
    bin_string = bin(7)[2:].zfill(3)
    vals = [int(b) for b in bin_string]
    qc = create_circuit(control, targets, c, vals)
    if 7 == 0:
        first_qc = qc
    job = execute(qc, backend, shots=100)
    result = job.result()
    print(result.get_counts())

    first_qc.draw(output='mpl')
    plt.show() # big number in qubit figure is "version"


if __name__ == '__main__':
    create_circuits()    