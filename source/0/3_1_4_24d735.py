# https://github.com/dsaf2007/QuantumComputing/blob/08e3dcc6f9210f4605e6eb1996b3534e43326141/3_1_4.py
import sys,time
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer
from qiskit.visualization import plot_state_city
from qiskit import IBMQ


def qrng():
 qreg_q = QuantumRegister(5, 'q')
 creg_c = ClassicalRegister(5, 'c')
 circuit = QuantumCircuit(qreg_q, creg_c)

 circuit.h(qreg_q[0])
 circuit.h(qreg_q[1])
 circuit.h(qreg_q[2])
 circuit.measure(qreg_q[0], creg_c[0])
 circuit.measure(qreg_q[1], creg_c[1])
 circuit.measure(qreg_q[2], creg_c[2])

 backend = Aer.get_backend('qasm_simulator')
 shots = 1024
 job_sim = backend.run(transpile(circuit,backend),shots)
 result_sim = job_sim.result()
 counts = result_sim.get_counts(circuit)
 print(counts)

 bits = ""
 for v in counts.values():
    if v > shots/(2**3) :
         bits += "1"
    else:
        bits += "0"
 return int(bits,2)


if __name__ == '__main__':
    start_time = time.time()
    numbers = [] 
    size = 10

    for i in range(size):
        n = qrng()
        numbers.append(n)

    print("list=" + str(numbers))
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #f = open('3_4.txt','w')
    #print(str(numbers),file=f)
    #f.close()