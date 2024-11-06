# https://github.com/dsaf2007/QuantumComputing/blob/6ad24dc66b736b9a593068996506c94d07ab3007/HomeWork2/Listing5_2.py
import sys,time,math
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer
from qiskit.visualization import plot_state_city
from qiskit import IBMQ

sys.path.append('../Config/')

def main():
    q = QuantumRegister(2,'q')
    c = ClassicalRegister(2,'c')
    superdense = QuantumCircuit(q,c)

    q_program = QuantumProgram()

    superdense.h(q[0])
    superdense.cx(q[0],q[1])

    superdense.z(q[0])
    superdense.x(q[0])
    superdense.barrier()

    superdense.cx(q[0],q[1])
    superdense.h(q[0])
    superdense.measure(q[0],c[0])
    superdense.measure(q[1],c[1])

    #print(superdense.get_qasms[0])

    backend = Aer.get_backend('qasm_simulator')
    shots = 1024
    
    job = backend.run(transpile(superdense,backend),shots)
    result = job.result()
    counts = result.get_counts(superdense)

    print("Counts:" + str(result.get_counts(superdense)))

    plot_histogram(counts,filename='histogram5_2.png')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print
    print("--- %s seconds ---" % (time.time() - start_time))

