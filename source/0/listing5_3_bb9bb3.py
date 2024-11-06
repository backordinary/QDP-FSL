# https://github.com/dsaf2007/QuantumComputing/blob/3655d5d7607dcb64033ba9d0f715c4f53fb87952/HomeWork2/Listing5_3.py
import sys,time,math
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer
from qiskit.visualization import plot_state_city
from qiskit import IBMQ

sys.path.append('../Config/')

def main():
    q = QuantumRegister(3,'q')
    c0 = ClassicalRegister(1,'c0')
    c1 = ClassicalRegister(1,'c1')
    c2 = ClassicalRegister(1,'c2')

    teleport = QuantumCircuit(q,c0,c1,c2)

    teleport.h(q[1])
    teleport.cx(q[1],q[2])

    teleport.ry(np.pi/4,q[0])

    teleport.cx(q[0],q[1])
    teleport.h(q[0])
    teleport.barrier()

    teleport.measure(q[0],c0[0])
    teleport.measure(q[1],c1[0])

    teleport.z(q[2]).c_if(c0,1)
    teleport.x(q[2]).c_if(c1,1)
    
    teleport.measure(q[2],c2[0])



    backend = Aer.get_backend('qasm_simulator')
    shots = 1024
    
    job = backend.run(transpile(teleport,backend),shots)
    result = job.result()
    counts = result.get_counts(teleport)

    print("Counts:" + str(result.get_counts(teleport)))

    # RESULTS
    #ALICE
    alice = {}
    alice['00'] = counts['0 0 0'] + counts['1 0 0']
    alice['10'] = counts['0 1 0'] + counts['1 1 0']
    alice['01'] = counts['0 0 1'] + counts['1 0 1']
    alice['11'] = counts['0 1 1'] + counts['1 1 1']
    plot_histogram(alice,filename='histogram5_3_alice.png')

    #BOB
    bob = {}
    bob['0'] = counts['0 0 0'] + counts['0 1 0'] + counts['0 0 1'] + counts['0 1 1']
    bob['1'] = counts['1 0 0'] + counts['1 1 0'] + counts['1 0 1'] + counts['1 1 1']
    plot_histogram(bob,filename='histogram5_3_bob.png')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

