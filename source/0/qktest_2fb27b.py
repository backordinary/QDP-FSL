# https://github.com/entropicalabs/manyQ/blob/0c16a41acec4ec167c9054ab366e8ca3fad41669/speedtests/qktest.py
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
import numpy as np
import time, sys

ftime = time.time

def speed(nbqubits, nb_circuits, repeat=1, depth=2, gpu=False):

    params = np.pi * np.random.rand(depth, nbqubits, nb_circuits)

    start_time = ftime()
    for _ in range(repeat):
        qc_list = []
        for n_c in range(nb_circuits):
            qr = QuantumRegister(nbqubits, 'qr')
            qc = QuantumCircuit(qr)

            for l in range(depth):
                qc.rx(np.pi/2, qr)
                for i in range(nbqubits):
                    qc.rz(params[l, i, n_c], qr[i])
                qc.rx(np.pi/2, qr)
                for i in range(nbqubits - 1):
                    qc.cz(qr[i], qr[i+1])

            qc.rx(np.pi/2, qr)

            qc.measure_all()
            qc_list.append(qc)

        job = execute(qc_list, Aer.get_backend('qasm_simulator'))
        job.result()
        
        
    end_time = ftime()

    return (end_time - start_time)/repeat


if __name__ == '__main__':    
    try: nbqubits = int( sys.argv[1])
    except:    nbqubits = 5
    try: nbcircuits = int( sys.argv[2])
    except: nbcircuits = 10
    try: depth = int( sys.argv[3])
    except: depth = 2
        
    t = speed(nbqubits, nbcircuits, depth=depth)

    print(f"nb qubits={nbqubits}, nb circuits={nbcircuits}, depth={depth}:")
    print(f"milliseconds {t*1000}")
    print(f"     seconds {t}")
    # print(f"     minutes {t/60}")
