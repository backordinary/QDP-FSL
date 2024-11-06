# https://github.com/dreaming-panda/quantum-ml/blob/360b18fa53bf05108c4b50036ce10553712fd79f/examples/xor/xor_strengthen.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
from tqdm import tqdm
def verification():
    weight = [[0, 0], [0, 1], [1, 0], [1, 1]]
    data = [[0, 0, 0],[0, 1, 1],[1, 0, 1], [1, 1, 0]]
    accuracy = []
    for w in weight:
        acc = 0
        for d in data:
            model_output = (w[0] * d[0]) ^ (w[1] * d[1])
            if model_output == d[2]:
                acc += 1
        accuracy.append(acc / 4.0)
    print(accuracy)
verification()
record = []
runs = 100
for p in tqdm(range(runs)):
    n = 10
    m = 10
    qc = QuantumCircuit(n, m)
    weights = [0, 1]
    features = [[2, 3], [6, 7]]
    labels = [4, 8]
    readout = [5, 9]

    for q in weights:
        qc.h(q)
    qc.barrier()

    for q in features[0]:
        qc.h(q)
    qc.barrier()
    qc.x(labels[0])
    qc.ccx(features[0][0], features[0][1], labels[0])
    qc.x(features[0][0])
    qc.x(features[0][1])
    qc.ccx(features[0][0], features[0][1], labels[0])
    qc.x(features[0][0])
    qc.x(features[0][1])
    qc.barrier()

    for q in features[1]:
        qc.h(q)
    qc.barrier()
    qc.x(labels[1])
    qc.ccx(features[1][0], features[1][1], labels[1])
    qc.x(features[1][0])
    qc.x(features[1][1])
    qc.ccx(features[1][0], features[1][1], labels[1])
    qc.x(features[1][0])
    qc.x(features[1][1])
    qc.barrier()


    for _ in range(p):
                """
                oracle
                """
                qc.ccx(weights[0], features[0][0], readout[0])
                qc.ccx(weights[1], features[0][1], readout[0])
                qc.barrier()
                qc.ccx(weights[0], features[1][0], readout[1])
                qc.ccx(weights[1], features[1][1], readout[1])
                qc.barrier()
                qc.h(labels[0])
                qc.mct([readout[1], labels[1], readout[0]], labels[0])
                qc.h(labels[0])
                qc.barrier()
                qc.x(labels[0])
                qc.x(labels[1])
                qc.x(readout[0])
                qc.x(readout[1])
                qc.h(labels[0])
                qc.mct([readout[1], labels[1], readout[0]], labels[0])
                qc.h(labels[0])
                qc.x(labels[0])
                qc.x(labels[1])
                qc.x(readout[0])
                qc.x(readout[1])
                qc.barrier()
                qc.x(labels[0])
                qc.x(readout[0])
                qc.h(labels[0])
                qc.mct([readout[1], labels[1], readout[0]], labels[0])
                qc.h(labels[0])
                qc.x(labels[0])
                qc.x(readout[0])
                qc.barrier()
                qc.x(labels[1])
                qc.x(readout[1])
                qc.h(labels[0])
                qc.mct([readout[1], labels[1], readout[0]], labels[0])
                qc.h(labels[0])
                qc.x(labels[1])
                qc.x(readout[1])
                qc.barrier()

                qc.ccx(weights[1], features[1][1], readout[1])
                qc.ccx(weights[0], features[1][0], readout[1])
                qc.barrier()
                qc.ccx(weights[1], features[0][1], readout[0])
                qc.ccx(weights[0], features[0][0], readout[0])
                qc.barrier()
                """
                diffusion
                """
                qc.barrier()
                qc.x(features[1][1])
                qc.x(features[1][0])
                qc.ccx(features[1][0], features[1][1], labels[1])
                qc.x(features[1][1])
                qc.x(features[1][0])
                qc.ccx(features[1][0], features[1][1], labels[1])
                qc.x(labels[1])
                qc.barrier()
                for q in features[1]:
                    qc.h(q)
                qc.barrier()
                qc.x(features[0][1])
                qc.x(features[0][0])
                qc.ccx(features[0][0], features[0][1], labels[0])
                qc.x(features[0][1])
                qc.x(features[0][0])
                qc.ccx(features[0][0], features[0][1], labels[0])
                qc.x(labels[0])
                qc.barrier()
                for q in features[0]:
                    qc.h(q)
                for q in weights:
                    qc.h(q)
                qc.barrier()



                qc.barrier()
                for q in [0, 1, 2, 3, 4, 6, 7, 8]:   
                    qc.x(q)
                qc.barrier()
                qc.h(weights[0])
                qc.mct([1, 2, 3, 4, 6, 7, 8], 0)
                qc.h(weights[0])
                qc.barrier()
                for q in [0, 1, 2, 3, 4, 6, 7, 8]:
                    qc.x(q)
                qc.barrier()


                for q in weights:
                    qc.h(q)
                qc.barrier()

                for q in features[0]:
                    qc.h(q)
                qc.barrier()
                qc.x(labels[0])
                qc.ccx(features[0][0], features[0][1], labels[0])
                qc.x(features[0][0])
                qc.x(features[0][1])
                qc.ccx(features[0][0], features[0][1], labels[0])
                qc.x(features[0][0])
                qc.x(features[0][1])
                qc.barrier()

                for q in features[1]:
                    qc.h(q)
                qc.barrier()
                qc.x(labels[1])
                qc.ccx(features[1][0], features[1][1], labels[1])
                qc.x(features[1][0])
                qc.x(features[1][1])
                qc.ccx(features[1][0], features[1][1], labels[1])
                qc.x(features[1][0])
                qc.x(features[1][1])
                qc.barrier()

                
    if p == 6:
        mpl = qc.draw(output='mpl')
        mpl.savefig('qc_xor_v2_plus.jpg'.format(p))
    qc.measure([weights[0], weights[1]], [weights[0], weights[1]])

    aer_sim = Aer.get_backend('aer_simulator')
    aer_sim.set_options(device='GPU')
    transpiled_qc = transpile(qc, aer_sim)
    qobj = assemble(transpiled_qc)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()
    if '0000000011' not in counts.keys():
         counts['0000000011'] = 0
    record.append(counts['0000000011'])

plt.figure(figsize=(12, 12))  
plt.plot(record)  
plt.ylabel('best weight appear times') 
plt.xlabel('grover_iterations') 
plt.title("xor_result_v2")
plt.savefig('xor_strengthen.jpg')
index = []
for i in range(len(record)):
    if record[i] >= 280:
            index.append(i)   
record.sort(reverse=True)
print(record)
print(index)

    