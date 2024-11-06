# https://github.com/dreaming-panda/quantum-ml/blob/360b18fa53bf05108c4b50036ce10553712fd79f/examples/allright/allright_v2.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
from tqdm import tqdm

def verification():
    weight = [[0, 0], [0, 1], [1, 0], [1, 1]]
    data = [[0, 0, 1],[0, 1, 1],[1, 0, 1], [1, 1, 1]]
    accuracy = []
    for w in weight:
        acc = 0
        for d in data:
            model_output = (w[0] * w[1])
            if model_output == d[2]:
                acc += 1
        accuracy.append(acc / 4.0)
    print(accuracy)
verification()
runs = 100
record = []
for p in tqdm(range(runs)):

    n = 6
    m = 6
    qc = QuantumCircuit(n, m)

    weights = [0, 1]
    features = [2, 3]
    label = 4
    readout = 5
    allqubits = [0, 1, 2, 3, 4, 5]
    WeightsAndFeatures = [0, 1, 2, 3]

    for q in WeightsAndFeatures:
        qc.h(q)
    qc.barrier()
    qc.x(label)
    qc.barrier()

    
    for _ in range(p):
        qc.ccx(weights[0], weights[1], readout)
        qc.barrier()
        qc.h(weights[0])
        qc.cz(readout, label)
        qc.h(weights[0])
        qc.barrier()
        qc.h(weights[0])
        qc.x(readout)
        qc.x(label)
        qc.cz(readout, label)
        qc.x(readout)
        qc.x(label)
        qc.h(weights[0])
        qc.barrier()
        qc.ccx(weights[0], weights[1], readout)
        qc.barrier()

        qc.x(label)
        for q in [0, 1, 2, 3]:
                qc.h(q)
        

        for q in [0, 1, 2, 3, 4]:
                qc.x(q)
        qc.barrier()
        qc.h(weights[0])
        qc.mct([1, 2, 3, 4], 0)
        qc.h(weights[0])
        qc.barrier()
        for q in [0, 1, 2, 3, 4]:
                qc.x(q)
                
        for q in [0, 1, 2, 3]:
                qc.h(q)
        qc.x(label)
        
    if p == 6:
       mpl = qc.draw(output='mpl')
       mpl.savefig('qc_allright_v2_plus.jpg')

    qc.measure([weights[0], weights[1]], [weights[0], weights[1]])


    aer_sim = Aer.get_backend('aer_simulator')
    aer_sim.set_options(device='GPU')
    transpiled_grover_circuit = transpile(qc, aer_sim)
    qobj = assemble(transpiled_grover_circuit)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()
    record.append(counts['000011'])

plt.figure(figsize=(12, 12))  
plt.plot(record)  
plt.ylabel('best weight appear times') 
plt.xlabel('grover_iterations') 
plt.title("allright_result_v2")
plt.savefig('allright_result_v2_plus.jpg')
index = []
for i in range(len(record)):
    if record[i] >= 360:
            index.append(i)   
record.sort(reverse=True)
print(record)
print(index)
