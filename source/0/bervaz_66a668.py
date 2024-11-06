# https://github.com/theheavygluon/berVaz/blob/5c03630002ddf23cc6ebfcb1d03b41448870ab98/berVaz.py
from qiskit import *
from qiskit.tools.visualization import plot_histogram
from matplotlib import pyplot as plt

#The sim function simulates the circuit on the qasm_simulator and gives 2 outputs: the circuit
# and the result of the state measurement 

def sim(n, plotTool='text'):
    N = [int(x) for x in str(n)]
    circuit = QuantumCircuit(len(n)+1,len(n))
    circuit.h([i for i in range(len(n))])
    circuit.x(len(n))
    circuit.h(len(n))

    circuit.barrier()


    i = 0
    while i < len(n):
        if N[i] == 1:
            circuit.cx(len(N) - (i+1),len(n))
        i+=1
        
    circuit.barrier()
    circuit.h([i for i in range(len(n))])
    circuit.barrier()
    circuit.measure([i for i in range(len(n))],[i for i in range(len(n))])
    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit,backend, shots = 1).result()
    output = result.get_counts()
    return circuit.draw('text'), output
   


def run(n,plotTool='text',device='ibmq_16_melbourne'):
    N = [int(x) for x in str(n)]
    circuit = QuantumCircuit(len(n)+1,len(n))
    circuit.h([i for i in range(len(n))])
    circuit.x(len(n))
    circuit.h(len(n))

    circuit.barrier()


    i = 0
    while i < len(n):
        if N[i] == 1:
            circuit.cx(len(N) - (i+1),len(n))
        i+=1
        
    circuit.barrier()
    circuit.h([i for i in range(len(n))])
    circuit.barrier()
    circuit.measure([i for i in range(len(n))],[i for i in range(len(n))])
    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q')
    qcomp = provider.get_backend(device)
    job = execute(circuit, backend=qcomp)
    from qiskit.tools.monitor import job_monitor
    job_monitor(job)
    result = job.result()
    return circuit.draw('text'), result, circuit

