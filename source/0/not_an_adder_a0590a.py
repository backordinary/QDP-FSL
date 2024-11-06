# https://github.com/richardbowmaker/qubits/blob/c21a992f40d394b8a4660a16569073138df1fedb/not_an_adder.py

import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

def not_an_adder():

    print("\nNot an adder")

    n = 8
    n_q = n
    n_b = n
    qc_output = QuantumCircuit(n_q, n_b)

    for j in range(n):
        qc_output.measure(j, j)

    print(qc_output)



    sim = Aer.get_backend('qasm_simulator')  # this is the simulator we'll use
    qobj = assemble(qc_output)  # this turns the circuit into an object our backend can run
    result = sim.run(qobj).result()  # we run the experiment and get the result from that experiment
    # from the results, we get a dictionary containing the number of times (counts)
    # each result appeared
    counts = result.get_counts()
    print("\nTotal counts are:", counts)

    # and display it on a histogram
    #plot_histogram(counts)

    qc_encode = QuantumCircuit(n)
    qc_encode.x(7)
    print(qc_encode)

    qc = qc_encode + qc_output
    print(qc)

    qobj = assemble(qc)
    counts = sim.run(qobj).result().get_counts()
    print("\nTotal counts are:", counts)
    #plot_histogram(counts)




