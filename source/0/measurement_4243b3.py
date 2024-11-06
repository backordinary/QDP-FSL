# https://github.com/AllenGabrielMarchen/HHL_implementation/blob/4f2b7f6e2a0eb36440420d88c7adb6cffc37b4f0/circuit/measurement.py
#measurement.py

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram

def measurement(qc,n_l,n_b,CU,backend,shots):
    
    t = transpile(qc, backend)
    qobj = assemble(t, shots=shots)
    results = backend.run(qobj).result()
    answer = results.get_counts()    
    plot_histogram(answer, title="Output Histogram").savefig('./outputs/output_histogram.png',facecolor='#eeeeee')

    return answer