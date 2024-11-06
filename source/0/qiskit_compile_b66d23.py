# https://github.com/PythonForChange/FilesFormat/blob/8d28fad30def753a4f6f93a06dfdd7cbcdb31308/qiskit_compile.py
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram,display
circuit=QuantumCircuit(2,2)
circuit.x(2)
circuit.h(0)
circuit.cx(0,1)
circuit.measure(0,1)
s=1024
backend=Aer.get_backend('qasm_simulator')
job=execute(circuit, backend, shots=s)
result=job.result()
counts=result.get_counts(circuit)
graph=plot_histogram(counts)
display(graph)
circuit.draw('mpl')
