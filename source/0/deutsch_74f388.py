# https://github.com/whnbaek/QC2020/blob/256bd6889de01f117fe8aa7730d1bfcc0414f8bd/hw6/deutsch.py
from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
from qiskit.visualization import plot_histogram

circuit = QuantumCircuit(9, 8)

circuit.x(0)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.h(4)
circuit.h(5)
circuit.h(6)
circuit.h(7)
circuit.h(8)
circuit.barrier()

# oracle
circuit.x(8)
for i in range(8, 0, -1):
    circuit.cx(i, i - 1)
circuit.cx(1 + 1, 1)
circuit.cx(2 + 1, 2)
circuit.cx(3 + 1, 3)
circuit.cx(4 + 1, 4)
circuit.cx(5 + 1, 5)
circuit.cx(6 + 1, 6)
circuit.cx(7 + 1, 7)
circuit.x(8)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.h(4)
circuit.h(5)
circuit.h(6)
circuit.h(7)
circuit.h(8)
circuit.barrier()

circuit.measure(range(1, 9), range(8))

simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots = 1000)

result = job.result()
counts = result.get_counts(circuit)
print("Total counts: ", counts)

draw = circuit.draw(output = 'mpl')
draw.savefig('deutsch_circuit.png', bbox_inches = 'tight')
hist = plot_histogram(counts)
hist.savefig('deutsch_histogram.png', bbox_inches = 'tight')
