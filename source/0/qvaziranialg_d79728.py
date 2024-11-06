# https://github.com/DanEscher98/SchoolNotes/blob/5bd3220ae5808a8599fbe5aa384d624740f43939/DataScience/Quantum/PyQuantum/QVaziraniAlg.py
from qiskit import *
from qiskit.tools.visualization import plot_histogram
from qiskit.visualization import plot_histogram


def VaziraniAlg(secret):
    n = len(secret)
    circuit = QuantumCircuit(n+1, n)
    circuit.x(n)
    circuit.h(range(n+1))

    circuit.barrier()
    for i, j in enumerate(reversed(s)):
        if j == '1':
            circuit.cx(i, n)

    circuit.barrier()
    circuit.h(range(n+1))
    circuit.barrier()
    circuit.measure(range(n), range(n))
    circuit.draw(filename='Output/QC.png')
    return circuit
    
def Simulator(circuit):
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend=simulator, shots=1).result()
    counts = result.get_counts()
    hist = plot_histogram(result.get_counts(circuit))
    hist.savefig('Output/QC-Hist.png', bbox_inches='tight')
    return counts


if __name__ == '__main__':
    s = input("Set a binary number: ")
    c = VaziraniAlg(s)
    r = Simulator(c)
    print("Your number is: {}".format(r))
