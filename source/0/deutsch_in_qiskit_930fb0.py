# https://github.com/przecze/deutschs_algorithm_in_qiskit_vs_cirq/blob/4e8e9adf2183091ac8d7e03afb2daf8a7d870c6d/deutsch_in_qiskit.py
import qiskit as qs
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator

from numpy import array, sqrt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.extensions.snapshot import Snapshot

from common import to_braket

def get_output_state(circuit, quantum_registers):
    """ returns the braket-formatted output state of a circuit"""
    c = circuit.copy()
    c.append(Snapshot('snap', 'statevector', 2), *quantum_registers)
    sim = QasmSimulator()
    result = sim.run(qs.transpile(c, sim), shots=1).result()
    snap = result.data()['snapshots']['statevector']['snap'][0]
    return to_braket(snap)


regs = [QuantumRegister(2, 'q'), ClassicalRegister(1, 'c')]

# state initialization: apply x on output bit, and H on all
init = QuantumCircuit(*regs)
init.x(1)
init.h(0)
init.h(1)
init.barrier()

# balanced is a c-not - output depends on input
balanced = QuantumCircuit(*regs)
balanced.cx(0,1)
balanced.barrier()

# unity
unbalanced0 = QuantumCircuit(*regs)
unbalanced0.barrier()

# unbalanced is a 'not' - output is fliped independent of input
unbalanced1 = QuantumCircuit(*regs)
unbalanced1.x(1)
unbalanced1.barrier()

# at the end, apply h to input bit and measure it
end = QuantumCircuit(*regs)
end.h(0)
end.measure(0, 0)

sim = QasmSimulator()
# expected: only 1s for balanced, only 0s for unbalanced
for kind, oracle in (('b', balanced), ('u', unbalanced0), ('u', unbalanced1)):
    circuit = init + oracle + end
    transpiled = qs.transpile(circuit)
    counts = sim.run(transpiled).result().get_counts()
    print(kind, counts)
