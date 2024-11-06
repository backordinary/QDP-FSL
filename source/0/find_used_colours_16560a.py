# https://github.com/drobiu/quantum-project/blob/26a2a49cabb0868c9bf5ad68e6861062da7ad6ea/src/logic/find_used_colours.py
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, BasicAer, execute
from qiskit.circuit.library import XGate

from src.arithmetic.comparator import comparator
from src.arithmetic.counter import mincount, count
from src.logic.oracles import oracle_a, oracle_b
from src.logic.query import query


def find_used_colours(n_positions, n_colors, secret_string):
    n = n_positions
    k = n_colors
    s = secret_string

    a = QuantumRegister(3)
    b = QuantumRegister(4)
    c = QuantumRegister(8)
    d = QuantumRegister(3)
    e = QuantumRegister(3)
    f = QuantumRegister(1)
    g = QuantumRegister(1)
    h = QuantumRegister(1)
    cl = ClassicalRegister(4)

    circuit = QuantumCircuit(a, b, c, d, e, f, g, h, cl)

    # Step 1
    circuit = oracle_a(circuit, c, a, s)
    # qc.barrier()

    # Step 2
    circuit.h(b[:])
    # qc.barrier()

    # Step 3
    circuit = query(circuit, b, c, s)
    # qc.barrier()

    # Step 4
    circuit = oracle_b(circuit, c, d, s)
    # qc.barrier()

    # Step 5
    circuit.x(e[-1])
    circuit = mincount(circuit, e, b)
    # qc.barrier()

    # Step 6
    circuit = comparator(circuit, a, e, f, 3)
    # qc.barrier()

    # Step 7
    circuit.x(g)
    circuit.x(a[:])
    multiply_controlled_x(circuit, a, g)
    circuit.x(a[:])
    # qc.barrier()

    # Step 8
    circuit.x(d[0], h)
    multiply_controlled_x(circuit, [e[0], f, g], h)
    circuit.x(f)
    multiply_controlled_x(circuit, [e[0], f, g], h)
    multiply_controlled_x(circuit, [b[0], f, g], h)
    circuit.x(f)
    # qc.barrier()

    # Step 9
    circuit.z(h)
    # qc.barrier()

    # Step 10
    circuit.x(f)
    multiply_controlled_x(circuit, [b[0], f, g], h)
    multiply_controlled_x(circuit, [e[0], f, g], h)
    circuit.x(f)
    multiply_controlled_x(circuit, [e[0], f, g], h)
    circuit.cx(d[0], h)
    circuit.x(a[:])
    circuit.cx(a[:], g)
    circuit.x(a[:])
    circuit.x(g)
    circuit = comparator(circuit, a, e, f, 3)
    circuit = count(circuit, e, b)
    circuit.x(e[2])
    circuit = oracle_b(circuit, c, d, s, do_inverse=True)
    circuit = query(circuit, b, c, s)

    #  Step 11
    circuit.h(b[:])
    # qc.barrier()

    #  Step 12
    circuit.measure(b[:], cl[:])

    # circuit.draw(output='text')

    print("\nResult from the local Qiskit simulator backend:\n")
    backend = BasicAer.get_backend("qasm_simulator")
    job = execute(circuit, backend=backend, shots=1)
    result = job.result()
    print(result.get_counts(circuit))

    return circuit


def multiply_controlled_x(circuit, control, qubit):
    num_c = len(control)

    cx = XGate().control(num_c)
    circuit.append(cx, [*control, qubit])


if __name__ == "__main__":
    qc = find_used_colours(4, 4, [2, 2, 2, 2])
