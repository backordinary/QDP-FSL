# https://github.com/sofferor/Quantum-Algorithms/blob/ee6678001d7d09e8edbd79c06b356dc12006c173/Simon%20Algorithm/simon.py
import copy
from qiskit import Aer
from qiskit import QuantumCircuit, assemble
from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import simon_oracle


def algorithm(b):
    n = len(b)
    simon_circuit = QuantumCircuit(n * 2, n * 2)

    simon_circuit.h(range(n))

    simon_circuit.barrier()

    simon_circuit += simon_oracle(b)

    simon_circuit.barrier()

    # Measure f(x)
    simon_circuit.measure(range(n, 2*n), range(n, 2*n))

    simon_circuit.barrier()
    # Now the state is "x + (x XOR b)"

    simon_circuit.h(range(n))
    # Now the state is vector "a" such (a dot-product b) = 0 (mod 2)

    simon_circuit.measure(range(n), range(n))
    print(simon_circuit.draw())


    # Run a simulation "shots" times
    aer_sim = Aer.get_backend('aer_simulator')
    shots = 1024
    qobj = assemble(simon_circuit, shots=shots)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()

    # "vectors" are only the measurements of [vector "a" such that (a dot-product b) = 0 (mod 2)] without the measurement of f(x)
    vectors = copy.deepcopy(counts)
    for vec, val in counts.items():
        del vectors[vec]
        vectors[vec[3:]] = val

    plot_histogram(vectors).show()



    # Now need to find "n-1" independent vectors and use Gaussian elimination to solve those equations and find out "b"



    # Assuming we found "b", we calculate the dot product of the vectors with "b" and verify it's indeed equals zero
    def bdotz(b, z):
        accum = 0
        for i in range(len(b)):
            accum += int(b[i]) * int(z[i])
        return (accum % 2)

    for z in vectors:
        print( '{}*{} = {} (mod 2)'.format(b, z, bdotz(b,z)) )


    print('bye')
