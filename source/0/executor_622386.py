# https://github.com/mine260309/qiskit_practice/blob/76c55f733874e2b0865b83104c9e8edea5c45978/executor.py
#!/usr/bin/env python

import sys


def execute(argv, verbose=False):
    from qiskit import QuantumProgram
    # Create the quantum program
    qp = QuantumProgram()

    # Load from filename
    circuit = qp.load_qasm_file(filename)

    # Get qasm source
    source = qp.get_qasm(circuit)
    if verbose:
        print(source)

    # Compile and run
    backend = 'local_qasm_simulator'
    qobj = qp.compile([circuit], backend)  # Compile your program
    result = qp.run(qobj, wait=2, timeout=240)

    if verbose:
        print(result)

    print(result.get_counts(circuit))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s <file>" % sys.argv[0])
    filename = sys.argv[1]
    execute(filename)
