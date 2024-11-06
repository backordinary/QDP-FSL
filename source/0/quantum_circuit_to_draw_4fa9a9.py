# https://github.com/jose/qsmell/blob/7d52e79cbcf1631d11257d01bd35fc488456cb3c/qsmell/utils/quantum_circuit_to_draw.py
#!/usr/bin/env python
#
# ------------------------------------------------------------------------------
# Given a [qiskit.circuit.QuantumCircuit](https://qiskit.org/documentation/apidoc/circuit.html)
# object, this script "pretty print" it as a draw.
#
# Usage example:
#    python quantum_circuit_to_draw.py
#        --module-name <str, e.g., wrapper_adapt_vqe>
#        --output-type <str, i.e., text, latex_source, or mpl>
#        --output-file <path, e.g., adapt_vqe.txt>
# ------------------------------------------------------------------------------

import argparse
import pathlib
import importlib
import sys

from qiskit import transpile

def main():
    parser = argparse.ArgumentParser(description='Convert a quantum circuit object into a matrix.')
    parser.add_argument('--module-name', '-i', help='Module name that has the quantum circuit object `qc`', required=True, type=str)
    parser.add_argument('--output-type', '-t', help='Type of the output: text, latex_source, or mpl', required=True, type=str)
    parser.add_argument('--output-file', '-o', action='store', help='Output file', required=True, type=pathlib.Path)
    args = parser.parse_args()

    module_name: str = args.module_name
    output_type: str = args.output_type
    output_file: str = args.output_file.as_posix()

    wrapper = importlib.import_module(module_name)
    # [Transpile](https://qiskit.org/documentation/apidoc/transpiler.html) the quantum circuit
    # [How can I Transpile a Quantum Circuit?](https://www.youtube.com/watch?v=8mrPNSctRIg)
    #
    # "Transpilation is the process of rewriting a given input circuit to match the topology of
    # a specific quantum device, and/or to optimize the circuit for execution on present day noisy
    # quantum systems."
    #
    qc = transpile(wrapper.qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)
    qc.draw(output=output_type, filename=output_file, justify='left')

    sys.exit(0)

if __name__ == "__main__":
    main()

# EOF
