# https://github.com/jose/qsmell/blob/7d52e79cbcf1631d11257d01bd35fc488456cb3c/qsmell/utils/quantum_circuit_to_matrix.py
#!/usr/bin/env python
#
# ------------------------------------------------------------------------------
# Given a [qiskit.circuit.QuantumCircuit](https://qiskit.org/documentation/apidoc/circuit.html)
# object, this script "pretty print" it as a coverage matrix.  Each row represents
# a quantum or classical bit, each column represents a timestamp in the circuit,
# and each cell represents a quantum operation performed in circuit.
#
# Usage example:
#    from quantum_circuit_to_matrix import Justify, qc2matrix
#
#    from qiskit import QuantumCircuit, QuantumRegister
#    reg = QuantumRegister(3, name='reg')
#    qc = QuantumCircuit(reg)
#    qc.x(reg[0])
#    qc.x(reg)
#    qc.x(reg[1])
#    qc.x(reg[0])
#    qc.x(reg[2])
#    qc.x(reg[1])
#
#    qc2matrix(qc, Justify.none, 'example.csv')
#
# Which prints out to the stdout the following dataframe
#            1    2    3    4    5    6    7    8
# q-reg-0  x()  x()  x()
# q-reg-1                 x()  x()  x()
# q-reg-2                                x()  x()
# 
# and to the provided output file the following matrix
# ;1;2;3;4;5;6;7;8
# q-reg-0;x();x();x();;;;;
# q-reg-1;;;;x();x();x();;
# q-reg-2;;;;;;;x();x()
#
# Or using its command line version as:
#    python quantum_circuit_to_matrix.py
#        --module-name <str, e.g., wrapper_adapt_vqe>
#        [--justify <str, i.e., "left" or "none">]
#        [--transpile <bool, transpile the circuit, if enable>]
#        --output-file <path, e.g., adapt_vqe.csv>
# ------------------------------------------------------------------------------

import argparse
import pathlib
import importlib
import sys
import enum

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Barrier
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit

# ------------------------------------------------------------ Utility functions

class Justify(enum.Enum):
    left = 'left'
    none = 'none'

    def __str__(self):
        return self.value

def extract_qubit_id(qubit: Qubit) -> str:
    quantumRegister: QuantumRegister = qubit.register
    id = 'q-%s-%d' % (quantumRegister.name, qubit.index)
    return(id)

def extract_clbit_id(clbit: Clbit) -> str:
    classicalRegister: ClassicalRegister = clbit.register
    id = 'c-%s-%d' % (classicalRegister.name, clbit.index)
    return(id)

def extract_op_id(operation: Instruction) -> str:
    params_types = []
    for param in operation.params:
        params_types.append(str(type(param).__name__))
    id = '%s(%s)' % (operation.name, ','.join(params_types))
    return(id)

# ------------------------------------------------------------- Module functions

def noalignqc2matrix(qc: QuantumCircuit) -> pd.DataFrame:
    # Collect QuantumCircuit's data
    qubits = qc.qubits
    clbits = qc.clbits
    qdata  = qc.data

    # Initialize matrix where the number of rows is equal to the number of qubits +
    # number of clbits and the number of columns is equal to the number of operations
    # in the circuit.  By default, each cell is initialized as False, as no qubit or
    # clbit has been involved in any operation.

    nbits  = len(qubits) + len(clbits) # Number of rows
    depth  = len(qdata) # Number of columns == number of operations
    matrix = np.empty((nbits, depth), dtype=str)

    col_names = range(1, len(qdata)+1)
    row_names = []
    # Collect qubits' names
    for qubit in qubits:
        row_names.append(extract_qubit_id(qubit))
    # Collect clbits' names
    for clbit in clbits:
        row_names.append(extract_clbit_id(clbit))

    # Initialize 'matrix' as a dataframe and name rows and columns accordingly
    df = pd.DataFrame(matrix, columns=col_names, index=row_names)

    # Populate dataframe with operations' data
    for index in col_names:
        circuitInstruction: CircuitInstruction = qdata[index-1]

        # Operation
        operation: Instruction = circuitInstruction.operation
        op_name = '%s' % (extract_op_id(operation))

        # In some qubit(s) and/or clbit(s)
        op_qubits = circuitInstruction.qubits
        for op_qubit in op_qubits:
            df[index][extract_qubit_id(op_qubit)] = op_name
        op_clbits = circuitInstruction.clbits
        for op_clbit in op_clbits:
            df[index][extract_clbit_id(op_clbit)] = op_name

    return(df)

def leftqc2matrix(qc: QuantumCircuit) -> pd.DataFrame:
    # Collect QuantumCircuit's data
    qubits = qc.qubits
    clbits = qc.clbits

    # All bits
    bits = qubits + clbits

    # Initialize matrix where the number of rows is equal to the number of qubits +
    # number of clbits and the number of columns is equal to the depth of the circuit.
    # By default, each cell is initialized as False, as no qubit or clbit has been
    # involved in any operation before analyzing it.

    nbits  = len(bits) # Number of rows
    depth  = qc.depth(filter_function=lambda x: True) # Number of columns
    matrix = np.empty((nbits, depth), dtype=str)

    col_names = range(1, depth+1) # Referenced as levels later in the code
    row_names = []
    # Collect qubits' names
    for qubit in qubits:
        row_names.append(extract_qubit_id(qubit))
    # Collect clbits' names
    for clbit in clbits:
        row_names.append(extract_clbit_id(clbit))

    # Initialize 'matrix' as a dataframe and name rows and columns accordingly
    df = pd.DataFrame(matrix, columns=col_names, index=row_names)

    # ----------
    # The following code is based on Qiskit's depth function:
    #     https://github.com/Qiskit/qiskit-terra/blob/0.21.1/qiskit/circuit/quantumcircuit.py#L1941
    # also described in [here](https://quantumcomputing.stackexchange.com/a/5772)

    # Assign each bit in the circuit a unique integer
    # to index into op_stack.
    # - bit_indices = {bit: idx for idx, bit in enumerate(self.qubits + self.clbits)}
    bit_indices = {bit: idx for idx, bit in enumerate(bits)}

    # If no bits, return 0
    if not bit_indices:
        return 0

    # A list that holds the height of each qubit and classical bit.
    op_stack = [0] * len(bit_indices)

    # Here we are playing a modified version of
    # Tetris where we stack gates, but multi-qubit
    # gates, or measurements have a block for each
    # qubit or cbit that are connected by a virtual
    # line so that they all stacked at the same depth.
    # Conditional gates act on all cbits in the register
    # they are conditioned on.
    # The max stack height is the circuit depth.
    for instruction in qc.data:
        print(instruction) # debug
        levels = []
        reg_ints = []
        for ind, reg in enumerate(instruction.qubits + instruction.clbits):
            # Add to the stacks of the qubits and
            # cbits used in the gate.
            reg_ints.append(bit_indices[reg])
            # - if filter_function(instruction):
            # -     levels.append(op_stack[reg_ints[ind]] + 1)
            # - else:
            # -     levels.append(op_stack[reg_ints[ind]])
            # Count all instructions including non _directive instructions (e.g., barriers)
            levels.append(op_stack[reg_ints[ind]] + 1)
        # Assuming here that there is no conditional
        # snapshots or barriers ever.
        if instruction.operation.condition:
            # Controls operate over all bits of a classical register
            # or over a single bit
            if isinstance(instruction.operation.condition[0], Clbit):
                condition_bits = [instruction.operation.condition[0]]
            else:
                condition_bits = instruction.operation.condition[0]
            for cbit in condition_bits:
                idx = bit_indices[cbit]
                if idx not in reg_ints:
                    reg_ints.append(idx)
                    levels.append(op_stack[idx] + 1)

        max_level = max(levels)
        for ind in reg_ints:
            op_stack[ind] = max_level
    # ----------

        #
        # Populate dataframe with operations' data
        #

        # Operation
        operation: Instruction = instruction.operation
        op_name = '%s' % (extract_op_id(operation))

        for index in reg_ints:
            level = op_stack[index]
            bit   = bits[index]
            if isinstance(bit, Qubit):
                df[level][extract_qubit_id(bit)] = op_name
            elif isinstance(bit, Clbit):
                df[level][extract_clbit_id(bit)] = op_name
            else:
                raise Exception('Unknown type of bit: ' + str(type(bit)))

        print(op_stack) # debug
        print(df) # debug

    return(df)

def qc2matrix(qc: QuantumCircuit, justify: Justify, output_file_path: str) -> None:
    df = None
    if justify == Justify.none:
        df = noalignqc2matrix(qc)
    elif justify == Justify.left:
        df = leftqc2matrix(qc)
    else:
        raise Exception('Unknown justify option ' + str(justify))

    sys.stdout.write(str(df) + '\n')
    df.to_csv(output_file_path, header=True, index=True, sep=';', mode='w')

# ------------------------------------------------------------------------- Main

def main():
    parser = argparse.ArgumentParser(description='Convert a quantum circuit object into a matrix.')
    parser.add_argument('--module-name', '-i', help='Module name that has the quantum circuit object `qc`', required=True, type=str)
    parser.add_argument('--justify', '-j', help='`left` (default) or `none`. It refers to where operations should be placed in the output circuit matrix. `none` results in each operation being placed in its own column.', required=False, type=Justify, choices=list(Justify), default=Justify.left)
    parser.add_argument('--transpile-circuit', '-t', help='Transpile the circuit, if enable', required=False, action='store_true', default=False)
    parser.add_argument('--output-file', '-o', action='store', help='Output file', required=True, type=pathlib.Path)
    args = parser.parse_args()

    module_name: str        = args.module_name
    justify: Justify        = args.justify
    transpile_circuit: bool = args.transpile_circuit
    output_file: str        = args.output_file.as_posix()

    wrapper = importlib.import_module(module_name)
    qc = wrapper.qc
    # [Transpile](https://qiskit.org/documentation/apidoc/transpiler.html) the quantum circuit
    # [How can I Transpile a Quantum Circuit?](https://www.youtube.com/watch?v=8mrPNSctRIg)
    #
    # "Transpilation is the process of rewriting a given input circuit to match the topology of
    # a specific quantum device, and/or to optimize the circuit for execution on present day noisy
    # quantum systems."
    #
    if transpile_circuit:
        qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)

    # Process it
    qc2matrix(qc, justify, output_file)

    sys.exit(0)

if __name__ == "__main__":
    main()

# EOF
