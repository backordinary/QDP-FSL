# https://github.com/qis-unipr/dqc-circuit/blob/cc887a47f2f52a08229f24e9b40661b296cb431e/distributed_circuit/dist_circuit.py
import warnings

import pygments
from pygments.formatters.terminal256 import Terminal256Formatter
from qiskit import QuantumRegister
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit, HAS_PYGMENTS
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.qasm import OpenQASMLexer, QasmTerminalStyle, QasmError

from dqc_parser import ast_to_dag
from .dist_qasm import DistQasm


class DistQuantumCircuit(QuantumCircuit):

    def __init__(self, *regs, name=None, global_phase=0, metadata=None):
        super().__init__(*regs, name=name, global_phase=global_phase, metadata=metadata)

    @staticmethod
    def from_qasm_file(path):
        """Take in a QASM file and generate a QuantumCircuit object.
        Args:
          path (str): Path to the file for a QASM program
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM
        """
        qasm = DistQasm(filename=path)
        return _circuit_from_qasm(qasm)

    @staticmethod
    def from_qasm_str(qasm_str):
        """Take in a QASM string and generate a QuantumCircuit object.
        Args:
          qasm_str (str): A QASM program string
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM
        """
        qasm = DistQasm(data=qasm_str)
        return _circuit_from_qasm(qasm)

    def etnswap(self, *qargs):
        from .instructions import EntSwapInstr

        qubits = []

        if not qargs:  # None
            qubits.extend(self.qubits)

        for qarg in qargs:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, range):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        return self.append(EntSwapInstr(len(qubits)), qubits, [])

    def remote_cx(self, *qargs):
        from .instructions import RemoteCxInstr

        qubits = []

        if not qargs:  # None
            qubits.extend(self.qubits)

        for qarg in qargs:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, range):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        return self.append(RemoteCxInstr(), qubits, [])

    def epr(self, *qargs):
        from .instructions import EPRInstr

        qubits = []

        if not qargs:  # None
            qubits.extend(self.qubits)

        for qarg in qargs:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, range):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        return self.append(EPRInstr(), qubits, [])

    def qasm(self, formatted=False, filename=None, encoding=None):
        """Return OpenQASM string.
        Args:
            formatted (bool): Return formatted Qasm string.
            filename (str): Save Qasm to file with name 'filename'.
            encoding (str): Optionally specify the encoding to use for the
                output file if ``filename`` is specified. By default this is
                set to the system's default encoding (ie whatever
                ``locale.getpreferredencoding()`` returns) and can be set to
                any valid codec or alias from stdlib's
                `codec module <https://docs.python.org/3/library/codecs.html#standard-encodings>`__
        Returns:
            str: If formatted=False.
        Raises:
            MissingOptionalLibraryError: If pygments is not installed and ``formatted`` is
                ``True``.
            QasmError: If circuit has free parameters.
        """
        from qiskit.circuit.controlledgate import ControlledGate

        if self.num_parameters > 0:
            raise QasmError("Cannot represent circuits with unbound parameters in OpenQASM 2.")

        existing_gate_names = [
            "ch",
            "cp",
            "cx",
            "cy",
            "cz",
            "crx",
            "cry",
            "crz",
            "ccx",
            "cswap",
            "csx",
            "cu",
            "cu1",
            "cu3",
            "dcx",
            "h",
            "i",
            "id",
            "iden",
            "iswap",
            "ms",
            "p",
            "r",
            "rx",
            "rxx",
            "ry",
            "ryy",
            "rz",
            "rzx",
            "rzz",
            "s",
            "sdg",
            "swap",
            "sx",
            "x",
            "y",
            "z",
            "t",
            "tdg",
            "u",
            "u1",
            "u2",
            "u3",
            "epr",
            "remoteCx",
            "entswap",
        ]

        existing_composite_circuits = []

        string_temp = self.header + "\n"
        string_temp += self.extension_lib + "\n"
        for register in self.qregs:
            string_temp += register.qasm() + "\n"
        for register in self.cregs:
            string_temp += register.qasm() + "\n"

        qreg_bits = {bit for reg in self.qregs for bit in reg}
        creg_bits = {bit for reg in self.cregs for bit in reg}
        regless_qubits = []
        regless_clbits = []

        if set(self.qubits) != qreg_bits:
            regless_qubits = [bit for bit in self.qubits if bit not in qreg_bits]
            string_temp += "qreg %s[%d];\n" % ("regless", len(regless_qubits))

        if set(self.clbits) != creg_bits:
            regless_clbits = [bit for bit in self.clbits if bit not in creg_bits]
            string_temp += "creg %s[%d];\n" % ("regless", len(regless_clbits))

        unitary_gates = []

        bit_labels = {
            bit: "%s[%d]" % (reg.name, idx)
            for reg in self.qregs + self.cregs
            for (idx, bit) in enumerate(reg)
        }

        bit_labels.update(
            {
                bit: "regless[%d]" % idx
                for reg in (regless_qubits, regless_clbits)
                for idx, bit in enumerate(reg)
            }
        )

        for instruction, qargs, cargs in self._data:
            if instruction.name == "measure":
                qubit = qargs[0]
                clbit = cargs[0]
                string_temp += "{} {} -> {};\n".format(
                    instruction.qasm(),
                    bit_labels[qubit],
                    bit_labels[clbit],
                )

            # If instruction is a root gate or a root instruction (in that case, compositive)

            elif instruction.name == "epr":
                qubits = qargs
                temp = "{} ".format(instruction.qasm())
                for qubit in qubits:
                    temp += "{},".format(bit_labels[qubit])
                temp = temp[:-1]
                temp += ";\n"
                string_temp += temp
                # print(temp)

            elif instruction.name == "remoteCx":
                qubits = qargs
                # clbits = cargs
                temp = "{} ".format(instruction.qasm())
                for qubit in qubits:
                    temp += "{},".format(bit_labels[qubit])
                temp = temp[:-1]
                # temp += " -> "
                # for cbit in clbits:
                #     temp += "{},".format(bit_labels[cbit])
                # temp = temp[:-1]
                temp += ";\n"
                string_temp += temp
                # print(temp)

            elif instruction.name == "entswap":
                qubits = qargs
                # clbits = cargs
                temp = "{} ".format(instruction.qasm())
                for qubit in qubits:
                    temp += "{},".format(bit_labels[qubit])
                temp = temp[:-1]
                # temp += " -> "
                # for cbit in clbits:
                #     temp += "{},".format(bit_labels[cbit])
                # temp = temp[:-1]
                temp += ";\n"
                string_temp += temp
                # print(temp)

            elif (
                    type(instruction)
                    in [
                        Gate,
                        Instruction,
                    ]
                    or (isinstance(instruction, ControlledGate) and instruction._open_ctrl)
            ):
                if instruction not in existing_composite_circuits:
                    if instruction.name in existing_gate_names:
                        old_name = instruction.name
                        instruction.name += "_" + str(id(instruction))

                        warnings.warn(
                            "A gate named {} already exists. "
                            "We have renamed "
                            "your gate to {}".format(old_name, instruction.name)
                        )

                    # Get qasm of composite circuit
                    qasm_string = self._get_composite_circuit_qasm_from_instruction(instruction)

                    # Insert composite circuit qasm definition right after header and extension lib
                    string_temp = string_temp.replace(
                        self.extension_lib, f"{self.extension_lib}\n{qasm_string}"
                    )

                    existing_composite_circuits.append(instruction)
                    existing_gate_names.append(instruction.name)

                # Insert qasm representation of the original instruction
                string_temp += "{} {};\n".format(
                    instruction.qasm(),
                    ",".join([bit_labels[j] for j in qargs + cargs]),
                )
            else:
                string_temp += "{} {};\n".format(
                    instruction.qasm(),
                    ",".join([bit_labels[j] for j in qargs + cargs]),
                )
            if instruction.name == "unitary":
                unitary_gates.append(instruction)

        # this resets them, so if another call to qasm() is made the gate def is added again
        for gate in unitary_gates:
            gate._qasm_def_written = False

        if filename:
            with open(filename, "w+", encoding=encoding) as file:
                file.write(string_temp)
            file.close()

        if formatted:
            if not HAS_PYGMENTS:
                raise MissingOptionalLibraryError(
                    libname="pygments>2.4",
                    name="formatted QASM output",
                    pip_install="pip install pygments",
                )
            code = pygments.highlight(
                string_temp, OpenQASMLexer(), Terminal256Formatter(style=QasmTerminalStyle)
            )
            return None
        else:
            return string_temp

    def decompose(self):
        """Call a decomposition pass on this circuit,
        to decompose one level (shallow decompose).

        Returns:
            QuantumCircuit: a circuit one level decomposed
        """
        # pylint: disable=cyclic-import
        from qiskit.transpiler.passes.basis.decompose import Decompose
        from qiskit.converters.circuit_to_dag import circuit_to_dag
        from dqc_parser.dag_to_dist_circuit import dag_to_dist_circuit
        pass_ = Decompose()
        decomposed_dag = pass_.run(circuit_to_dag(self))
        return dag_to_dist_circuit(decomposed_dag)


def _circuit_from_qasm(qasm):
    # pylint: disable=cyclic-import
    from dqc_parser.dag_to_dist_circuit import dag_to_dist_circuit

    ast = qasm.parse()
    # print(ast.qasm())
    dag = ast_to_dag(ast)
    return dag_to_dist_circuit(dag)
