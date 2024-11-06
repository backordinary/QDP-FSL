# https://github.com/UST-QuAntiL/quantum-circuit-generator/blob/c3c41df4441043f575c61817922a602b38d5c650/app/services/algorithms/grover_algorithm.py
from qiskit import QuantumCircuit
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.circuit.library import GroverOperator


class GroverAlgorithm:
    @classmethod
    def create_circuit(
        cls,
        oracle,
        iterations=1,
        reflection_qubits=None,
        initial_state=None,
        barriers=False,
    ):
        """
        :param oracle: QuantumCircuit object (generated from qasm string)
        :param iterations: how often the amplification is applied
        :param reflection_qubits: integer list of qubits, where diffuser is applied
        :param initial_state: QuantumCircuit object (generated from qasm string), by default a layer of H-gates is applied.
        If reflection_qubits are provided, H-gates are only applied to them and the last qubit is initialized as |0> - |1>.
        :param barriers: boolean flag, wether or not to insert barriers
        :return: OpenQASM Circuit

        Creates the circuit for Grover search algorithm with the specified parameters.
        """

        grover_op = GroverOperator(
            oracle, reflection_qubits=reflection_qubits, insert_barriers=barriers
        )

        # default initial state with H-gate layer and last qubit as H|1>
        if initial_state is None and reflection_qubits is not None:
            init = QuantumCircuit(oracle.num_qubits)
            for i in reflection_qubits:
                init.h(i)
            # initialize last qubit with |0> - |1> state
            init.x(oracle.num_qubits - 1)
            init.h(oracle.num_qubits - 1)
        else:
            init = initial_state

        ap = AmplificationProblem(
            oracle, state_preparation=init, grover_operator=grover_op
        )

        grover = Grover(iterations=iterations).construct_circuit(ap)
        grover = grover.decompose("Q").decompose("Q")

        return grover
