# https://github.com/electricalgorithm/BernsteinVaziraniAlgorithm/blob/5d8d88790ed1db3c2e5a64de34552c86c2e0cd2d/bernsteinvazirani.py
"""
This module implements the Bernstein-Vazirani Algorithm.
"""

from qiskit import Aer
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit.instruction import Instruction


class BVAlgorithm:
    """This class implements the Bernstein-Vazirani Algorithm."""

    @staticmethod
    def create_oracle_from_number(number: int) -> Instruction:
        """This method returns a Oracle block which will be used in
        the Bernstein-Vazirani Algorithm.

        Parameters
        ----------
        number_str : str
            The number in binary form as string.

        Returns
        -------
        Instruction
            Oracle block to be fed into the algorithm.
        """
        # Convert the integer into binary string.
        number_str = format(number, "b")
        inputs_count = len(number_str)

        # Create a quantum circuit with the number of input qubits + 1 output qubit.
        oracle = QuantumCircuit(inputs_count + 1, inputs_count)

        # Apply the CNOTs to the inputs as "1"s.
        for index, qubit in enumerate(reversed(number_str)):
            if qubit == "1":
                oracle.cx(index, inputs_count)

        inst = oracle.to_instruction()
        inst.name = "SecretNumberOracle"
        return inst

    @staticmethod
    def simulate(secret_number_oracle: Instruction) -> dict:
        """_summary_

        Parameters
        ----------
        secret_no_oracle : Instruction
            The secret number to look for with the algoritm.

        Returns
        -------
        dict
            A dictionary with result attribute which is the found number.
        """
        # Create the circuit.
        circuit = BVAlgorithm._construct_the_circuit(secret_number_oracle)

        # Run the simulation.
        aer_sim = Aer.get_backend("aer_simulator")
        transpiled_dj_circuit = transpile(circuit, aer_sim)
        qobj = assemble(transpiled_dj_circuit)
        results = aer_sim.run(qobj).result()

        # Get the answer.
        answer = results.get_counts()
        answer_as_list = list(answer.keys())
        answer_int = int(answer_as_list[0], 2)

        return {"result": answer_int}

    @staticmethod
    def _construct_the_circuit(function_block: QuantumCircuit) -> QuantumCircuit:
        """It creates the circuit for the Bernstein-Vazirani Algorithm.

        Parameters
        ----------
        function_block : QuantumCircuit
            The secret number block to check with the Bernstein-Vazirani Algorithm.

        Returns
        -------
        QuantumCircuit
            The circuit for the Bernstein-Vazirani Algorithm.
        """
        # Get the number of input qubits.
        input_length = function_block.num_qubits - 1

        _circuit = QuantumCircuit(input_length + 1, input_length)

        # Apply Hadamard gates to all input qubits.
        for qubit in range(input_length):
            _circuit.h(qubit)

        # Convert the last qubit to |-) state.
        _circuit.x(input_length)
        _circuit.h(input_length)
        _circuit.barrier()

        # Apply the oracle block.
        _circuit.append(
            function_block,
            range(function_block.num_qubits),
            range(function_block.num_clbits),
        )
        _circuit.barrier()

        # Apply Hadamard gates to all input qubits.
        for qubit in range(input_length):
            _circuit.h(qubit)
        _circuit.barrier()

        # Measure all input qubits and put them to classical bits.
        for qubit in range(input_length):
            _circuit.measure(qubit, qubit)

        return _circuit


if __name__ == "__main__":
    print("========================================")
    print("Bernstein-Vazirani Algorithm Simulation")
    print("========================================")

    # Get the number of input qubits.
    secret_number = int(input("> Enter the secret number to search for: "))

    # Get the oracle block.
    block_to_test = BVAlgorithm.create_oracle_from_number(secret_number)

    # Run the algorithm.
    result = BVAlgorithm.simulate(block_to_test)
    print(f"Result: {result['result']}")
