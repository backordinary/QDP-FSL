# https://github.com/electricalgorithm/QuantumFourierTransform/blob/96c56d7d35c192582e7c3f45dde56fca9a32bdf1/quantum_fourier.py
"""
This module contains the Quantum Fourier Transform class.
"""

from numpy import pi
from qiskit import QuantumCircuit, Aer, transpile, assemble


class QuantumFourierTransform:
    """Class for operations of the Quantum Fourier Transform."""

    @staticmethod
    def simulate(state: int) -> dict:
        """Simulate the QFT and iQFT.

        Parameters
        ----------
            state (int): The state to simulate.

        Returns
        -------
            dict: The simulated state.
        """
        # Get the number of qubits.
        qubit_count = state.bit_length()

        # Create the circuit.
        circuit = QuantumCircuit(qubit_count, qubit_count)

        # Apply the initial state.
        for qubit in range(qubit_count):
            if state & (1 << qubit):
                circuit.x(qubit)

        # Apply the QFT.
        circuit = QuantumFourierTransform.qft(circuit)

        # Apply the inverse QFT.
        circuit = QuantumFourierTransform.iqft(circuit)

        # Append the measurement.
        circuit.measure(range(qubit_count), range(qubit_count))

        # Run the simulation.
        simulator = Aer.get_backend("aer_simulator")
        circuit = transpile(circuit, simulator)
        job = simulator.run(assemble(circuit))
        result = job.result().get_counts()

        answer_as_list = list(result.keys())
        answer_int = int(answer_as_list[0], 2)
        return {"result": answer_int}

    @staticmethod
    def qft(circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply QFT to a circuit.

        Parameters
        ----------
            circuit (QuantumCircuit):  The circuit to apply the QFT to.

        Returns
        -------
            QuantumCircuit: The circuit with the QFT applied.
        """
        # Apply the QFT to the circuit.
        circuit = QuantumFourierTransform._qft_append_circuit(
            circuit, circuit.num_qubits - 1
        )

        # Apply the swaps to the circuit.
        circuit = QuantumFourierTransform._qft_append_swaps(circuit)

        return circuit

    @staticmethod
    def qft_circuit(qubit_count: int) -> QuantumCircuit:
        """Create a QFT circuit with given Qubit count.

        Parameters
        ----------
        qubit_count : int
            The number of qubits to use in the circuit.

        Returns
        -------
        QuantumCircuit
            The QFT circuit.
        """
        return QuantumFourierTransform.qft(QuantumCircuit(qubit_count))

    @staticmethod
    def iqft_circuit(qubit_count: int) -> QuantumCircuit:
        """Create a iQFT circuit with given Qubit count.

        Parameters
        ----------
        qubit_count : int
            The number of qubits to use in the circuit.

        Returns
        -------
        QuantumCircuit
            The iQFT circuit.
        """
        return QuantumFourierTransform.iqft(QuantumCircuit(qubit_count))

    @staticmethod
    def iqft(circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply inverse QFT to a circuit.

        Parameters
        ----------
            circuit (QuantumCircuit):  The circuit to apply the IQFT to.

        Returns
        -------
            QuantumCircuit: The circuit with the iQFT applied.
        """
        # Apply the swaps to the circuit.
        circuit = QuantumFourierTransform._qft_append_swaps(circuit, inverse=True)
        circuit.barrier()

        # Apply the QFT to the circuit.
        circuit = QuantumFourierTransform._iqft_append_circuit(circuit, 0)

        return circuit

    @staticmethod
    def _qft_append_circuit(circuit: QuantumCircuit, qubit: int) -> QuantumCircuit:
        """Apply a rotation to a qubit.

        Parameters
        ----------
            circuit (QuantumCircuit): The circuit to apply the rotation to.
            qubit (int): The qubit to apply the rotation to.

        Returns
        -------
            QuantumCircuit: The circuit with the rotation applied.
        """
        # Recursive stop condition.
        if qubit < 0:
            return circuit

        # Construct the minimal QFT circuit.
        circuit.h(qubit)
        for qubit_line in reversed(range(qubit)):
            circuit.cp(pi / 2 ** (qubit - qubit_line), qubit_line, qubit)
        circuit.barrier()

        # Recursively apply the QFT to the next qubit.
        return QuantumFourierTransform._qft_append_circuit(circuit, qubit - 1)

    @staticmethod
    def _iqft_append_circuit(circuit: QuantumCircuit, qubit: int) -> QuantumCircuit:
        """Apply a rotation to a qubit.

        Parameters
        ----------
            circuit (QuantumCircuit): The circuit to apply the rotation to.
            qubit (int): The qubit to apply the rotation to.

        Returns
        -------
            QuantumCircuit: The circuit with the rotation applied.
        """
        # Recursive stop condition.
        if qubit >= circuit.num_qubits:
            return circuit

        # Construct the minimal QFT circuit.
        for qubit_line in range(qubit):
            circuit.cp(-pi / 2 ** (qubit - qubit_line), qubit_line, qubit)
        circuit.h(qubit)
        circuit.barrier()

        # Recursively apply the QFT to the next qubit.
        return QuantumFourierTransform._iqft_append_circuit(circuit, qubit + 1)

    @staticmethod
    def _qft_append_swaps(
        circuit: QuantumCircuit, inverse: bool = False
    ) -> QuantumCircuit:
        """Apply swaps to a circuit.

        Parameters
        ----------
            circuit (QuantumCircuit): The circuit to apply the swaps to.
            qubit (int): The qubit to apply the rotation to.

        Returns
        -------
            QuantumCircuit: The circuit with the swaps applied.
        """
        qubit_count = circuit.num_qubits
        qubits_list = (
            reversed(range(qubit_count // 2))
            if not inverse
            else range(qubit_count // 2)
        )
        for qubit in qubits_list:
            circuit.swap(qubit, qubit_count - qubit - 1)
        return circuit



if __name__ == "__main__":
    print("===================================")
    print("Quantum Fourier Transform Simulator")
    print("===================================")

    # Get the input state as integer decimal.
    state_int = int(input("> Enter the state as decimal integer: "))

    # Run the algorithm.
    result = QuantumFourierTransform.simulate(state_int)
    print(f"iQFT result: {result['result']}")
