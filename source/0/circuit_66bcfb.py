# https://github.com/Bright-Shard/QCrypt/blob/27cc468b6420e62ff82e933eef4832d1d872cb54/qcrypt/circuit.py
import qiskit
from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister


class QCryptCircuit:
    # CONSTRUCTOR
    def __init__(self) -> None:
        # Configuration
        # Bit storage: The 0th bit is for the XORed bit, the 1st bit is for the bit to entangle
        self.bits: ClassicalRegister = ClassicalRegister(2, "bits")
        self.qubits: QuantumRegister = QuantumRegister(2, "qubits")
        self.runner = Aer.get_backend("qasm_simulator")
        self.__circuit: QuantumCircuit = QuantumCircuit(self.qubits, self.bits)
        # Ensure bits are only modified after the circuit has been reset // all qubits are at 0
        self._reset = True

    def run(self, control_bits: bin, target_bits: bin) -> bin:
        if len(control_bits) != len(target_bits):
            print("ERROR: CONTROL BITS AND TARGET BITS AREN'T THE SAME LENGTH!")
            exit(1)

        result: bin = ""

        for i in range(0, len(control_bits), 1):
            self.__set_control_bit(int(control_bits[i], 2))  # The value of the bit that was XORed
            self.__set_target_bit(int(target_bits[i], 2))  # The value of the bit to be changed by entanglement
            self.__entangle()  # Entangle the qubits
            result += self.__execute()  # Run the circuit, then add the result to the ciphertext
            self.__reset()  # Reset the circuit for the next iteration

        return result

    def __reset(self) -> None:
        self.__circuit.reset(self.qubits)
        self._reset = True

    def __set_control_bit(self, bit: int) -> None:
        if bit == 1 and self._reset:
            self.__circuit.x(self.qubits[0])
        elif not self._reset:
            print("Circuit is not reset!")

    def __set_target_bit(self, bit: int) -> None:
        if bit == 1 and self._reset:
            self.__circuit.x(self.qubits[1])
        elif not self._reset:
            print("Circuit is not reset!")

    def __entangle(self) -> None:
        if self._reset:
            self.__circuit.cx(self.qubits[0], self.qubits[1])
            self.__circuit.measure(self.qubits, self.bits)
        elif not self._reset:
            print("Circuit is not reset!")

    def __execute(self) -> str:
        self._reset = False
        executor = qiskit.execute(self.__circuit, self.runner, shots=1200)
        results: dict = executor.result().get_counts()
        final_result: dict = {"state": "0", "chance": "0"}

        for qubit_state in results.keys():
            chance = results[qubit_state]
            if chance > int(final_result["chance"]):
                final_result["chance"] = chance
                final_result["state"] = qubit_state

        return final_result["state"][0]
