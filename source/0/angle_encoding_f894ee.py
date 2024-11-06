# https://github.com/mbeisel/VQA-workflow/blob/6c7ec57dd4826f2130b6d72735c118ca35f04ffb/services/circuit-generation-service/api/services/encodings/angle_encoding.py
from qiskit import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from builtins import Exception


class AngleEncoding:
    @classmethod
    def angle_encode_vector(cls, vector, rotation="y"):
        """
        :param vector: input vector containing floats to encode
        :rotation string: x, y, or z to indicate which rotation shall be applied
        :return: OpenQASM Circuit
        """

        q = QuantumRegister(len(vector))
        encoding_subcircuit = QuantumCircuit(q)

        for i, value in enumerate(vector):
            if rotation.lower() == "x":
                encoding_subcircuit.rx(2 * value, i)
            elif rotation.lower() == "y":
                encoding_subcircuit.ry(2 * value, i)
            elif rotation.lower() == "z":
                encoding_subcircuit.rz(2 * value, i)
            else:
                raise Exception("Invalid rotation.")

        return encoding_subcircuit
