# https://github.com/VoicuTomut/QMatches/blob/6d25ad2be8437362c8915b4ef56ee643f82a8266/qmatches/gates/switch.py
import numpy as np
from qiskit import QuantumCircuit


class GZB:
    def __init__(self, theta):
        self.theta = theta
        self.mat = np.array(
            [
                [1, 0, 0, 0],
                [0, -np.sin(self.theta), np.cos(self.theta), 0],
                [0, np.cos(self.theta), np.sin(self.theta), 0],
                [0, 0, 0, -1],
            ]
        )
        self.name = "GZB( " + str(theta) + ")"
        self.gate = self.get_gate()

    def get_gate(self):

        qc = QuantumCircuit(2, name=self.name)
        qc.cx(0, 1)
        qc.z(0)
        qc.cry((np.pi / 2 - self.theta) * 2, 1, 0)
        qc.cx(0, 1)

        return qc.to_gate()

    def ad_to(self, qc, q0, q1):
        qc.cx(q0, q1)
        qc.z(q0)
        qc.cry((np.pi / 2 - self.theta) * 2, q1, q0)
        qc.cx(q0, q1)
