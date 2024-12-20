# https://github.com/dylanljones/qclib/blob/93d048b5e7cc1eec3e5ecc30d998e05e8baeeef2/qclib/libary/interferometer.py
# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import qiskit
import numpy as np
from abc import ABC, abstractmethod
from ..circuit import measure, run


class AbstractInterferometer(ABC):

    def __init__(self, num_qubits, num_steps=0, shots=1028, backend=None):
        self.areg = qiskit.AncillaRegister(1)
        self.creg = qiskit.ClassicalRegister(1)
        self.qreg = qiskit.QuantumRegister(num_qubits)
        self.shots = shots
        self.backend = backend
        self.num_steps = num_steps

    def set_stepnum(self, step):
        self.num_steps = step

    @abstractmethod
    def prepare(self, qc, qreg) -> None:
        """Adds gates to initialize the state to evolve."""
        pass

    @abstractmethod
    def evolve(self, qc, qreg) -> None:
        """Adds gates to evolve the initial state in time."""
        pass

    def build_circuit(self, alpha="x", beta="x", gamma="z", index=0):
        qc = qiskit.QuantumCircuit(self.qreg, self.areg, self.creg)
        anc = qc.ancillas[0]
        qreg = self.qreg
        # Initialize ground state
        self.prepare(qc, qreg)
        qc.barrier()

        # Entangle ancilla qubit with work qubits
        qc.h(anc)
        getattr(qc, "c" + alpha)(anc, qreg[index], ctrl_state=0)

        # Evolve state
        self.evolve(qc, qc.qregs[0])

        # Entangle ancilla qubit with work qubits
        getattr(qc, "c" + beta)(anc, qreg[index], ctrl_state=1)
        qc.h(anc)

        # measure ancilla qubit
        measure(qc, anc, qc.cregs[0], basis=gamma)

        return qc

    def measure(self, alpha="x", beta="x", basis="z", index=0, shots=0):
        shots = shots or self.shots
        qc = self.build_circuit(alpha, beta, basis, index)
        return run(qc, shots=shots, gpu=False)

    def measure_xx(self, basis="z", index=0):
        return self.measure("x", "x", basis, index).expectation(0)

    def measure_yy(self, basis="z", index=0):
        return self.measure("y", "y", basis, index).expectation(0)

    def measure_xy(self, basis="z", index=0):
        return self.measure("x", "y", basis, index).expectation(0)

    def measure_yx(self, basis="z", index=0):
        return self.measure("y", "x", basis, index).expectation(0)


class TrotterInterferometer(AbstractInterferometer):

    def __init__(self, init_circ, trotter_step_circ, num_steps=0, shots=1028, backend=None):
        num_qubits = init_circ.num_qubits
        assert num_qubits == trotter_step_circ.num_qubits
        super().__init__(num_qubits, num_steps, shots, backend)

        self._init = init_circ.to_instruction()
        self._step = trotter_step_circ.to_instruction()

    def prepare(self, qc, qreg) -> None:
        qc.append(self._init, qreg[:])

    def evolve(self, qc, qreg) -> None:
        for _ in range(self.num_steps):
            qc.append(self._step, qreg[:])

    def measure_imag(self, num_step=0, index=0):
        if num_step:
            self.set_stepnum(num_step)
        return self.measure_xx(basis="z", index=index)

    def measure_gf_imag(self, num_steps, index=0):
        gf = np.zeros(num_steps, dtype=np.float64)
        for step in range(num_steps):
            print(f"\rMeasuring: {step}/{num_steps - 1}", end="", flush=True)
            gf[step] = self.measure_imag(step, index)
        print()
        return gf
