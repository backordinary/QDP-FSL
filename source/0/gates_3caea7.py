# https://github.com/maxaksel/Toffoli-Optimization/blob/1dc92ef3e05bad075d25b1c7c279e484d9bf4132/benchmarking/gates.py
"""
Module containing tools for generating Toffoli gate pulse schedules.
"""

from typing import List
from math import pi
import numpy as np

#Qiskit-related imports
from qiskit import pulse, QuantumCircuit
from qiskit.pulse.schedule import Schedule
from qiskit.providers.ibmq import IBMQBackend

# Superstaq-related imports
import cirq
from cirq_superstaq import Service, AceCRPlusMinus, AceCRMinusPlus
import cirq_superstaq
from cirq.contrib.svg import SVGCircuit


class Gates:
    """
    Class for managing gate-level creation of Toffoli gates for QPT benchmarking
    """

    def __init__(self, backend: IBMQBackend):
        self.backend = backend

    def get_canonical_full_toffoli(self, qubits: List[int]) -> QuantumCircuit:
        qc = QuantumCircuit(self.backend.configuration().n_qubits)

        q0 = qubits[0]
        q1 = qubits[1]
        q2 = qubits[2]

        qc.h(q2)
        qc.cx(q1, q2)
        qc.tdg(q2)
        qc.cx(q0, q2)
        qc.t(q2)
        qc.cx(q1, q2)
        qc.tdg(q2)
        qc.cx(q0, q2)
        qc.t(q1)
        qc.t(q2)
        qc.cx(q0, q1)
        qc.t(q0)
        qc.tdg(q1)
        qc.h(q2)
        qc.cx(q0, q1)

        return qc

    def get_canonical_linear_toffoli(self, qubits: List[int]) -> QuantumCircuit:
        qc = QuantumCircuit(self.backend.configuration().n_qubits)

        q0 = qubits[0]
        q1 = qubits[1]
        q2 = qubits[2]

        qc.h(q2)
        qc.t(q0)
        qc.t(q1)
        qc.t(q2)
        qc.cx(q0, q1)
        qc.cx(q1, q2)
        qc.cx(q0, q1)
        qc.t(q2)
        qc.cx(q1, q2)
        qc.cx(q0, q1)
        qc.tdg(q1)
        qc.tdg(q2)
        qc.cx(q1, q2)
        qc.cx(q0, q1)
        qc.tdg(q2)
        qc.cx(q1, q2)
        qc.h(q2)

        return qc

class PulseGates:
    """
    Class for managing creation of Toffoli gate pulse schedules for QPT benchmarking.
    """

    def __init__(self, backend: IBMQBackend, ss_host: str, ss_api: str):
        """
        Constructor for PulseGates class.

        :param backend: IBMQBackend being targeted
        :param ss_host: Superstaq host server (can be set to empty string)
        :param ss_api: Superstaq client API key
        """
        self.backend = backend

        if ss_host == "":
            self.service = Service(api_key=ss_api)
        else:
            self.service = Service(remote_host=ss_host, api_key=ss_api) # Superstaq API key

    def get_canonical_linear_toffoli(self, qubits: List[int]) -> Schedule:
        """
        Get a pulse schedule for the standard or "canonical" quantum circuit implementation of
        the Toffoli gate.


        :param qubits: the qubits over which the Toffoli gate should act 
        :return: a Qiskit Schedule object
        """
        q0 = qubits[0]
        q1 = qubits[1]
        q2 = qubits[2]

        with pulse.build(self.backend, name='canonical_linear_toffoli') as canonical_toffoli_schedule:
            pulse.u3(np.pi/2, 0, np.pi, q2)  # Hadamard

            # T gates on qubits 0, 1, 2
            pulse.u3(0, 0, np.pi/4, q0)
            pulse.u3(0, 0, np.pi/4, q1)
            pulse.u3(0, 0, np.pi/4, q2)

            pulse.cx(q0, q1)
            pulse.cx(q1, q2)
            pulse.cx(q0, q1)

            pulse.u3(0, 0, np.pi/4, q2)

            pulse.cx(q1, q2)
            pulse.cx(q0, q1)

            pulse.u3(0, 0, -np.pi/4, q1)
            pulse.u3(0, 0, -np.pi/4, q2)

            pulse.cx(q1, q2)
            pulse.cx(q0, q1)

            pulse.u3(0, 0, -np.pi/4, q2)

            pulse.cx(q1, q2)

            pulse.u3(np.pi/2, 0, np.pi, q2)  # Hadamard

        return canonical_toffoli_schedule

    def get_optimized_linear_toffoli(self, qubits: List[int], target: str) -> Schedule:
        """
        Get a pulse schedule of the native gate-level optimized Toffoli gate for linear
        qubit architectures.

        :param qubits: the qubits over which the Toffoli gate should act
        :param target: a string representing the IBM Q target device
        :return: a Qiskit Schedule object
        """
        q0 = cirq.LineQubit(qubits[0])
        q1 = cirq.LineQubit(qubits[1])
        q2 = cirq.LineQubit(qubits[2])

        # q0 = qubits[0]
        # q1 = qubits[1]
        # q2 = qubits[2]
        opt_toffoli = cirq.Circuit(
            # Specify pulse here
            cirq.Moment(cirq.rz(3*np.pi/4)(q0), cirq.rz(np.pi/4)(q1), cirq.rz(-np.pi/2)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("-+", np.pi/2)(q0, q1), cirq.rx(np.pi/2)(q2)),
            cirq.Moment(cirq.rz(-np.pi/2)(q0), cirq.rz(np.pi/2)(q1), cirq.rz(np.pi/4)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("-+", -np.pi/2)(q1, q2)),
            cirq.Moment(cirq_superstaq.AceCR("+-", np.pi/2)(q0, q1)),
            cirq.Moment(cirq.rz(np.pi/2)(q0), cirq.rz(-np.pi/2)(q1), cirq.rz(np.pi/4)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("+-", np.pi/2)(q1, q2)),
            cirq.Moment(cirq_superstaq.AceCR("-+", np.pi/2)(q0, q1)),
            cirq.Moment(cirq.rz(-np.pi/2)(q0), cirq.rz(np.pi/4)(q1), cirq.rz(-np.pi/4)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("-+", np.pi/2)(q1, q2)),
            cirq.Moment(cirq_superstaq.AceCR("+-", np.pi/2)(q0, q1)),
            cirq.Moment(cirq.rz(-np.pi/2)(q1), cirq.rz(-np.pi/4)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("+-")(q1, q2)),
            cirq.Moment(cirq.rz(np.pi/2)(q2)),
            cirq.Moment(cirq.rx(np.pi/2)(q2)),
            cirq.Moment(cirq.rz(np.pi)(q2))
        )

        print(cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cirq.TOFFOLI), cirq.unitary(opt_toffoli),
                                                              atol=1e-8))

        opt_compiler_object = self.service.ibmq_compile(opt_toffoli, target=target)
        opt_schedule = opt_compiler_object.pulse_sequence

        return opt_schedule

    def get_optimized_linear_toffoli_no_sandwich(self, qubits: List[int], target: str) -> Schedule:
        """
        Get a pulse schedule of the native gate-level optimized Toffoli gate for linear
        qubit architectures. Do not use sandwiching method.

        :param qubits: the qubits over which the Toffoli gate should act
        :param target: a string representing the IBM Q target device
        :return: a Qiskit Schedule object
        """
        q0 = cirq.LineQubit(qubits[0])
        q1 = cirq.LineQubit(qubits[1])
        q2 = cirq.LineQubit(qubits[2])

        # q0 = qubits[0]
        # q1 = qubits[1]
        # q2 = qubits[2]
        opt_toffoli = cirq.Circuit(
            # Specify pulse here
            cirq.Moment(cirq.rz(3 * np.pi / 4)(q0), cirq.rz(np.pi / 4)(q1), cirq.rz(-np.pi / 2)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("-+")(q0, q1), cirq.rx(np.pi / 2)(q2)),
            cirq.Moment(cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.rz(-np.pi / 2)(q0), cirq.rz(np.pi / 2)(q1), cirq.rz(np.pi / 4)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("-+")(q1, q2)),
            cirq.Moment(cirq.rx(-np.pi / 2)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("+-")(q0, q1)),
            cirq.Moment(cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.rz(np.pi / 2)(q0), cirq.rz(-np.pi / 2)(q1), cirq.rz(np.pi / 4)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("+-")(q1, q2)),
            cirq.Moment(cirq.rx(np.pi / 2)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("-+")(q0, q1)),
            cirq.Moment(cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.rz(-np.pi / 2)(q0), cirq.rz(np.pi / 4)(q1), cirq.rz(-np.pi / 4)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("-+")(q1, q2)),
            cirq.Moment(cirq.rx(np.pi / 2)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("+-")(q0, q1)),
            cirq.Moment(cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.rz(-np.pi / 2)(q1), cirq.rz(-np.pi / 4)(q2)),
            cirq.Moment(cirq_superstaq.AceCR("+-")(q1, q2)),
            cirq.Moment(cirq.rz(np.pi / 2)(q2)),
            cirq.Moment(cirq.rx(np.pi / 2)(q2)),
            cirq.Moment(cirq.rz(np.pi)(q2))
        )

        print(cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cirq.TOFFOLI), cirq.unitary(opt_toffoli),
                                                              atol=1e-8))

        opt_compiler_object = self.service.ibmq_compile(opt_toffoli, target=target)
        opt_schedule = opt_compiler_object.pulse_sequence

        return opt_schedule
