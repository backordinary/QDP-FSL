# https://github.com/SchmidtMoritz/IAGQ/blob/476610f4975247e99ecf0ff3ffcbbe6dafcb901c/src/IAGQ/circuit_runner.py
from qiskit import *
from qiskit.utils import QuantumInstance
from qiskit.opflow import (
    CircuitOp,
    CircuitSampler,
    MatrixExpectation,
    CircuitStateFn,
    StateFn,
    MatrixOp,
    PauliExpectation,
)
import numpy as np


class CircuitRunner:
    """encapsulation class of all circuit executions and measurements"""

    def __init__(self, ansatz, observable, backend):
        self.ansatz = ansatz
        self.observable = observable
        self.exact = False  # flag: measurement by sampling or exact expectation value computation via opflow
        self.total_shots = 0  # counter of total amount of shots used

        if backend == "simulator":  # measure using simulated shots
            self.backend = Aer.get_backend("qasm_simulator")
            np.random.seed(40)
        elif backend == "opflow":  # measure computing exact expactaion
            self.exact = True
            self.backend = "opflow"
        else:  # use actual quantum computers as backend
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
            if backend == "lima":
                self.backend = provider.get_backend("ibmq_lima")
            elif backend == "quito":
                self.backend = provider.get_backend("ibmq_quito")
            elif backend == "manila":
                self.backend = provider.get_backend("ibmq_manila")
            elif backend == "belem":
                self.backend = provider.get_backend("ibmq_belem")
            else:
                raise Exception("Unknown Backend!")

    def run(self, parameter_values, shots, comp_variance=False, squared=False):

        """
        :param parameter_values: parameter values
        :param shots: shots per measurement
        :param comp_variance: if measurement via samples, return samplevariance of samples
        :param squared: use squared of Observable as Observable, used for <O>^2 -<O^2> variance computation
        """
        if shots == 0:
            assert self.exact

        self.total_shots += shots

        bound_circuit = self.ansatz.get_bound_circuit(parameter_values)
        circuit_state_fn = CircuitStateFn(bound_circuit)

        if squared:  # construct new squared observable
            observable_state_fn = StateFn(
                MatrixOp(self.observable.get_matrix() @ self.observable.get_matrix()),
                is_measurement=True,
            )
        else:
            observable_state_fn = StateFn(
                MatrixOp(self.observable.get_matrix()), is_measurement=True
            )

        measurement = observable_state_fn @ circuit_state_fn

        if not self.exact:

            # qiskit doesnt use the numpy random seed
            # generate new qiskit random seed with numpy using fixed numpy seed -
            # > reproducibility only requiring fixed numpy random seed

            seed = np.random.randint(-(2**63), 2**63, dtype=np.int64)
            q_instance = QuantumInstance(self.backend, shots=shots, seed_simulator=seed)

            expectation_func = PauliExpectation()

            sampler = CircuitSampler(q_instance, attach_results=True)

            samples = sampler.convert(expectation_func.convert(measurement))

            expectation = samples.eval().real

            if comp_variance:
                variance = expectation_func.compute_variance(samples).real
                return expectation, variance
            else:
                return expectation

        else:  # opflow
            expectation = measurement.eval().real

            if comp_variance:
                variance = 0.0  # no sample variance using opflow
                return expectation, variance
            else:
                return expectation
