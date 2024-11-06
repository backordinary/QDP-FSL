# https://github.com/ogorodnikov/m1/blob/caf4108efd76b580f4d8e60a66033b85cb06948c/app/core-service/core/algorithms/amplitude_estimation.py
"""The Quantum Phase Estimation-based Amplitude Estimation algorithm."""

from typing import Optional, Union, List, Tuple, Dict
from collections import OrderedDict
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance


class AmplitudeEstimation():

    def __init__(
        self,
        num_eval_qubits: int,
        phase_estimation_circuit: Optional[QuantumCircuit] = None,
        iqft: Optional[QuantumCircuit] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ):
        
        if num_eval_qubits < 1:
            raise ValueError("The number of evaluation qubits must at least be 1.")

        super().__init__()

        # set quantum instance
        self.quantum_instance = quantum_instance

        # get parameters
        self._m = num_eval_qubits  # pylint: disable=invalid-name
        self._M = 2 ** num_eval_qubits  # pylint: disable=invalid-name

        self._iqft = iqft
        self._pec = phase_estimation_circuit

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
    ) -> None:
        """Set quantum instance.

        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    def construct_circuit(
        self, estimation_problem, measurement: bool = False
    ):
        
        # use custom Phase Estimation circuit if provided
        if self._pec is not None:
            pec = self._pec

        # otherwise use the circuit library -- note that this does not include the A operator
        else:
            from qiskit.circuit.library import PhaseEstimation

            pec = PhaseEstimation(self._m, estimation_problem.grover_operator, iqft=self._iqft)

        # combine the Phase Estimation circuit with the A operator
        circuit = QuantumCircuit(*pec.qregs)
        circuit.compose(
            estimation_problem.state_preparation,
            list(range(self._m, circuit.num_qubits)),
            inplace=True,
        )
        circuit.compose(pec, inplace=True)

        # add measurements if necessary
        if measurement:
            cr = ClassicalRegister(self._m)
            circuit.add_register(cr)
            circuit.measure(list(range(self._m)), list(range(self._m)))

        return circuit


    def evaluate_measurements(
        self,
        circuit_results: Union[Dict[str, int], np.ndarray],
        threshold: float = 1e-6,
    ) -> Tuple[Dict[int, float], Dict[float, float]]:
        """Evaluate the results from the circuit simulation.

        Given the probabilities from statevector simulation of the QAE circuit, compute the
        probabilities that the measurements y/gridpoints a are the best estimate.

        Args:
            circuit_results: The circuit result from the QAE circuit. Can be either a counts dict
                or a statevector.
            threshold: Measurements with probabilities below the threshold are discarded.

        Returns:
            Dictionaries containing the a gridpoints with respective probabilities and
                y measurements with respective probabilities, in this order.
        """
        # compute grid sample and measurement dicts
        if isinstance(circuit_results, dict):
            samples, measurements = self._evaluate_count_results(circuit_results)
        else:
            samples, measurements = self._evaluate_statevector_results(circuit_results)

        # cutoff probabilities below the threshold
        samples = {a: p for a, p in samples.items() if p > threshold}
        measurements = {y: p for y, p in measurements.items() if p > threshold}

        return samples, measurements


    def _evaluate_statevector_results(self, statevector):
        # map measured results to estimates
        measurements = OrderedDict()  # type: OrderedDict
        num_qubits = int(np.log2(len(statevector)))
        for i, amplitude in enumerate(statevector):
            b = bin(i)[2:].zfill(num_qubits)[::-1]
            y = int(b[: self._m], 2)  # chop off all except the evaluation qubits
            measurements[y] = measurements.get(y, 0) + np.abs(amplitude) ** 2

        samples = OrderedDict()  # type: OrderedDict
        for y, probability in measurements.items():
            if y >= int(self._M / 2):
                y = self._M - y
            # due to the finite accuracy of the sine, we round the result to 7 decimals
            a = np.round(np.power(np.sin(y * np.pi / 2 ** self._m), 2), decimals=7)
            samples[a] = samples.get(a, 0) + probability

        return samples, measurements

    def _evaluate_count_results(self, counts):
        # construct probabilities
        measurements = OrderedDict()
        samples = OrderedDict()
        shots = self._quantum_instance._run_config.shots
        
        print(f'QAE REF shots: {shots}')
        
        for state, count in counts.items():
            y = int(state.replace(" ", "")[: self._m][::-1], 2)
            probability = count / shots
            measurements[y] = probability
            a = np.round(np.power(np.sin(y * np.pi / 2 ** self._m), 2), decimals=7)
            samples[a] = samples.get(a, 0.0) + probability

        print(f'QAE REF samples: {samples}')
        print(f'QAE REF measurements: {measurements}')
        
        return samples, measurements


    def estimate(self, estimation_problem):
        
        result = AmplitudeEstimationResult()
        result.num_evaluation_qubits = self._m
        result.post_processing = estimation_problem.post_processing

        if self._quantum_instance.is_statevector:
            circuit = self.construct_circuit(estimation_problem, measurement=False)
            # run circuit on statevector simulator
            statevector = self._quantum_instance.execute(circuit).get_statevector()
            result.circuit_results = statevector
            
            # store number of shots: convention is 1 shot for statevector,
            # needed so that MLE works!
            result.shots = 1
        else:
            # run circuit on QASM simulator
            circuit = self.construct_circuit(estimation_problem, measurement=True)
            counts = self._quantum_instance.execute(circuit).get_counts()
            result.circuit_results = counts
            
            print(f'QAE REF counts: {counts}')
            
            # store shots
            result.shots = sum(counts.values())

        samples, measurements = self.evaluate_measurements(result.circuit_results)

        result.samples = samples
        
        result.samples_processed = {
            estimation_problem.post_processing(a): p for a, p in samples.items()
        }
        result.measurements = measurements

        # determine the most likely estimate
        result.max_probability = 0
        for amplitude, (mapped, prob) in zip(samples.keys(), result.samples_processed.items()):
            if prob > result.max_probability:
                result.max_probability = prob
                result.estimation = amplitude
                result.estimation_processed = mapped

        # store the number of oracle queries
        result.num_oracle_queries = result.shots * (self._M - 1)

        return result





class AmplitudeEstimationResult():
    """The ``AmplitudeEstimation`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._num_evaluation_qubits = None
        self._mle = None
        self._mle_processed = None
        self._samples = None
        self._samples_processed = None
        self._y_measurements = None
        self._max_probability = None

    @property
    def num_evaluation_qubits(self) -> int:
        """Returns the number of evaluation qubits."""
        return self._num_evaluation_qubits

    @num_evaluation_qubits.setter
    def num_evaluation_qubits(self, num_evaluation_qubits: int) -> None:
        """Set the number of evaluation qubits."""
        self._num_evaluation_qubits = num_evaluation_qubits

    @property
    def mle_processed(self) -> float:
        """Return the post-processed MLE for the amplitude."""
        return self._mle_processed

    @mle_processed.setter
    def mle_processed(self, value: float) -> None:
        """Set the post-processed MLE for the amplitude."""
        self._mle_processed = value

    @property
    def samples_processed(self) -> Dict[float, float]:
        """Return the post-processed measurement samples with their measurement probability."""
        return self._samples_processed

    @samples_processed.setter
    def samples_processed(self, value: Dict[float, float]) -> None:
        """Set the post-processed measurement samples."""
        self._samples_processed = value

    @property
    def mle(self) -> float:
        r"""Return the MLE for the amplitude, in $[0, 1]$."""
        return self._mle

    @mle.setter
    def mle(self, value: float) -> None:
        r"""Set the MLE for the amplitude, in $[0, 1]$."""
        self._mle = value

    @property
    def samples(self) -> Dict[float, float]:
        """Return the measurement samples with their measurement probability."""
        return self._samples

    @samples.setter
    def samples(self, value: Dict[float, float]) -> None:
        """Set the measurement samples with their measurement probability."""
        self._samples = value

    @property
    def measurements(self) -> Dict[int, float]:
        """Return the measurements as integers with their measurement probability."""
        return self._y_measurements

    @measurements.setter
    def measurements(self, value: Dict[int, float]) -> None:
        """Set the measurements as integers with their measurement probability."""
        self._y_measurements = value

    @property
    def max_probability(self) -> float:
        """Return the maximum sampling probability."""
        return self._max_probability

    @max_probability.setter
    def max_probability(self, value: float) -> None:
        """Set the maximum sampling probability."""
        self._max_probability = value

