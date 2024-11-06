# https://github.com/Giermek222/BFQCA-Benchmarking-For-Quantum-Classification-Algorithms-/blob/4b8fa5456b1ee8e6570fba877418335e15752536/BFQCA_Cowskit/custom_algorithms/CUSTOM%20knn.py
import numpy as np

from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.result import Result
from qiskit import QuantumCircuit, BasicAer

from typing import Optional, Union, Dict, List
from sklearn.base import ClassifierMixin
import scipy.stats as stats

from cowskit.encodings import AmplitudeEncoding
from cowskit.utils.utils import _construct_circuits_from_test_data, _kneighbors

import logging
logger = logging.getLogger(__name__)

class QKNN_Base:
    def __init__(self,
                 n_neighbors: int = 4,
                 encoding: AmplitudeEncoding = None,
                 quantum_instance: QuantumInstance = None):
        """
        :param n_neighbors: number of neighbours that the algorithm will compare new data with
        :param encoding: class encoding classical data into quantum vector. Only Amplitude Neocding is currently supported
        :param quantum_instance: Quantum Instance to run quantum gates on
        """
        self.encoding = encoding
        self.quantum_instance = quantum_instance
        self.n_neighbors = n_neighbors
        self.training_parameters = np.asarray([])
        self.training_labels = np.asarray([])

    def fit(self, training_parameters, training_labels):
        """
        :param training_parameters: data that the algorithm will learn on
        :param training_labels: labels that the algorithm will use to categorize training data
        :return: nothing. it just sets data as parameter of class
        """
        self.training_parameters = np.asarray(training_parameters)
        self.training_labels = np.asarray(training_labels)

    def execute(
            self, quantum_circuit: Union[QuantumCircuit, List[QuantumCircuit]]
    ) -> Result:
        """
        :param quantum_circuit: circuit to be executed
        """
        result = self.quantum_instance.execute(quantum_circuit)
        return result

    def _compute_fidelity(self, counts: Dict[str, int]):
        """
        :param counts: dictionary of 0s and 1s
        :return: calculated fidelity
        """
        counts_0 = counts.get('0', 0)
        counts_1 = counts.get('1', 1)
        # squared fidelity
        f_2 = np.abs(counts_0 - counts_1) / self.quantum_instance.run_config.shots
        return np.sqrt(f_2)

    def _get_fidelities(self,
                        results: Result,
                        test_size: int) -> np.ndarray:
        """
        :param results: results from the simulation
        :param test_size: size of test data
        :return: np.ndarray with fidelities of each test data point
        """
        train_size = self.training_parameters.shape[0]
        all_counts = results.get_counts()  # List[Dict(str, int)]

        fidelities = np.empty(
            shape=(test_size, train_size)
        )

        for i, (counts) in enumerate(all_counts):
            fidelity = self._compute_fidelity(counts)
            # the i-th subarray of the ndarray `fidelities` contains
            # the values that we will use for the majority voting to
            # predict the label of the i-th test input data
            fidelities[i // train_size][i % train_size] = fidelity

        return fidelities

class CustomNeighbors(ClassifierMixin, QKNN_Base):

    def __init__(self,
                 n_neighbors: int = 4,
                 encoding_map: Optional[AmplitudeEncoding] = None,
                 quantum_instance: Optional[Union[QuantumInstance, Backend]] = None):

        
        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                            shots=1024,
                            optimization_level=1)
        super().__init__(n_neighbors, encoding_map, quantum_instance)

    def _majority_voting(self,
                         y_train: np.ndarray,
                         fidelities: np.ndarray) -> np.ndarray:

        """
        calculate which class test case should belong into

        :param y_train:
        :param fidelities:
        :return:
        """

        k_nearest = _kneighbors(y_train, fidelities, self.n_neighbors)

        # getting most frequent values in `k_nearest`
        # in a more efficient way than looping and
        # using, for instance, collections.Counter
        n_queries = self.training_parameters.shape[0]
        if n_queries == 1:
            # case of 1D array
            labels, _ = stats.mode(k_nearest)
        else:
            labels, _ = stats.mode(k_nearest, axis=1)

        # eventually flatten the np.ndarray
        # returned by stats.mode
        return labels.real.flatten()

    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        circuits = _construct_circuits_from_test_data(X_test, self.training_parameters, self.encoding)
        results = self.execute(circuits)
        # the execution results are employed to compute
        # fidelities which are used for the majority voting
        fidelities = self._get_fidelities(results, len(X_test))

        return self._majority_voting(self.training_labels, fidelities)
