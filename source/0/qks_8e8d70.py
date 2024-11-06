# https://github.com/diracq/quantum-kitchen-sink/blob/7cce2f6caf887c8cf5daea9526d53c719ca40e36/src/qks.py
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from .config import QKSConfig
from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit import Aer
from typing import Callable, List

simulator = QasmSimulator()


class QuantumKitchenSinks():
    """
    QKS

    param config: QKSConfig
        the quantum computer object

    References:
        Quantum Kitchen Sinks was introduced in:
        C. M. Wilson, J. S. Otterbach, N. Tezak, R. S. Smith,
        G. E. Crooks, and M. P. da Silva 2018. Quantum Kitchen
        Sinks. An algorithm for machine learning on near-term
        quantum computers. <https://arxiv.org/pdf/1806.08321.pdf>
    """

    def __init__(self, config: QKSConfig):
        self.n_episodes = config.n_episodes
        
        self.qubits = config.qubits
        self.scale = config.scale
        self.distribution = config.distribution
        self.n_trials = config.n_trials
        self.num_cols = self.n_episodes * self.qubits
        self.tiling = config.tiling
        

    def qks_preprocess(self, X: Dataset, vectorize_input: Callable, input_dim):
        """Generate set of random parameters for X and apply the QKS transformation

        param X : Dataset sized (n_samples, input_dim) after vectorization Training data
        param vectorize_input : Callable vectorization function from a QKSDataset
        """
        self.input_dim = input_dim
        if not hasattr(self, 'omega'):
            self.r = 1 if input_dim / self.qubits < 1 else int(input_dim / self.qubits)
            self.omega = self._make_omega()
            self.beta = self._make_beta()

        n_samples = len(X)
        random_generated_parameters = self._get_theta(X, vectorize_input)
        return self._transform(random_generated_parameters, n_samples)

    def _transform(self, thetas, n_samples):
        """Apply the QKS transformation to the random parameters"""
        transformations = []
        for theta in tqdm(thetas):
            avg_measurements = self._run_simulator(theta)
            transformations.append(avg_measurements)
        return np.array(transformations).reshape(n_samples, self.num_cols)

    def _make_omega(self):
        """A (q x p) dimensional matrix that is used to encode the input vector into q gate parameters."""

        def _create_selection_matrix():
            """Generates a matrix of 0s and 1s to zero out `r` values per matrix"""
            matrix_size = self.input_dim * self.qubits
            m = np.zeros(matrix_size)
            for i in range(self.r):
                m[i] = 1
            np.random.shuffle(m)
            selection_matrix = m.reshape(self.qubits, self.input_dim)
            return selection_matrix
        
        def _create_selection_matrix_with_tiling():
            """Generates a matrix of 0s and 1s to zero out `r` values per matrix"""
            m = np.zeros((self.qubits, self.input_dim))
            a = int(np.sqrt(self.qubits)) #full image  sidelength (in tiles)
            b = int(np.sqrt(self.input_dim)) #full image  sidelength (in pixels)
            c = int(np.ceil(b/a))    #single tile sidelength (in pixels)
            for entry in range(self.input_dim):
                column = np.floor(entry/b)
                row = entry%b
                tilecol = int(np.floor(column/c))
                tilerow = int(np.floor(row/c))
                m[tilecol*a+tilerow][entry] = 1
            return m

        def _create_selection_matrix_with_bar_tiling():
            """Generates a matrix of 0s and 1s to zero out `r` values per matrix"""
            m = np.zeros((self.qubits, self.input_dim))
            b = int(np.sqrt(self.input_dim)) #full image sidelength (in pixels)
            width = int(np.floor(b/self.qubits)) #bar width (in tiles)
            for entry in range(self.input_dim):
                bar_num = int(np.floor(entry/width))
                m[bar_num][entry] = 1
            return m

        size = (self.n_episodes, self.qubits, self.input_dim)

        if self.distribution == "normal":
            dist = np.random.normal(loc=0, scale=self.scale, size=size)
        else:
            raise AttributeError(
                "QKS currently only implemented for normal distributions. Use distribution = 'normal'.")

        if self.tiling:
            selection_matrix = np.array([_create_selection_matrix_with_tiling() for _ in range(self.n_episodes)])
        # TODO: Add bar tiling option
        else:
            selection_matrix = np.array([_create_selection_matrix() for _ in range(self.n_episodes)])
        omega = dist * selection_matrix  # matrix chooses which values to keep

        return omega

    def _make_beta(self):
        """random q-dimensional bias vector"""
        return np.random.uniform(low=0, high=(2 * np.pi), size=(self.n_episodes, self.qubits))

    def _get_theta(self, X: Dataset, vectorize_input: Callable) -> List[float]:
        """
        A linear transformation to get our set of random parameters to feed into the quantum circuit

        param X: Dataset
        return: List[float]
        """
        thetas = []
        for x, _ in X:
            x = vectorize_input(x)
            for e in range(self.n_episodes):
                theta = self.omega[e].dot(x.T) + self.beta[e]
                thetas.append(theta)
        return thetas

    def _build_and_compile(self, thetas):
        """ 
        Creates the quantum circuit and compiles the program into an executable.
        Mimics the circuits from the appendix of the paper, with the exception of
        the ordering of the circuit.cx gates (before and after compilation they still do not match).
        This is probably best just left up to the compiler.
        For ease of reading the printed operations, the qubits are looped over several times.
        """
        n_qubits = self.qubits
        qubits = np.arange(n_qubits)
        program = QuantumCircuit(n_qubits, n_qubits)

        sq = int(np.sqrt(n_qubits))
        lim = n_qubits - sq - 1

        for m in qubits:
            program.rx(thetas[m], m)

        for m in qubits:
            m_1 = m + 1
            m_sq = m + sq
            skip = (m_1) % sq

            if m_1 < n_qubits:
                if (m_sq >= n_qubits):
                    program.cx(m, m_1)
                else:
                    program.cx(m, m_sq)

            if (m < lim) and (skip != 0): program.cx(m, m_1)

        for m in qubits:
            program.measure(m, m)

        return program

    def _run_simulator(self, theta):
        self.executable = self._build_and_compile(theta)
        job = simulator.run(self.executable, shots=self.n_trials)
        result = job.result()
        counts = result.get_counts(self.executable)
        q_counts = [0 for _ in range(self.qubits)]

        for key in counts:
            for i, char in enumerate(key):
                if char == '1':
                    q_counts[i] += counts[key]
        
        q_counts = [x/self.n_trials for x in q_counts]
            
        return q_counts


