# https://github.com/Farquhar13/QNN-Gen/blob/4901f6bc16942732210ec2b5b16433f32f10fab3/qnn_gen/model.py
"""
Will need to import Measurement for default_measurements
"""

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister
import qiskit
import numpy as np
from .utility import Gate

class Model(ABC):
    """
    Class to define a QML model, whether it be a particular ansatz or a perceptron model.

    Abstract class. Derived classes overwrite circuit() and may overwrite default_measuremnt().
    Derived classes should ensure that the self.n_qubits and self.parameters attributes are updated.
    """

    def __init__(self):
        """
        Attributes:
            - n_qubits (int): Number of encoded qubits in the encoding circuit to be joined with
            the model circuit. In other words, number of qubits not counting ancillary qubits.
            - params (np.ndarray): The model's learnable parameters
        """
        self.n_qubits = None
        self.parameters = None

    @abstractmethod
    def circuit():
        """
        Abstract method. Overwrite to return the circuit.

        Returns:
            - (qiskit.QuantumCircuit): The circuit defined by the model
        """
        pass


    def default_measurement():
        """
        Often, the model implies what measurements are sensible.

        Returns:
            - (Measurement): The default measurement for the model
        """
        raise NotImplementedError

    @staticmethod
    def print_models():
        """
        Prints the available derived classes.
        """
        print("TensorTreeNetwork" + "\n",
              "BinaryPerceptron")


class TreeTensorNetwork(Model):
    """
    Class to implement Tensor Tree Networks.

    Assumptions:
        - Number of features is a power of two.
    """

    def __init__(self, n_qubits=None, angles=None, rotation_gate=Gate.RY, entangling_gate=Gate.CX):
        """
        Inputs:
            - n_qubits=None (int): Number of encoded qubits
            - anlges=None (list): Parameters for rotation_gate. Sets self.parameters
            - rotation_gate=Gate.RY (qiskit.circuit.library.standard_gates): 1-qubit rotation gate
            - entangling_gate=Gate.CX (qiskit.circuit.library.standard_gates): 2-qubit entangling gate
        """
        self.n_layers = None
        self.measurement_qubit = None
        self.parameters = angles
        if isinstance(self.parameters, (list, np.ndarray)):
            self.n_parameters = len(self.parameters)
        else:
            self.n_parameters = None
        self.n_qubits = n_qubits # may update attributes above
        self.rotation_gate = rotation_gate
        self.entangling_gate = entangling_gate

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        """
        Sets self.n_qubits.

        Assumes n_qubits is a power of two.

        Also may update related attributes:
            - self.measurement_qubit, self.n_parameters, self.parameters
        """
        # If None is passed to to the constructor
        if n_qubits == None:
            self._n_qubits = None
            return

        # Check input
        if np.log2(n_qubits) % 1 != 0:
            import warnings
            warnings.warn("TTN assumes n_qubits is a power of two.")

        # Set related attributes
        if self.measurement_qubit == None:
            self.measurement_qubit = n_qubits // 2

        # Check if parameters have been initialized already
        if not isinstance(self.parameters, (list, np.ndarray)):
            n_layers = int(np.log2(n_qubits))
            self.n_parameters = 2**(n_layers + 1) - 1 # Geometric series, a=1, r=2
            self.parameters = self.get_rand_angles(n_qubits)

        # Set self.n_qubits
        self._n_qubits = n_qubits

    def get_rand_angles(self, n_qubits):
        """
        Generates random numbers from [0, pi)
        """
        rand_angles = np.random.uniform(low=0, high=np.pi, size=self.n_parameters)

        return rand_angles


    def circuit(self):
        """
        Returns:
            - qiskit.QuantumCircuit
        """
        n_qubits = self.n_qubits
        angles = self.parameters

        qc = QuantumCircuit(n_qubits)

        def layer(angle_idx):
            qubit_pairs = [(active_qubits[2*i], active_qubits[2*i+1]) for i in range(int(len(active_qubits)/2))]

            next_active = []

            for q in qubit_pairs:
                qc.append(self.rotation_gate(angles[angle_idx]), [q[0]])
                qc.append(self.rotation_gate(angles[angle_idx + 1]), [q[1]])

                if q[0] < self.measurement_qubit:
                    qc.append(self.entangling_gate, [q[0], q[1]])
                    next_active.append(q[1])
                else:
                    qc.append(self.entangling_gate, [q[1], q[0]])
                    next_active.append(q[0])

                angle_idx += 2

            return next_active, angle_idx

        active_qubits = np.arange(n_qubits) # qubit indices
        n_active = len(active_qubits)

        angle_idx = 0
        while n_active >= 2:
            active_qubits, angle_idx = layer(angle_idx)
            n_active = len(active_qubits)

        qc.append(self.rotation_gate(angles[-1]), [self.measurement_qubit])

        return qc


    def default_measurement(self, observable=None, measurement_qubit=None):
        """
        Default measurement is the Z expectation of the measurement_qubit

        Inputs:
            - observable=None (Observable): If None, defaults to Observable.Z()
            - measurement_qubit (int): If None, defaults to self.measurement_quibt = n_qubits//2

        Returns:
            - Expectation() measurement object
        """
        from .observable import Observable
        from .measurement import Expectation

        if observable is None:
            observable = Observable.Z()

        if measurement_qubit is None:
            measurement_qubit = self.measurement_qubit

        expectation = Expectation(measurement_qubit, observable)

        return expectation


class BinaryPerceptron(Model):
    """
    Class to implement Binary Perceptron models.

    Assumptions:
        - Number of features is a power of two.
        - Features are binary elements in {-1, 1}
    """

    def __init__(self, n_qubits=None, weights=None):
        """
        Attribuites:
            - n_qubits (int)
            - weights (np.ndarray): A binary vector with elements in {-1, 1}
        """
        # Ensure weights is a np array if passed
        if weights is not None:
            self.parameters = np.array(weights)
        else:
            self.parameters = weights
        self.measurement_qubit = None
        self.n_qubits = n_qubits

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        """
        Input:
            n_qubits (int): number of encoded qubits

        Sets self.n_qubits. Assumes n_qubits is a power of two.
        Also may update self.n_layers, self.measurement_qubit, and self.angles
        """
        if n_qubits == None:
            self._n_qubits = None
            return

        #n_qubits -= 1 # Not counting ancilla
        assert np.log2(n_qubits) % 1 == 0, "Assumes n_qubits is a power of two."
        self._n_qubits = n_qubits

        if self.measurement_qubit is None:
            if n_qubits == 1:
                self.measurement_qubit = 0
            else:
                self.measurement_qubit = n_qubits

        if self.parameters is None:
            self.parameters = self.init_random_weights(self.n_qubits)

    def init_random_weights(self, n_qubits):
        return np.random.choice([-1, 1], 2**n_qubits)

    def circuit(self):
        """
        Returns the circuit for the model.
        """
        w = self.parameters

        # Not including ancilla
        N = self.n_qubits
        qubit_idx_list = list(range(N))

        # Generate computational basis vectors in Dirac notation
        basis_labels = [("{:0%db}"%N).format(k) for k in range(len(w))]

        # No need for ancilla if N = 1
        if N == 1:
            Sx = QuantumCircuit(N)
            qubit_ancilla_idx_list = qubit_idx_list

        else:
            qc = QuantumCircuit(N + 1) # + 1 for ancilla

            qubit_ancilla_idx_list = qubit_idx_list + [N] # [N] is the index of the ancilla

        # Create multi-controlled Z gate, or single Z gate if N = 1qubit. Likewise for X
        Z = qiskit.circuit.library.standard_gates.ZGate()
        X = qiskit.circuit.library.standard_gates.XGate()
        if N == 1:
            z_op = Z
            x_op = X

        else:
            z_op = Z.control(N-1)
            x_op = X.control(N)

        # Find all components with a -1 factor in i (and thus our target state vector)
        indices = np.where(w == -1)[0]
        if indices.size > 0:
            for idx in indices:
                # Need to switch qubits in the 0 state so CZ will take effect
                for i, b in enumerate(basis_labels[idx]):
                    if b == '0':
                        qc.x((N-1)-i) # (N-1)-i is to match the qubit ordering Qiskit uses (reversed)

                qc.append(z_op, qubit_idx_list)

                # And switch the flipped qubits back
                for i, b in enumerate(basis_labels[idx]):
                    if b == '0':
                        qc.x((N-1)-i)

        qc.h(qubit_idx_list)
        qc.x(qubit_idx_list)
        qc.append(x_op, qubit_ancilla_idx_list)

        return qc


    def default_measurement(self, observable=None, measurement_qubit=None):
        """
        A measurement on the ancilla qubit. If the probability of the ancilla being in the |1> is
        greater the than the threshold, output gives label 1. Otherwise, label 0.

        Default threshold is 0.5.

        Returns
            - ProbabilityThreshold Measurement object.
        """
        from .measurement import ProbabilityThreshold

        if measurement_qubit == None:
            qubits = self.measurement_qubit

        pt = ProbabilityThreshold(qubits=qubits,
                                 p_zero=False,
                                 threshold=0.5,
                                 labels=[1, 0],
                                 observable=observable)

        return pt


class EntangledQubit(Model):
    """
    Class to an engtangled qubit classifier.
    """

    def __init__(self,
        n_qubits=None,
        angles=None,
        n_layers=2,
        gate_set=[Gate.RXX, Gate.RZX]):
        """
        Inputs:
            - n_qubits=None (int): Number of encoded qubits
            - anlges=None (list): Parameters for rotation gate
            - gate_set=[Gate.RXX, Gate.RZX] (list(qiskit gates)): Parameterized two qubit
            gates to use. self.circuit() loops through the list of gates and repeat until
            n_layers have been constructed.
        """
        self.n_layers = n_layers
        self.measurement_qubit = 0
        self.parameters = angles
        if angles is None:
            self.n_parameters = None
        else:
            self.n_parameters = len(angles)
        self.n_qubits = n_qubits # may update attributes above
        self.gate_list = gate_set

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        """
        Sets self.n_qubits. Assumes n_qubits is a power of two.

        Also may update self.n_layers, self.measurement_qubit, and self.angles
        """
        if n_qubits == None:
            self._n_qubits = None
            return

        self._n_qubits = n_qubits

        if self.parameters is None:
            self.n_parameters = (n_qubits - 1) * self.n_layers
            self.parameters = self.get_rand_angles(self.n_parameters)

    def get_rand_angles(self, n_parameters):
        """
        Generates random numbers from [0, 2*pi)
        """
        rand_angles = np.random.uniform(0, 2*np.pi, n_parameters)

        return rand_angles

    def circuit(self):
        """
        Returns:
            - qiskit.QuantumCircuit
        """
        qc = QuantumCircuit(self.n_qubits)

        for layer_idx in range(self.n_layers):
            # Loop through gate list, return to beginning after end of the list
            gate = self.gate_list[layer_idx % len(self.gate_list)]

            for gate_idx in range(1, self.n_qubits):
                gate_parameter = self.parameters[layer_idx*(self.n_qubits-1) + gate_idx - 1]
                qc.append(gate(gate_parameter), [self.measurement_qubit, gate_idx])

        return qc

    def default_measurement(self, observable=None, measurement_qubit=None):
        """
        Default measurement is the Z expectation of the measurement_qubit

        Inputs:
            - observable=None (qnn_gen.Observable): If None, defaults to Observable.Y()
            - measurement_qubit (int): If None, defaults to self.measurement_qubit,
            which itself defaults to qubit 0.

        Returns:
            - Expectation() measurement object
        """
        from .observable import Observable
        from .measurement import Expectation

        if observable is None:
            observable = Observable.Y()

        if measurement_qubit is None:
            measurement_qubit = self.measurement_qubit

        expectation = Expectation(measurement_qubit, observable)

        return expectation
