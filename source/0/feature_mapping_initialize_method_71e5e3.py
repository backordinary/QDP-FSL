# https://github.com/cemeterysummoning/fruse/blob/40b420231b230a9e36a9be7ac2a2181038666eac/feature_mapping_initialize_method.py
# for importing functions
from sklearn import preprocessing
import numpy as np

from qiskit import QuantumCircuit

from atexit import register
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import Aer

import math
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.reset import Reset

# Initializer code

_EPS = 1e-3  # global variable used to chop very small numbers to zero

class Initialize(Instruction):
    def __init__(self, params):
        """Create new initialize composite.

        params (list): vector of complex amplitudes to initialize to
        """
        #print(params)
        #print(math.log2(len(params)))
        num_qubits = math.log2(len(params))
        #print(num_qubits)

        # Check if param is a power of 2
        if num_qubits == 0 or not num_qubits.is_integer():
            raise QiskitError("Desired statevector length not a positive power of 2.")

        # Check if probabilities (amplitudes squared) sum to 1
        if not math.isclose(sum(np.absolute(params) ** 2), 1.0,
                            abs_tol=_EPS):
            raise QiskitError("Sum of amplitudes-squared does not equal one.")

        num_qubits = int(num_qubits)

        super().__init__("initialize", num_qubits, 0, params)

    def _define(self):
            """Calculate a subcircuit that implements this initialization

            Implements a recursive initialization algorithm, including optimizations,
            from "Synthesis of Quantum Logic Circuits" Shende, Bullock, Markov
            https://arxiv.org/abs/quant-ph/0406176v5

            Additionally implements some extra optimizations: remove zero rotations and
            double cnots.
            """
            # call to generate the circuit that takes the desired vector to zero
            disentangling_circuit = self.gates_to_uncompute()

            # invert the circuit to create the desired vector from zero (assuming
            # the qubits are in the zero state)
            initialize_instr = disentangling_circuit.to_instruction().inverse()

            q = QuantumRegister(self.num_qubits, 'q')
            initialize_circuit = QuantumCircuit(q, name='init_def')
            for qubit in q:
                initialize_circuit.append(Reset(), [qubit])
            initialize_circuit.append(initialize_instr, q[:])

            self.definition = initialize_circuit


    def gates_to_uncompute(self):
        """Call to create a circuit with gates that take the desired vector to zero.

        Returns:
            QuantumCircuit: circuit to take self.params vector to :math:`|{00\\ldots0}\\rangle`
        """
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q, name='disentangler')

        # kick start the peeling loop, and disentangle one-by-one from LSB to MSB
        remaining_param = self.params

        for i in range(self.num_qubits):
            # work out which rotations must be done to disentangle the LSB
            # qubit (we peel away one qubit at a time)
            (remaining_param,
            thetas,
            phis) = Initialize._rotations_to_disentangle(remaining_param)

            # perform the required rotations to decouple the LSB qubit (so that
            # it can be "factored" out, leaving a shorter amplitude vector to peel away)

            add_last_cnot = True
            if np.linalg.norm(phis) != 0 and np.linalg.norm(thetas) != 0:
                add_last_cnot = False

            if np.linalg.norm(phis) != 0:
                rz_mult = self._multiplex(U1Gate, phis, last_cnot=add_last_cnot)
                circuit.append(rz_mult.to_instruction(), q[i:self.num_qubits])

            if np.linalg.norm(thetas) != 0:
                ry_mult = self._multiplex(RYGate, thetas, last_cnot=add_last_cnot)
                circuit.append(ry_mult.to_instruction().reverse_ops(), q[i:self.num_qubits])

        return circuit

    def _rotations_to_disentangle(local_param):
        """
        Static internal method to work out Ry and Rz rotation angles used
        to disentangle the LSB qubit.
        These rotations make up the block diagonal matrix U (i.e. multiplexor)
        that disentangles the LSB.

        [[Ry(theta_1).Rz(phi_1)  0   .   .   0],
        [0         Ry(theta_2).Rz(phi_2) .  0],
                                    .
                                        .
        0         0           Ry(theta_2^n).Rz(phi_2^n)]]
        """
        remaining_vector = []
        thetas = []
        phis = []

        param_len = len(local_param)

        for i in range(param_len // 2):
            # Ry and Rz rotations to move bloch vector from 0 to "imaginary"
            # qubit
            # (imagine a qubit state signified by the amplitudes at index 2*i
            # and 2*(i+1), corresponding to the select qubits of the
            # multiplexor being in state |i>)
            (remains,
            add_theta,
            add_phi) = Initialize._bloch_angles(local_param[2 * i: 2 * (i + 1)])

            remaining_vector.append(remains)

            # rotations for all imaginary qubits of the full vector
            # to move from where it is to zero, hence the negative sign
            thetas.append(-add_theta)
            phis.append(-add_phi)

        return remaining_vector, thetas, phis

    @staticmethod
    def _bloch_angles(pair_of_complex):
        """
        Static internal method to work out rotation to create the passed-in
        qubit from the zero vector.
        """
        [a_complex, b_complex] = pair_of_complex
        # Force a and b to be complex, as otherwise numpy.angle might fail.
        a_complex = complex(a_complex)
        b_complex = complex(b_complex)
        mag_a = np.absolute(a_complex)
        final_r = float(np.sqrt(mag_a ** 2 + np.absolute(b_complex) ** 2))
        if final_r < _EPS:
            theta = 0
            phi = 0
            final_r = 0
            final_t = 0
        else:
            theta = float(2 * np.arccos(mag_a / final_r))
            a_arg = np.angle(a_complex)
            b_arg = np.angle(b_complex)
            final_t = a_arg + b_arg
            phi = b_arg - a_arg

        return final_r * np.exp(1.J * final_t / 2), theta, phi

    def _multiplex(self, target_gate, list_of_angles, last_cnot=True):
        """
        Return a recursive implementation of a multiplexor circuit,
        where each instruction itself has a decomposition based on
        smaller multiplexors.

        The LSB is the multiplexor "data" and the other bits are multiplexor "select".

        Args:
            target_gate (Gate): Ry or Rz gate to apply to target qubit, multiplexed
                over all other "select" qubits
            list_of_angles (list[float]): list of rotation angles to apply Ry and Rz
            last_cnot (bool): add the last cnot if last_cnot = True

        Returns:
            DAGCircuit: the circuit implementing the multiplexor's action
        """
        list_len = len(list_of_angles)
        local_num_qubits = int(math.log2(list_len)) + 1

        q = QuantumRegister(local_num_qubits)
        circuit = QuantumCircuit(q, name="multiplex" + local_num_qubits.__str__())

        lsb = q[0]
        msb = q[local_num_qubits - 1]

        # case of no multiplexing: base case for recursion
        if local_num_qubits == 1:
            circuit.append(target_gate(list_of_angles[0]), [q[0]])
            return circuit

        # calc angle weights, assuming recursion (that is the lower-level
        # requested angles have been correctly implemented by recursion
        angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]],
                                np.identity(2 ** (local_num_qubits - 2)))

        # calc the combo angles
        list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

        # recursive step on half the angles fulfilling the above assumption
        multiplex_1 = self._multiplex(target_gate, list_of_angles[0:(list_len // 2)], False)
        circuit.append(multiplex_1.to_instruction(), q[0:-1])

        # attach CNOT as follows, thereby flipping the LSB qubit
        circuit.append(CXGate(), [msb, lsb])

        # implement extra efficiency from the paper of cancelling adjacent
        # CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
        # second lower-level multiplex)
        multiplex_2 = self._multiplex(target_gate, list_of_angles[(list_len // 2):], False)
        if list_len > 1:
            circuit.append(multiplex_2.to_instruction().reverse_ops(), q[0:-1])
        else:
            circuit.append(multiplex_2.to_instruction(), q[0:-1])

        # attach a final CNOT
        if last_cnot:
            circuit.append(CXGate(), [msb, lsb])

        return circuit


    def broadcast_arguments(self, qargs, cargs):
            flat_qargs = [qarg for sublist in qargs for qarg in sublist]

            if self.num_qubits != len(flat_qargs):
                raise QiskitError("Initialize parameter vector has %d elements, therefore expects %s "
                                "qubits. However, %s were provided." %
                                (2**self.num_qubits, self.num_qubits, len(flat_qargs)))
            yield flat_qargs, []

    def validate_parameter(self, parameter):
            """Initialize instruction parameter can be int, float, and complex."""
            if isinstance(parameter, (int, float, complex)):
                return complex(parameter)
            elif isinstance(parameter, np.number):
                return complex(parameter.item())
            else:
                raise CircuitError("invalid param type {0} for instruction  "
                                "{1}".format(type(parameter), self.name))


    def initialize(self, params, qubits):
        """Apply initialize to circuit."""
        #print(params)
        if not isinstance(qubits, list):
            qubits = [qubits]
        return self.append(Initialize(params), qubits)


    QuantumCircuit.initialize = initialize




# feature mapping method
def state_preperation(circuit, index, array):
    circuit.initialize(array, index)


def normalize_data(array):
    normalized_arr = preprocessing.normalize([array])
    #print(normalized_arr)
    arr = np.reshape(normalized_arr, len(array))
    #print(arr)
    return arr