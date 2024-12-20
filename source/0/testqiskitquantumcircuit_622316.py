# https://github.com/rubenandrebarreiro/semi-quantum-conference-key-agreement-prototype/blob/fd24b8df32f22fcf6f0b0caf6422d2a2ea46e792/test/ibm_qiskit/circuit/TestQiskitQuantumCircuit.py
"""
Semi-Quantum Conference Key Agreement (SQCKA)

Author:
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

Supervisors:
- Andre Nuno Souto (ansouto@fc.ul.pt)
- Antonio Maria Ravara (aravara@fct.unl.pt)

Acknowledgments:
- Paulo Alexandre Mateus (pmat@math.ist.utl.pt)
"""

# Import Packages and Libraries

# Import Unittest for Python's Unitary Tests
import unittest

# Import N-Dimensional Arrays and Squared Roots from NumPy
from numpy import array, sqrt

# Import Assert_All_Close from NumPy.Testing
from numpy.testing import assert_allclose

# Import Aer and execute from Qiskit
from qiskit import Aer, execute

# Import QiskitQuantumCircuit from IBM_Qiskit.Circuit
from src.ibm_qiskit.circuit import QiskitQuantumCircuit

# Import QiskitQuantumRegister from IBM_Qiskit.Circuit.Quantum
from src.ibm_qiskit.circuit.registers.quantum import QiskitQuantumRegister

# Import QiskitClassicalRegister from IBM_Qiskit.Circuit.Classical
from src.ibm_qiskit.circuit.registers.classical import QiskitClassicalRegister


# Test Cases for the Prepare/Measure in the X-Basis (Diagonal Basis)
class PrepareMeasureXBasisTests(unittest.TestCase):

    # Test #1 for the Prepare/Measure in the X-Basis (Diagonal Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) The Qubit is prepared/measured in the X-Basis (Diagonal Basis);
    def test_prepare_measure_x_basis_1(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_x_basis_1 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeasxbasis1", num_qubits)
        qiskit_classical_register_prepare_measure_x_basis_1 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeasxbasis1", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_x_basis_1 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeasxbasis1",
                                                      qiskit_quantum_register_prepare_measure_x_basis_1,
                                                      qiskit_classical_register_prepare_measure_x_basis_1,
                                                      global_phase=0)

        # Prepare/Measure the Qubit in the X-Basis (Diagonal Basis)
        qiskit_quantum_circuit_prepare_measure_x_basis_1 \
            .prepare_measure_single_qubit_in_x_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_x_basis_1.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the X-Basis (Diagonal Basis) be performed
        assert_allclose(final_state_vector, array([((1. / sqrt(2.)) + 0.j), ((1. / sqrt(2.)) + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #2 for the Prepare/Measure in the X-Basis (Diagonal Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    # 3) The Qubit is prepared/measured in the X-Basis (Diagonal Basis);
    def test_prepare_measure_x_basis_2(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_x_basis_2 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeasxbasis2", num_qubits)
        qiskit_classical_register_prepare_measure_x_basis_2 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeasxbasis2", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_x_basis_2 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeasxbasis2",
                                                      qiskit_quantum_register_prepare_measure_x_basis_2,
                                                      qiskit_classical_register_prepare_measure_x_basis_2,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_prepare_measure_x_basis_2 \
            .apply_pauli_x(qiskit_quantum_register_prepare_measure_x_basis_2.quantum_register[0])

        # Prepare/Measure the Qubit in the X-Basis (Diagonal Basis)
        qiskit_quantum_circuit_prepare_measure_x_basis_2 \
            .prepare_measure_single_qubit_in_x_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_x_basis_2.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the X-Basis (Diagonal Basis) be performed
        assert_allclose(final_state_vector, array([((1. / sqrt(2.)) + 0.j), (-(1. / sqrt(2.)) + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #3 for the Prepare/Measure in the X-Basis (Diagonal Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Hadamard Gate to the 1st Qubit, then, |0⟩ ↦ |+⟩;
    # 3) The Qubit is prepared/measured in the X-Basis (Diagonal Basis);
    def test_prepare_measure_x_basis_3(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_x_basis_3 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeasxbasis3", num_qubits)
        qiskit_classical_register_prepare_measure_x_basis_3 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeasxbasis3", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_x_basis_3 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeasxbasis3",
                                                      qiskit_quantum_register_prepare_measure_x_basis_3,
                                                      qiskit_classical_register_prepare_measure_x_basis_3,
                                                      global_phase=0)

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+⟩)
        qiskit_quantum_circuit_prepare_measure_x_basis_3 \
            .apply_hadamard(qiskit_quantum_register_prepare_measure_x_basis_3.quantum_register[0])

        # Prepare/Measure the Qubit in the X-Basis (Diagonal Basis)
        qiskit_quantum_circuit_prepare_measure_x_basis_3 \
            .prepare_measure_single_qubit_in_x_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_x_basis_3.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the X-Basis (Diagonal Basis) be performed
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #4 for the Prepare/Measure in the X-Basis
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    # 3) It is applied the Hadamard Gate to the 1st Qubit, then, |1⟩ ↦ |-⟩;
    # 4) The Qubit is prepared/measured in the X-Basis;
    def test_prepare_measure_x_basis_4(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_x_basis_4 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeasxbasis4", num_qubits)
        qiskit_classical_register_prepare_measure_x_basis_4 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeasxbasis4", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_x_basis_4 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeasxbasis4",
                                                      qiskit_quantum_register_prepare_measure_x_basis_4,
                                                      qiskit_classical_register_prepare_measure_x_basis_4,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_prepare_measure_x_basis_4 \
            .apply_pauli_x(qiskit_quantum_register_prepare_measure_x_basis_4.quantum_register[0])

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|1⟩ ↦ |-⟩)
        qiskit_quantum_circuit_prepare_measure_x_basis_4 \
            .apply_hadamard(qiskit_quantum_register_prepare_measure_x_basis_4.quantum_register[0])

        # Prepare/Measure the Qubit in the X-Basis (Diagonal Basis)
        qiskit_quantum_circuit_prepare_measure_x_basis_4 \
            .prepare_measure_single_qubit_in_x_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_x_basis_4.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the X-Basis be performed
        assert_allclose(final_state_vector, array([(0. + 0.j), (1. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)


# Test Cases for the Prepare/Measure in the Y-Basis (Diagonal Basis)
class PrepareMeasureYBasisTests(unittest.TestCase):

    # Test #1 for the Prepare/Measure in the Y-Basis (Diagonal Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) The Qubit is prepared/measured in the Y-Basis (Diagonal Basis);
    def test_prepare_measure_y_basis_1(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_y_basis_1 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeasybasis1", num_qubits)
        qiskit_classical_register_prepare_measure_y_basis_1 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeasybasis1", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_y_basis_1 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeasybasis1",
                                                      qiskit_quantum_register_prepare_measure_y_basis_1,
                                                      qiskit_classical_register_prepare_measure_y_basis_1,
                                                      global_phase=0)

        # Prepare/Measure the Qubit in the Y-Basis (Computational Basis)
        qiskit_quantum_circuit_prepare_measure_y_basis_1 \
            .prepare_measure_single_qubit_in_y_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_y_basis_1.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the Y-Basis (Diagonal Basis) be performed
        assert_allclose(final_state_vector, array([((1. / sqrt(2.)) + 0.j), (1. / sqrt(2.)) * (0. + 1.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #2 for the Prepare/Measure in the Y-Basis (Diagonal Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    # 3) The Qubit is prepared/measured in the Y-Basis (Diagonal Basis);
    def test_prepare_measure_y_basis_2(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_y_basis_2 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeasybasis2", num_qubits)
        qiskit_classical_register_prepare_measure_y_basis_2 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeasybasis2", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_y_basis_2 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeasybasis2",
                                                      qiskit_quantum_register_prepare_measure_y_basis_2,
                                                      qiskit_classical_register_prepare_measure_y_basis_2,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_prepare_measure_y_basis_2 \
            .apply_pauli_x(qiskit_quantum_register_prepare_measure_y_basis_2.quantum_register[0])

        # Prepare/Measure the Qubit in the Y-Basis (Diagonal Basis)
        qiskit_quantum_circuit_prepare_measure_y_basis_2 \
            .prepare_measure_single_qubit_in_y_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_y_basis_2.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the Y-Basis (Diagonal Basis) be performed
        assert_allclose(final_state_vector, array([((1. / sqrt(2.)) + 0.j), -(1. / sqrt(2.)) * (0. + 1.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #3 for the Prepare/Measure in the Y-Basis (Diagonal Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Hadamard Gate to the 1st Qubit, then, |0⟩ ↦ |+⟩;
    # 3) The Qubit is prepared/measured in the Y-Basis (Diagonal Basis);
    def test_prepare_measure_y_basis_3(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_y_basis_3 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeasybasis3", num_qubits)
        qiskit_classical_register_prepare_measure_y_basis_3 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeasybasis3", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_y_basis_3 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeasybasis3",
                                                      qiskit_quantum_register_prepare_measure_y_basis_3,
                                                      qiskit_classical_register_prepare_measure_y_basis_3,
                                                      global_phase=0)

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+⟩)
        qiskit_quantum_circuit_prepare_measure_y_basis_3 \
            .apply_hadamard(qiskit_quantum_register_prepare_measure_y_basis_3.quantum_register[0])

        # Prepare/Measure the Qubit in the Y-Basis (Diagonal Basis)
        qiskit_quantum_circuit_prepare_measure_y_basis_3 \
            .prepare_measure_single_qubit_in_y_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_y_basis_3.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the Y-Basis (Diagonal Basis) be performed
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #4 for the Prepare/Measure in the Y-Basis (Diagonal Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    # 3) It is applied the Hadamard Gate to the 1st Qubit, then, |1⟩ ↦ |-⟩;
    # 4) The Qubit is prepared/measured in the Y-Basis (Diagonal Basis);
    def test_prepare_measure_y_basis_4(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_y_basis_4 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeasybasis4", num_qubits)
        qiskit_classical_register_prepare_measure_y_basis_4 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeasybasis4", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_y_basis_4 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeasybasis4",
                                                      qiskit_quantum_register_prepare_measure_y_basis_4,
                                                      qiskit_classical_register_prepare_measure_y_basis_4,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_prepare_measure_y_basis_4 \
            .apply_pauli_x(qiskit_quantum_register_prepare_measure_y_basis_4.quantum_register[0])

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|1⟩ ↦ |-⟩)
        qiskit_quantum_circuit_prepare_measure_y_basis_4 \
            .apply_hadamard(qiskit_quantum_register_prepare_measure_y_basis_4.quantum_register[0])

        # Prepare/Measure the Qubit in the Y-Basis (Diagonal Basis)
        qiskit_quantum_circuit_prepare_measure_y_basis_4 \
            .prepare_measure_single_qubit_in_y_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_y_basis_4.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the Y-Basis (Diagonal Basis) be performed
        assert_allclose(final_state_vector, array([(0. + 0.j), (0. + 1.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)


# Test Cases for the Prepare/Measure in the Z-Basis (Computational Basis)
class PrepareMeasureZBasisTests(unittest.TestCase):

    # Test #1 for the Prepare/Measure in the Z-Basis (Computational Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) The Qubit is prepared/measured in the Z-Basis (Computational Basis);
    def test_prepare_measure_z_basis_1(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_z_basis_1 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeaszbasis1", num_qubits)
        qiskit_classical_register_prepare_measure_z_basis_1 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeaszbasis1", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_z_basis_1 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeaszbasis1",
                                                      qiskit_quantum_register_prepare_measure_z_basis_1,
                                                      qiskit_classical_register_prepare_measure_z_basis_1,
                                                      global_phase=0)

        # Prepare/Measure the Qubit in the Z-Basis (Computational Basis)
        qiskit_quantum_circuit_prepare_measure_z_basis_1 \
            .prepare_measure_single_qubit_in_z_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_z_basis_1.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the Z-Basis (Computational Basis) be performed
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #2 for the Prepare/Measure in the Z-Basis (Computational Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    # 3) The Qubit is prepared/measured in the Z-Basis (Computational Basis);
    def test_prepare_measure_z_basis_2(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_z_basis_2 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeaszbasis2", num_qubits)
        qiskit_classical_register_prepare_measure_z_basis_2 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeaszbasis2", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_z_basis_2 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeaszbasis2",
                                                      qiskit_quantum_register_prepare_measure_z_basis_2,
                                                      qiskit_classical_register_prepare_measure_z_basis_2,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_prepare_measure_z_basis_2 \
            .apply_pauli_x(qiskit_quantum_register_prepare_measure_z_basis_2.quantum_register[0])

        # Prepare/Measure the Qubit in the Z-Basis (Computational Basis)
        qiskit_quantum_circuit_prepare_measure_z_basis_2 \
            .prepare_measure_single_qubit_in_z_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_z_basis_2.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the Z-Basis (Computational Basis) be performed
        assert_allclose(final_state_vector, array([(0. + 0.j), (1. + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #3 for the Prepare/Measure in the Z-Basis (Computational Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Hadamard Gate to the 1st Qubit, then, |0⟩ ↦ |+⟩;
    # 3) The Qubit is prepared/measured in the Z-Basis (Computational Basis);
    def test_prepare_measure_z_basis_3(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_z_basis_3 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeaszbasis3", num_qubits)
        qiskit_classical_register_prepare_measure_z_basis_3 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeaszbasis3", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_z_basis_3 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeaszbasis3",
                                                      qiskit_quantum_register_prepare_measure_z_basis_3,
                                                      qiskit_classical_register_prepare_measure_z_basis_3,
                                                      global_phase=0)

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+⟩)
        qiskit_quantum_circuit_prepare_measure_z_basis_3 \
            .apply_hadamard(qiskit_quantum_register_prepare_measure_z_basis_3.quantum_register[0])

        # Prepare/Measure the Qubit in the Z-Basis (Computational Basis)
        qiskit_quantum_circuit_prepare_measure_z_basis_3 \
            .prepare_measure_single_qubit_in_z_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_z_basis_3.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the Z-Basis (Computational Basis) be performed
        assert_allclose(final_state_vector, array([((1. / sqrt(2.)) + 0.j), ((1. / sqrt(2.)) + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #4 for the Prepare/Measure in the Z-Basis (Computational Basis)
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    # 3) It is applied the Hadamard Gate to the 1st Qubit, then, |1⟩ ↦ |-⟩;
    # 4) The Qubit is prepared/measured in the Z-Basis (Computational Basis);
    def test_prepare_measure_z_basis_4(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_prepare_measure_z_basis_4 = \
            QiskitQuantumRegister.QiskitQuantumRegister("qrmeaszbasis4", num_qubits)
        qiskit_classical_register_prepare_measure_z_basis_4 = \
            QiskitClassicalRegister.QiskitClassicalRegister("crmeaszbasis4", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_prepare_measure_z_basis_4 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcmeaszbasis4",
                                                      qiskit_quantum_register_prepare_measure_z_basis_4,
                                                      qiskit_classical_register_prepare_measure_z_basis_4,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_prepare_measure_z_basis_4 \
            .apply_pauli_x(qiskit_quantum_register_prepare_measure_z_basis_4.quantum_register[0])

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|1⟩ ↦ |-⟩)
        qiskit_quantum_circuit_prepare_measure_z_basis_4 \
            .apply_hadamard(qiskit_quantum_register_prepare_measure_z_basis_4.quantum_register[0])

        # Prepare/Measure the Qubit in the Z-Basis (Computational Basis)
        qiskit_quantum_circuit_prepare_measure_z_basis_4 \
            .prepare_measure_single_qubit_in_z_basis(0, 0, 0, 0, is_final_measurement=False)

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = execute(qiskit_quantum_circuit_prepare_measure_z_basis_4.quantum_circuit,
                                     state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Prepare/Measure in the Z-Basis (Computational Basis) be performed
        assert_allclose(final_state_vector, array([((1. / sqrt(2.)) + 0.j), -((1. / sqrt(2.)) + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)


# Test Cases for the Pauli-I Gate
class PauliIGateTests(unittest.TestCase):

    # Test #1 for the Pauli-I Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-I Gate to the 1st Qubit, then, |0⟩ ↦ |0⟩;
    def test_apply_pauli_i_1(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_i_1 = QiskitQuantumRegister.QiskitQuantumRegister("qrpaulii1", num_qubits)
        qiskit_classical_register_pauli_i_1 = QiskitClassicalRegister.QiskitClassicalRegister("crpaulii1", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_i_1 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpaulii1",
                                                      qiskit_quantum_register_pauli_i_1,
                                                      qiskit_classical_register_pauli_i_1,
                                                      global_phase=0)

        # Apply the Pauli-I Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_i_1.apply_pauli_i(qiskit_quantum_register_pauli_i_1.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_i_1.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Pauli-I Gate be applied
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #2 for the Pauli-I Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-I Gate to the 1st Qubit, then, |0⟩ ↦ |0⟩;
    # 3) It is applied, again, the Pauli-I Gate to the 1st Qubit, then, |0⟩ ↦ |0⟩;
    def test_apply_pauli_i_2(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_i_2 = QiskitQuantumRegister.QiskitQuantumRegister("qrpaulii2", num_qubits)
        qiskit_classical_register_pauli_i_2 = QiskitClassicalRegister.QiskitClassicalRegister("crpaulii2", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_i_2 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpaulii2",
                                                      qiskit_quantum_register_pauli_i_2,
                                                      qiskit_classical_register_pauli_i_2,
                                                      global_phase=0)

        # Apply the Pauli-I Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_i_2.apply_pauli_i(qiskit_quantum_register_pauli_i_2.quantum_register[0])

        # Apply the Pauli-I Gate to the 1st Qubit of the Quantum Circuit, again (|0⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_i_2.apply_pauli_i(qiskit_quantum_register_pauli_i_2.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_i_2.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the two Pauli-I Gates be applied
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)


# Test Cases for the Pauli-X Gate
class PauliXGateTests(unittest.TestCase):

    # Test #1 for the Pauli-X Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    def test_apply_pauli_x_1(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_x_1 = QiskitQuantumRegister.QiskitQuantumRegister("qrpaulix1", num_qubits)
        qiskit_classical_register_pauli_x_1 = QiskitClassicalRegister.QiskitClassicalRegister("crpaulix1", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_x_1 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpaulix1",
                                                      qiskit_quantum_register_pauli_x_1,
                                                      qiskit_classical_register_pauli_x_1,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_pauli_x_1.apply_pauli_x(qiskit_quantum_register_pauli_x_1.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_x_1.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Pauli-X Gate be applied
        assert_allclose(final_state_vector, array([(0. + 0.j), (1. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #2 for the Pauli-X Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    # 3) It is applied, again, the Pauli-X Gate to the 1st Qubit, then, |1⟩ ↦ |0⟩;
    def test_apply_pauli_x_2(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_x_2 = QiskitQuantumRegister.QiskitQuantumRegister("qrpaulix2", num_qubits)
        qiskit_classical_register_pauli_x_2 = QiskitClassicalRegister.QiskitClassicalRegister("crpaulix2", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_x_2 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpaulix2",
                                                      qiskit_quantum_register_pauli_x_2,
                                                      qiskit_classical_register_pauli_x_2,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_pauli_x_2.apply_pauli_x(qiskit_quantum_register_pauli_x_2.quantum_register[0])

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit, again (|1⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_x_2.apply_pauli_x(qiskit_quantum_register_pauli_x_2.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_x_2.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the two Pauli-X Gates be applied
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #3 for the Pauli-X Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Hadamard Gate to the 1st Qubit, then, |0⟩ ↦ |+⟩;
    # 3) It is applied the Pauli-X Gate to the 1st Qubit, then, |+⟩ ↦ |+⟩;
    def test_apply_pauli_x_3(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_x_3 = QiskitQuantumRegister.QiskitQuantumRegister("qrpaulix3", num_qubits)
        qiskit_classical_register_pauli_x_3 = QiskitClassicalRegister.QiskitClassicalRegister("crpaulix3", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_x_3 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpaulix3",
                                                      qiskit_quantum_register_pauli_x_3,
                                                      qiskit_classical_register_pauli_x_3,
                                                      global_phase=0)

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+⟩)
        qiskit_quantum_circuit_pauli_x_3.apply_hadamard(qiskit_quantum_register_pauli_x_3.quantum_register[0])

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit, again (|+⟩ ↦ |+⟩)
        qiskit_quantum_circuit_pauli_x_3.apply_pauli_x(qiskit_quantum_register_pauli_x_3.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_x_3.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Single Qubits (Hadamard and Pauli-X) Gates be applied
        assert_allclose(final_state_vector, array([((1. / sqrt(2.)) + 0.j), ((1. / sqrt(2.)) + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #4 for the Pauli-X Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Hadamard Gate to the 1st Qubit, then, |0⟩ ↦ |+⟩;
    # 3) It is applied the Pauli-X Gate to the 1st Qubit, then, |+⟩ ↦ |+⟩;
    # 4) It is applied, again, the Hadamard Gate to the 1st Qubit, then, |+⟩ ↦ |0⟩;
    def test_apply_pauli_x_4(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_x_4 = QiskitQuantumRegister.QiskitQuantumRegister("qrpaulix4", num_qubits)
        qiskit_classical_register_pauli_x_4 = QiskitClassicalRegister.QiskitClassicalRegister("crpaulix4", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_x_4 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpaulix4",
                                                      qiskit_quantum_register_pauli_x_4,
                                                      qiskit_classical_register_pauli_x_4,
                                                      global_phase=0)

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+⟩)
        qiskit_quantum_circuit_pauli_x_4.apply_hadamard(qiskit_quantum_register_pauli_x_4.quantum_register[0])

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit, again (|+⟩ ↦ |+⟩)
        qiskit_quantum_circuit_pauli_x_4.apply_pauli_x(qiskit_quantum_register_pauli_x_4.quantum_register[0])

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|+⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_x_4.apply_hadamard(qiskit_quantum_register_pauli_x_4.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_x_4.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Single Qubits (Hadamard and Pauli-X) Gates be applied
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)


# Test Cases for the Pauli-Y Gate
class PauliYGateTests(unittest.TestCase):

    # Test #1 for the Pauli-Y Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-Y Gate to the 1st Qubit, then, |0⟩ ↦ |+i⟩;
    def test_apply_pauli_y_1(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_y_1 = QiskitQuantumRegister.QiskitQuantumRegister("qrpauliy1", num_qubits)
        qiskit_classical_register_pauli_y_1 = QiskitClassicalRegister.QiskitClassicalRegister("crpauliy1", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_y_1 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpauliy1",
                                                      qiskit_quantum_register_pauli_y_1,
                                                      qiskit_classical_register_pauli_y_1,
                                                      global_phase=0)

        # Apply the Pauli-Y Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+i⟩)
        qiskit_quantum_circuit_pauli_y_1.apply_pauli_y(qiskit_quantum_register_pauli_y_1.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_y_1.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Pauli-Y Gate be applied
        assert_allclose(final_state_vector, array([(0. + 0.j), (0. + 1.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #2 for the Pauli-Y Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-Y Gate to the 1st Qubit, then, |0⟩ ↦ |+i⟩;
    # 3) It is applied, again, the Pauli-Y Gate to the 1st Qubit, then, |+i⟩ ↦ |0⟩;
    def test_apply_pauli_y_2(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_y_2 = QiskitQuantumRegister.QiskitQuantumRegister("qrpauliy2", num_qubits)
        qiskit_classical_register_pauli_y_2 = QiskitClassicalRegister.QiskitClassicalRegister("crpauliy2", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_y_2 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpauliy2",
                                                      qiskit_quantum_register_pauli_y_2,
                                                      qiskit_classical_register_pauli_y_2,
                                                      global_phase=0)

        # Apply the Pauli-Y Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+i⟩)
        qiskit_quantum_circuit_pauli_y_2.apply_pauli_y(qiskit_quantum_register_pauli_y_2.quantum_register[0])

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit, again (|+i⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_y_2.apply_pauli_y(qiskit_quantum_register_pauli_y_2.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_y_2.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the two Pauli-Y Gates be applied
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #3 for the Pauli-Y Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-X Gate to the 1st Qubit, then, |0⟩ ↦ |1⟩;
    # 3) It is applied the Pauli-Y Gate to the 1st Qubit, then, |1⟩ ↦ -i|0⟩;
    def test_apply_pauli_y_3(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_y_3 = QiskitQuantumRegister.QiskitQuantumRegister("qrpauliy3", num_qubits)
        qiskit_classical_register_pauli_y_3 = QiskitClassicalRegister.QiskitClassicalRegister("crpauliy3", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_y_3 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpauliy3",
                                                      qiskit_quantum_register_pauli_y_3,
                                                      qiskit_classical_register_pauli_y_3,
                                                      global_phase=0)

        # Apply the Pauli-X Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |1⟩)
        qiskit_quantum_circuit_pauli_y_3.apply_pauli_x(qiskit_quantum_register_pauli_y_3.quantum_register[0])

        # Apply the Pauli-Y Gate to the 1st Qubit of the Quantum Circuit, again (|1⟩ ↦ -i|0⟩)
        qiskit_quantum_circuit_pauli_y_3.apply_pauli_y(qiskit_quantum_register_pauli_y_3.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_y_3.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Single Qubits (Pauli-X and Pauli-Y) Gates be applied
        assert_allclose(final_state_vector, array([(0. - 1.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #4 for the Pauli-Y Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Hadamard Gate to the 1st Qubit, then, |0⟩ ↦ |+⟩;
    # 3) It is applied the Pauli-Y Gate to the 1st Qubit, then, |+⟩ ↦ 1/sqrt(2)i x (-|0⟩ + |1⟩);
    def test_apply_pauli_y_4(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_y_4 = QiskitQuantumRegister.QiskitQuantumRegister("qrpauliy4", num_qubits)
        qiskit_classical_register_pauli_y_4 = QiskitClassicalRegister.QiskitClassicalRegister("crpauliy4", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_y_4 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpauliy4",
                                                      qiskit_quantum_register_pauli_y_4,
                                                      qiskit_classical_register_pauli_y_4,
                                                      global_phase=0)

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+⟩)
        qiskit_quantum_circuit_pauli_y_4.apply_hadamard(qiskit_quantum_register_pauli_y_4.quantum_register[0])

        # Apply the Pauli-Y Gate to the 1st Qubit of the Quantum Circuit, again (|+⟩ ↦ 1/sqrt(2)i x (-|0⟩ + |1⟩))
        qiskit_quantum_circuit_pauli_y_4.apply_pauli_y(qiskit_quantum_register_pauli_y_4.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_y_4.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Single Qubits (Hadamard and Pauli-Y) Gates be applied
        assert_allclose(final_state_vector, array([(0. - ((1. / sqrt(2.)) * 1.j)), (0. + ((1. / sqrt(2.)) * 1.j))]),
                        rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)


# Test Cases for the Pauli-Z Gate
class PauliZGateTests(unittest.TestCase):

    # Test #1 for the Pauli-Z Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-Z Gate to the 1st Qubit, then, |0⟩ ↦ |0⟩;
    def test_apply_pauli_z_1(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_z_1 = QiskitQuantumRegister.QiskitQuantumRegister("qrpauliz1", num_qubits)
        qiskit_classical_register_pauli_z_1 = QiskitClassicalRegister.QiskitClassicalRegister("crpauliz1", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_z_1 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpauliz1",
                                                      qiskit_quantum_register_pauli_z_1,
                                                      qiskit_classical_register_pauli_z_1,
                                                      global_phase=0)

        # Apply the Pauli-Z Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_z_1.apply_pauli_z(qiskit_quantum_register_pauli_z_1.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_z_1.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Pauli-Z Gate be applied
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #2 for the Pauli-Z Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Pauli-Z Gate to the 1st Qubit, then, |0⟩ ↦ |0⟩;
    # 3) It is applied, again, the Pauli-Z Gate to the 1st Qubit, then, |0⟩ ↦ |0⟩;
    def test_apply_pauli_z_2(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_pauli_z_2 = QiskitQuantumRegister.QiskitQuantumRegister("qrpauliz2", num_qubits)
        qiskit_classical_register_pauli_z_2 = QiskitClassicalRegister.QiskitClassicalRegister("crpauliz2", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_pauli_z_2 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qcpauliz2",
                                                      qiskit_quantum_register_pauli_z_2,
                                                      qiskit_classical_register_pauli_z_2,
                                                      global_phase=0)

        # Apply the Pauli-Z Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_z_2.apply_pauli_z(qiskit_quantum_register_pauli_z_2.quantum_register[0])

        # Apply the Pauli-Z Gate to the 1st Qubit of the Quantum Circuit, again (|0⟩ ↦ |0⟩)
        qiskit_quantum_circuit_pauli_z_2.apply_pauli_z(qiskit_quantum_register_pauli_z_2.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_pauli_z_2.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the two Pauli-Z Gates be applied
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)


# Test Cases for the Hadamard Gate
class HadamardGateTests(unittest.TestCase):

    # Test #1 for the Hadamard Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Hadamard Gate to the 1st Qubit, then, |0⟩ ↦ |+⟩;
    def test_apply_hadamard_1(self):

        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_hadamard_1 = QiskitQuantumRegister.QiskitQuantumRegister("qrhadamard1", num_qubits)
        qiskit_classical_register_hadamard_1 = QiskitClassicalRegister.QiskitClassicalRegister("crhadamard1", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_hadamard_1 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qchadamard1",
                                                      qiskit_quantum_register_hadamard_1,
                                                      qiskit_classical_register_hadamard_1,
                                                      global_phase=0)

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+⟩)
        qiskit_quantum_circuit_hadamard_1.apply_hadamard(qiskit_quantum_register_hadamard_1.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_hadamard_1.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the Hadamard Gate be applied
        assert_allclose(final_state_vector, array([((1. / sqrt(2.)) + 0.j), ((1. / sqrt(2.)) + 0.j)]), rtol=1e-7,
                        atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)

    # Test #2 for the Hadamard Gate
    # Description of the Test Case:
    # 1) The Quantum Circuit is created with a Quantum Register,
    #    with 1 Qubit initialized in the state |0⟩;
    # 2) It is applied the Hadamard Gate to the 1st Qubit, then, |0⟩ ↦ |+⟩;
    # 3) It is applied, again, the Hadamard Gate to the 1st Qubit, then, |+⟩ ↦ |1⟩;
    def test_apply_hadamard_2(self):
        # The number of Qubits and Bits, for Quantum and Classical Registers, respectively
        num_qubits = num_bits = 1

        # Creation of the IBM Qiskit's Quantum and Classical Registers
        qiskit_quantum_register_hadamard_2 = QiskitQuantumRegister.QiskitQuantumRegister("qrhadamard2", num_qubits)
        qiskit_classical_register_hadamard_2 = QiskitClassicalRegister.QiskitClassicalRegister("crhadamard2", num_bits)

        # Creation of the IBM Qiskit's Quantum Circuit with one Quantum and Classical Registers
        qiskit_quantum_circuit_hadamard_2 = \
            QiskitQuantumCircuit.QiskitQuantumCircuit("qchadamard2",
                                                      qiskit_quantum_register_hadamard_2,
                                                      qiskit_classical_register_hadamard_2,
                                                      global_phase=0)

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit (|0⟩ ↦ |+⟩)
        qiskit_quantum_circuit_hadamard_2.apply_hadamard(qiskit_quantum_register_hadamard_2.quantum_register[0])

        # Apply the Hadamard Gate to the 1st Qubit of the Quantum Circuit, again (|+⟩ ↦ |0⟩)
        qiskit_quantum_circuit_hadamard_2.apply_hadamard(qiskit_quantum_register_hadamard_2.quantum_register[0])

        # Getting the Backend for the State Vector Representation
        # (i.e., the Quantum State represented as State Vector)
        state_vector_backend = Aer.get_backend('statevector_simulator')

        # Execute the Quantum Circuit and store the Quantum State in a final state vector
        final_state_vector = \
            execute(qiskit_quantum_circuit_hadamard_2.quantum_circuit, state_vector_backend).result().get_statevector()

        # Assert All Close, from NumPy's Testing, for the State Vector of the Qubit,
        # after the two Hadamard Gates be applied
        assert_allclose(final_state_vector, array([(1. + 0.j), (0. + 0.j)]), rtol=1e-7, atol=1e-7)

        # Dummy Assert Equal for Unittest
        self.assertEqual(True, True)


# Configuration of the Test Suites
if __name__ == '__main__':

    # Test Cases for the Measurements in the X-, Y- and Z-Basis
    prepare_measure_x_basis_tests_suite = unittest.TestLoader().loadTestsFromTestCase(PrepareMeasureXBasisTests)
    prepare_measure_y_basis_tests_suite = unittest.TestLoader().loadTestsFromTestCase(PrepareMeasureYBasisTests)
    prepare_measure_z_basis_tests_suite = unittest.TestLoader().loadTestsFromTestCase(PrepareMeasureZBasisTests)

    # Test Cases for the Pauli Gates
    pauli_i_gate_tests_suite = unittest.TestLoader().loadTestsFromTestCase(PauliIGateTests)
    pauli_x_gate_tests_suite = unittest.TestLoader().loadTestsFromTestCase(PauliXGateTests)
    pauli_y_gate_tests_suite = unittest.TestLoader().loadTestsFromTestCase(PauliYGateTests)
    pauli_z_gate_tests_suite = unittest.TestLoader().loadTestsFromTestCase(PauliZGateTests)

    # Test Cases for the Hadamard Gates
    hadamard_gate_tests_suite = unittest.TestLoader().loadTestsFromTestCase(HadamardGateTests)

    # Create a Global for all the Test Cases established
    all_test_cases = unittest.TestSuite([prepare_measure_x_basis_tests_suite,
                                         prepare_measure_y_basis_tests_suite,
                                         prepare_measure_z_basis_tests_suite,
                                         pauli_i_gate_tests_suite, pauli_x_gate_tests_suite,
                                         pauli_y_gate_tests_suite, pauli_z_gate_tests_suite,
                                         hadamard_gate_tests_suite])
