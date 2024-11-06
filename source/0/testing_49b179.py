# https://github.com/JDfaker/QCsimulator/blob/3b3006cd80a5f36cfac25b366ef5fefb34f1c554/testing.py
'''
This module holds the Utility_Testing, Gate_For_Test, Gate_Testing, Sparse_Testing and Grover_Testing classes, and the Test_Run class to allow pytest to run them. \n
'''
import qiskit
import numpy as np
import quantum_circuit
import sparse_matrix
import scipy.sparse


class Utility_Testing:
    '''
    Defines generic methods used in multiple test classes. \n
    '''

    def __init__(self, seed=None):
        '''
        Constructor for Utility_Testing. Used if a seeded generator is desired. \n
        @param seed: allows the random matrix generator to be seeded. If not specified, the generator is used with a random seed. \n
        '''
        self.seed = np.random.seed(seed)

    def get_random_matrices(self):
        '''
        Creates two random matrices with values between 1 and 50 and random shapes. \n
        The matrices are created so as to be compatible for the purposes of matrix products. \n
        '''
        test_matrix_1 = np.random.randint(2, 50, (np.random.randint(2, 10), np.random.randint(2, 10)))
        test_matrix_2 = np.random.randint(2, 50, np.shape(np.transpose(test_matrix_1)))
        return test_matrix_1, test_matrix_2

    def run_qiskit_circuit(self, circ):
        '''
        Runs the statevector simulator on the provided qiskit circuit, producing output equivalent to that of our simulator. \n
        @param circ: qiskit circuit to run.  \n
        @return outputstate: array of statevectors returned from running the circuit. \n
        '''
        backend = qiskit.Aer.get_backend('statevector_simulator')
        job = qiskit.execute(circ, backend)
        result = job.result()
        outputstate = result.get_statevector(circ, decimals=3)
        return outputstate


class Gate_For_Test:
    '''
    Defines gates to be tested. \n
    '''

    def __init__(self, qiskit_name, our_name, num_qubits):
        '''
        Constructor for gates. \n
        @param qiskit_name: defines the name of the method for the gate in the qiskit library \n
        @param our_name: defines the corresponding method name in our simulator \n
        @param num_qubits - defines how many qubits the gate operates on and therefore how many must be provided to not raise an error \n
        '''
        self.qiskit_name = qiskit_name
        self.our_name = our_name
        self.num_qubits = num_qubits


class Gate_Testing:
    '''
    Tests the given gate with both qiskit and our simulator and compares the results. \n
    '''

    def __init__(self, qubits):
        '''
        Constructor for the gate_testing class. Creates both test circuits and defines a database of gates. \n
        @param qubits: size of test circuits. \n
        '''
        self.num_qubits = qubits
        self.qiskit_circ = qiskit.QuantumCircuit(qubits)
        self.our_circ = quantum_circuit.QuantumCircuit(qubits)

        self.gate_database = [Gate_For_Test("h", "apply_hardmard", 1), Gate_For_Test("x", "apply_pauliX", 1),
                              Gate_For_Test("z", "apply_pauliZ", 1), Gate_For_Test("swap", "apply_swap", 2)]

    def gate_test(self, gate_input, *test_qubits):
        '''
        Applies the specified gate to the specified qubits. \n
        Confirms that the number of qubits provided is valid for both the size of the register and the gate. \n
        Additionally confirms that no duplicate qubits have been provided. \n
        Having done so, executes the gate on both circuits and ensures that the results match to within a given tolerance. \n

        Quick gate name reference: \n
        Hadamard - h \n
        Pauli-X - x \n
        Pauli-Z - z \n
        SWAP - swap \n

        @param gate_input: qiskit name of gate to be tested. \n
        @param test_qubits: qubits for the gate to operate on. In the event of a controlled gate, the first is the control and the second is the target. \n
        '''
        assert np.size(np.unique(np.array(test_qubits))) == np.size(
            np.array(test_qubits)), "Test qubits may not contain any duplicates. {} != {}".format(
            np.unique(np.array(test_qubits)), np.array(test_qubits))
        assert np.max(np.array(
            test_qubits)) < self.num_qubits, "Test qubit index greater than largest register index. {} >= {}".format(
            np.max(np.array(test_qubits)), self.num_qubits)

        test_qubits_ours = np.subtract(np.dot(np.ones_like(test_qubits), self.num_qubits - 1), test_qubits)
        gate_this_test = next((x for x in self.gate_database if x.qiskit_name == gate_input), None)

        assert len(
            test_qubits) <= gate_this_test.num_qubits, "Number of test qubits provided is greater than number given gate operates on. {} > {}".format(
            len(test_qubits), gate_this_test.num_qubits)

        exec("self.qiskit_circ." + str(gate_this_test.qiskit_name) + "(*test_qubits)")
        qiskit_output = Utility_Testing().run_qiskit_circuit(self.qiskit_circ)
        exec("self.our_circ." + str(gate_this_test.our_name) + "(*test_qubits_ours)")
        our_output = np.transpose(sparse_matrix.SparseMatrix.numpy(self.our_circ.state))[0].astype(complex)

        assert (np.abs(np.subtract(qiskit_output,
                                   our_output)) <= 0.00005).all(), "The states after the gate's application do not match. {} != {}".format(
            qiskit_output, our_output)


class Sparse_Testing:
    '''
    Methods to test the functionality of Sparse Matrix methods. \n
    '''

    def __init__(self, test_matrix_1=None, test_matrix_2=None, seed=None):
        '''
        Constructor for sparse matrix testing. Note that if either matrix is None, both will be randomly generated, to ensure
        consistency of dot product methods. \n
        Only test_matrix_1 is used for basic_sparsify_test, sparse_multiply_test, sparse_transpose_test, and get_attribute_test.
        In this case test_matrix_2 should be set to anything other than None if a specific test_matrix_1 is desired. \n

        @param test_matrix_1: First matrix to multiply. Will be generated randomly if not provided. \n
        @param test_matrix_2: Second matrix to multiply. Will be generated randomly if not provided. \n
        '''
        self.seed = seed

        if test_matrix_1 == None or test_matrix_2 == None:
            self.test_matrix_1, self.test_matrix_2 = Utility_Testing(self.seed).get_random_matrices()
        else:
            self.test_matrix_1 = test_matrix_1
            self.test_matrix_2 = test_matrix_2

    def basic_sparsify_test(self):
        '''
        Ensures that the sparsify and numpy methods are consistent for the given matrix by converting to sparse and back again
        and comparing the result with the original matrix passed to the method. \n
        '''
        test_matrix = self.test_matrix_1
        test_matrix = sparse_matrix.SparseMatrix.sparsify(test_matrix)
        test_matrix = sparse_matrix.SparseMatrix.numpy(test_matrix)
        assert np.equal(test_matrix,
                        self.test_matrix_1).all(), "The matrix does not match its original form. {} != {}".format(
            test_matrix, self.test_matrix_1)

    def sparse_dot_test(self):
        '''
        Performs a dot product with both scipy.sparse and our class and confirm that the results are identical. \n
        '''
        our_dot = sparse_matrix.SparseMatrix.numpy(sparse_matrix.SparseMatrix.sparsify(self.test_matrix_1).dot(
            sparse_matrix.SparseMatrix.sparsify(self.test_matrix_2)))
        scipy_dot = scipy.sparse.csc_matrix(self.test_matrix_1).dot(
            scipy.sparse.csc_matrix(self.test_matrix_2)).toarray()
        assert np.equal(our_dot, scipy_dot).all(), "The output matrices do not match. {} != {}".format(our_dot,
                                                                                                       scipy_dot)

    def sparse_tensor_dot_test(self):
        '''
        Performs a tensor dot product with both scipy.sparse and our class and confirm that the results are identical. \n
        '''
        our_tensor_dot = sparse_matrix.SparseMatrix.numpy(
            sparse_matrix.SparseMatrix.sparsify(self.test_matrix_1).tensordot(
                sparse_matrix.SparseMatrix.sparsify(self.test_matrix_2)))
        scipy_tensor_dot = scipy.sparse.kron(scipy.sparse.csc_matrix(self.test_matrix_1),
                                             scipy.sparse.csc_matrix(self.test_matrix_2)).toarray()
        assert np.equal(our_tensor_dot, scipy_tensor_dot).all(), "The output matrices do not match.{} != {}".format(
            our_tensor_dot, scipy_tensor_dot)

    def sparse_multiply_test(self, multiple):
        '''
        Multiplies both the scipy matrix and our simulator's matrix by a scalar and compares the results. \n

        @param multiple: scalar to multiply matrices by. \n
        '''
        our_multiply = sparse_matrix.SparseMatrix.numpy(
            sparse_matrix.SparseMatrix.sparsify(self.test_matrix_1).multiply(multiple))
        scipy_multiply = scipy.sparse.csc_matrix(self.test_matrix_1).multiply(multiple).toarray()
        assert (our_multiply == scipy_multiply).all(), "The output matrices do not match.{} != {}".format(our_multiply,
                                                                                                          scipy_multiply)

    def sparse_minus_test(self):
        '''
        Subtracts both matrices with both our simulator and scipy and compares the results. \n
        '''
        our_subtract = sparse_matrix.SparseMatrix.numpy(
            sparse_matrix.SparseMatrix.sparsify(np.transpose(self.test_matrix_1)).minus(
                sparse_matrix.SparseMatrix.sparsify(self.test_matrix_2)))
        scipy_subtract = (scipy.sparse.csc_matrix(np.transpose(self.test_matrix_1)) - scipy.sparse.csc_matrix(
            self.test_matrix_2)).toarray()
        assert (our_subtract == scipy_subtract).all(), "The output matrices do not match.{} != {}".format(our_subtract,
                                                                                                          scipy_subtract)

    def sparse_transpose_test(self):
        '''
        Transposes the test matrix with both our simulator and scipy and compares the results. \n
        '''
        our_transpose = sparse_matrix.SparseMatrix.numpy(
            sparse_matrix.SparseMatrix.sparsify(self.test_matrix_1).transpose())
        scipy_transpose = (scipy.sparse.csc_matrix(self.test_matrix_1).transpose()).toarray()
        assert (our_transpose == scipy_transpose).all(), "The output matrices do not match.{} != {}".format(
            our_transpose, scipy_transpose)

    def get_attribute_test(self, operation, *args):
        '''
        Performs the 'get' method corresponding to the operation given on the sparse form of test_matrix, passing the given
        arguments, then compares the output to the original dense matrix to ensure correctness. \n

        Operation quick reference: \n
        row - Gets all nonzero entries of stated row as a dictionary with their positions in the row as keys. \n
        col - Gets all nonzero entries of stated column as a dictionary with their positions in the column as keys. \n
        value - Gets value at the given row and column. \n
        nonzero_rows - Gets all rows with nonzero elements. \n
        nonzero_cols - Gets all columns with nonzero elements. \n

        @param operation: operation to perform. \n
        @param args: arguments of the test function. Should be a single int for row and col, a pair of ints (row, col) for value, and
        nothing for nonzero_rows and nonzero_cols. \n
        '''
        output = []
        exec("output.append(sparse_matrix.SparseMatrix.sparsify(self.test_matrix_1).get_" + str(operation) + "(*args))")
        output = output[0]
        if operation == "col":
            out_keys = output.keys()
            for key in out_keys:
                assert output[key] == self.test_matrix_1[int(key)][
                    args[0]], "An incorrect nonzero value has been retrieved by get_col. {} != {}".format(output[key],
                                                                                                          self.test_matrix_1[
                                                                                                              int(key)][
                                                                                                              args[0]])
            assert not np.any(np.array([self.test_matrix_1[x][args[0]] for x in range(len(self.test_matrix_1)) if
                                        x not in out_keys])), "Nonzero values in the column have not been retrived by get_col: {}.".format(
                np.array([self.test_matrix_1[x][args[0]] for x in range(len(self.test_matrix_1)) if x not in out_keys]))
        elif operation == "row":
            out_keys = output.keys()
            for key in out_keys:
                assert output[key] == self.test_matrix_1[args[0]][
                    int(key)], "An incorrect nonzero value has been retrieved by get_row. {} != {}".format(output[key],
                                                                                                           self.test_matrix_1[
                                                                                                               args[0]][
                                                                                                               int(
                                                                                                                   key)])
            assert not np.any(np.array(
                [self.test_matrix_1[args[0]][x] for x in range(len(self.test_matrix_1[args[0]])) if
                 x not in out_keys])), "Nonzero values in the row have not been retrived by get_row: {}.".format(
                np.array([self.test_matrix_1[args[0]][x] for x in range(len(self.test_matrix_1[args[0]])) if
                          x not in out_keys]))
        elif operation == "value":
            assert output == self.test_matrix_1[
                args], "The found value does not match the corresponding value in the dense matrix. {} != {}".format(
                output, self.test_matrix_1[args])
        elif operation == "nonzero_rows":
            assert output == [x for x in range(len(self.test_matrix_1)) if self.test_matrix_1[
                x].any() == True], "Rows with nonzero elements exist that have not been retrived by get_nonzero_rows: {}.".format(
                [x for x in range(len(self.test_matrix_1)) if self.test_matrix_1[x].any() == True and x not in output])
        elif operation == "nonzero_cols":
            assert output == [x for x in range(len(self.test_matrix_1[0])) if np.array(
                [self.test_matrix_1[i][x] for i in range(len(
                    self.test_matrix_1))]).any() == True], "Columns with nonzero elements exist that have not been retrived by get_nonzero_cols: {}.".format(
                [x for x in range(len(self.test_matrix_1[0])) if np.array([self.test_matrix_1[i][x] for i in range(
                    len(self.test_matrix_1))]).any() == True and x not in output])
        else:
            raise ValueError("Invalid operation provided to get_attribute_test ({})".format(operation))


class Grover_Testing:
    '''
    Contains methods to perform Grover's algorithm with both simulators and compare the results.
    '''

    def __init__(self, qubits):
        '''
        Initialises Grover test by creating circuits and then putting them into a state of superposition by applying a hadamard
        gate to each qubit. \n

        @param qubits: number of qubits for test circuits. \n
        '''
        self.qiskit_circ = qiskit.QuantumCircuit(qubits)
        self.our_circ = quantum_circuit.QuantumCircuit(qubits)
        self.num_qubits = qubits

        for i in range(self.num_qubits):
            self.qiskit_circ.h(i)
            self.our_circ.apply_hardmard(i)

    def our_grover_test(self):
        '''
        Performs a Grover test on our simulator, stopping iterations when the target state has been located to a suitable
        precision. \n

        @return out: state at the time the target qubit was located to within the set tolerance. \n
        '''
        while np.transpose(sparse_matrix.SparseMatrix.numpy(self.our_circ.state))[0][self.target] < 0.999:
            self.our_circ.apply_grover_oracle(self.target)
            self.our_circ.apply_amplification()

        out = np.transpose(sparse_matrix.SparseMatrix.numpy(self.our_circ.state))[0]
        return out

    def qiskit_oracle(self):
        '''
        Uses the same method as our simulator to construct an oracle matrix, then converts it to a qiskit gate. \n

        @return oracle_gate: the oracle matrix converted to a qiskit gate. \n
        '''
        I = np.eye(2 ** self.num_qubits)
        oracle = I
        if isinstance(self.target, int):
            oracle[self.target][self.target] = -1
        else:
            for mark in self.target:
                oracle[mark][mark] = -1

        oracle_gate = qiskit.extensions.UnitaryGate(oracle)
        return oracle_gate

    def qiskit_diffuser(self):
        '''
        General Grover diffuser converted from the version given at https://qiskit.org/textbook/ch-algorithms/grover.html. \n
        Takes the number of qubits and constructs and applies a diffuser of that size. \n
        '''
        for qubit in range(self.num_qubits):
            self.qiskit_circ.h(qubit)
        for qubit in range(self.num_qubits):
            self.qiskit_circ.x(qubit)
        self.qiskit_circ.h(self.num_qubits - 1)
        self.qiskit_circ.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)  # multi-controlled-toffoli
        self.qiskit_circ.h(self.num_qubits - 1)
        for qubit in range(self.num_qubits):
            self.qiskit_circ.x(qubit)
        for qubit in range(self.num_qubits):
            self.qiskit_circ.h(qubit)

    def qiskit_grover_test(self):
        '''
        Performs a Grover test on our simulator, stopping iterations when the target state has been located to a suitable
        precision. Qiskit builds circuits and then runs them, rather than running automatically at every step like our simulator,
        meaning that the circuit must be manually run at each stage. \n

        @return out: state after running the algorithm until the target was located to within the tolerance. \n
        '''
        out = Utility_Testing().run_qiskit_circuit(self.qiskit_circ)
        while out[self.target] * np.conj(out[self.target]) < complex(0.999) * np.conj(complex(0.999)):
            self.qiskit_circ.append(self.qiskit_oracle(), range(self.num_qubits))
            self.qiskit_diffuser()
            out = Utility_Testing().run_qiskit_circuit(self.qiskit_circ)
        return out

    def grover_test(self, target):
        '''
        Performs a Grover test with both simulators searching for the target state. Compares the index of the found state, and
        the value of the state itself within a given tolerance (fairly large, as the discrepency between the two simulators
        increases significantly with the number of qubits in the circuit). Because of differences in output format, the squares
        of the results are compared instead of the raw results. \n

        @param target: target state expressed as a decimal int. \n
        '''
        self.target = target
        qiskit_result = self.qiskit_grover_test()
        our_result = self.our_grover_test()
        assert np.where(qiskit_result ** 2 >= complex(0.999) ** 2) == np.where(
            our_result >= 0.999), "The simulators did not find the same state. {} != {}".format(
            np.where(qiskit_result ** 2 >= complex(0.999) ** 2), np.where(our_result >= 0.999))
        assert np.abs(np.real(np.amax(qiskit_result * np.conj(qiskit_result))) - np.amax(
            our_result) ** 2) <= 0.005, "The converted values of the found states do not match to within +/- 0.005. {} != {}".format(
            np.real(np.amax(qiskit_result * np.conj(qiskit_result))), np.amax(our_result) ** 2)


class Test_Run:
    '''
    Runs all tests when used with pytest.\n
    '''

    def test_everything(self, qubits=5, h_target=0, x_target=0, z_target=0, swap_targets=(0, 1), test_matrix_1=None,
                        test_matrix_2=None, multiple=5, col_target=0, row_target=0, value_target=(0, 0),
                        grover_target=0):
        '''
        @param qubits: size of quantum registers used for tests
        @param h_target/x_target/z_target: target qubits of h-gate/x-gate/z-gate tests respectively
        @param swap_targets: target qubits of swap-gate test
        @param test_matrix_1, test_matrix_2: matrices to test sparse matrix method. Will be randomly generated if either is None.
        @param multiple: scalar to use for testing the sparse matrix multiply method
        @param col_target/row_target: target column/row respecitvely for get_col and get_row tests
        @param value_target: target value for get_value test
        @param grover_target: target state for grover search testing
        '''
        Gate_Testing(qubits).gate_test("h", h_target)
        Gate_Testing(qubits).gate_test("x", x_target)
        Gate_Testing(qubits).gate_test("z", z_target)
        Gate_Testing(qubits).gate_test("swap", *swap_targets)

        Sparse_Testing(test_matrix_1, test_matrix_2).basic_sparsify_test()
        Sparse_Testing(test_matrix_1, test_matrix_2).sparse_dot_test()
        Sparse_Testing(test_matrix_1, test_matrix_2).sparse_tensor_dot_test()
        Sparse_Testing(test_matrix_1, test_matrix_2).sparse_multiply_test(multiple)
        Sparse_Testing(test_matrix_1, test_matrix_2).sparse_minus_test()
        Sparse_Testing(test_matrix_1, test_matrix_2).sparse_transpose_test()
        Sparse_Testing(test_matrix_1, test_matrix_2).get_attribute_test("col", col_target)
        Sparse_Testing(test_matrix_1, test_matrix_2).get_attribute_test("row", row_target)
        Sparse_Testing(test_matrix_1, test_matrix_2).get_attribute_test("value", *value_target)
        Sparse_Testing(test_matrix_1, test_matrix_2).get_attribute_test("nonzero_rows")
        Sparse_Testing(test_matrix_1, test_matrix_2).get_attribute_test("nonzero_cols")

        Grover_Testing(qubits).grover_test(grover_target)
