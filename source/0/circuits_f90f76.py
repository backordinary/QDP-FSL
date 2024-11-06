# https://github.com/kacy-zh/2021-Quantum-Winter-Adders/blob/d0962f42051de59be6eb4cb0f37ef2ed99efcc8c/Adders/circuits.py
import numpy as np
import qiskit as q

class circuit_builder:
    """ Builds the specified circuit 
    
    init:
        qubits: The number of qubits in the circuit
        inputs: A list of all the gates in the circuit in chronological order
    
    """

    def __init__(self, qubits, inputs):

        # Associated variables
        self.gates = []
        self.matrices = []
        self.matrix = None
        self.inputs = inputs
        self.qubits = qubits
        
        # For each gate, add it
        for gate in inputs:
            cnot_matrix = cnot(gate[1]).get_matrix()
            self.gates.append(cnot_matrix)
            intermediate_matrix = self.step_matrix(gate[0], cnot_matrix, gate[2])
            self.matrices.append(intermediate_matrix)
        
        # Multiply all matrices
        self.matrix = self.matrices.pop(0)
        for i in self.matrices:
            self.matrix = np.matmul(i, self.matrix)
        
    def get_quantum_circuit(self, initial_ones):
        """ Returns the qiskit quantum circuit 

        Args:
            initial_ones: A list [0...1] which specifies which qubits start in the state |0> or |1>
        """
        quantum_circuit = q.QuantumCircuit(self.qubits, self.qubits)
        
        # Initiliase states
        for i in initial_ones:
            quantum_circuit.initialize([0, 1], i)
        
        # Add in gates
        for gate in self.inputs:
            controls = [x-1 for x in gate[1][0]]
            controls = [x+gate[0] for x in controls]
            quantum_circuit.mct(controls, gate[0] + gate[1][1] - 1)
        
        # At the end, measure
        quantum_circuit.measure(list(range(self.qubits)), list(range(self.qubits)))
            
        return quantum_circuit
            
    def step_matrix(self, iden_start, ccnot, iden_end):
        """ Produces I (x) ccnot (x) I
    
        Args:
            iden_start: Number of I at the start
            ccnot: A gate in matrix form
            iden_end: Number of I at the end
        """
        start = mt.identity_matrix(pow(2,iden_start))
        final = np.kron(start, ccnot)
        end = mt.identity_matrix(pow(2,iden_end))
        return np.kron(final, end)

    def get_matrix(self):
        """ Returns the numerical matrix """
        return self.matrix

class mt:
    """ Matrix helper functions """
    
    # Create an empty matrix
    def empty_matrix(w, h):
        return [[0 for x in range(w)] for y in range(h)] 

    # Create the identity matrix
    def identity_matrix(w):
        m = [[0 for x in range(w)] for y in range(w)] 
        for i in range(w):
            m[i][i] = 1
        return m

    # Prints a matrix
    def print_matrix(m):
        for row in m:
            for cell in row:
                print(cell," ",end="")
            print()

class gate:
    """ Simple gate class """

    def set_matrix(self, inputs):
        pass
    
    def get_matrix(self):
        return self.matrix
    
class cnot(gate):
    """ cnot, ccnot etc. gate class """

    def __init__(self, inputs):
        self.set_matrix(inputs)
        
    def set_matrix(self, inputs):
        self.controls = inputs[0]
        self.target = inputs[1]
        self.matrix = self.cnot_builder(self.controls, self.target)
        
    # Builds the matrix for the cnot with an array of controls and a target
    def cnot_builder(self, controls, target):

        # Find the maximum number
        if len(controls) > 0:
            maximum = max(controls)
        else:
            maximum = target
            
        if target > maximum:
            maximum = target

        # Find the missing values
        missing = [1] * (maximum)
        missing[target-1] = 0

        # Keep a sum of the constant values
        constant_val = 0

        # Find not missing and update constant_vals
        for i in controls:
            missing[i-1] = 0   
            constant_val += pow(2,maximum-i)

        # If there is a missing number then update its value
        for i in range(len(missing)):
            if missing[i] == 1:
                missing[i] = pow(2,maximum-i-1)

        # Truncate 0s
        missing = [i for i in missing if i != 0]  

        # Permute sum calculates what pairs get swapped
        target_val = pow(2,maximum-target)
        permute_sum = self.permute_sums(None, missing)
        for i in range(len(permute_sum)):
            permute_sum[i] = [constant_val + permute_sum[i], constant_val + permute_sum[i] + target_val]

        # Creates the matrix
        matrix_length = pow(2,maximum)
        matrix = mt.identity_matrix(matrix_length)

        # Make necessary changes to the cnot matrix
        for pairs in permute_sum:
            i = pairs[0]
            j = pairs[1]
            matrix[i][i] = 0
            matrix[j][j] = 0
            matrix[i][j] = 1
            matrix[j][i] = 1

        return matrix

    # Permutes the sums of all numbers in current_list with each other
    def permute_sums(self, answer, current_list):
        # If no list given just return 0
        if len(current_list) == 0:
            return [0]

        # If at start then start off list
        elif answer is None:
            answer = [0, current_list[0]]

        # Add 0 and current_list[0] to all entries
        else:
            add = answer[:]
            for i in range(len(add)):
                add[i] += current_list[0]
            answer.extend(add)

        # If last, just return answer
        if len(current_list) == 1:
            return answer

        # If not last, continue and remove one from the current list
        return self.permute_sums(answer, current_list[1:])


