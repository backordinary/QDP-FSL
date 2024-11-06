# https://github.com/Qonic-Team/qonic/blob/f876de99c3e0ffae53b8c02572c18ccce2edcd13/source_dir/build/lib/qonic/QProgram/QProgram.py
# the tequila library will be used to construct quantum circuits
import tequila as tq
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.exceptions import CircuitError
import qiskit

from forest.benchmarking.operator_tools import kraus2superop, vec, unvec

from math import sqrt, log2
import numpy as np
from numpy import pi, e
import copy

# define a class structure for quantum programs using the syntax of quantum hoare logic
class QProgram:
    def skip(): # skip operation does nothing
        pass
    
    def __init__(self, q̅=[[1, 0]], backend=None):
        if hasattr(np.shape(q̅), '__len__') and len(np.shape(q̅)) > 1:
            self.q̅ = np.array(1) # register statevector
            for qubit in q̅:
                self.q̅ = np.kron(self.q̅, qubit)
        else:
            self.q̅ = q̅ # quantum register
            
        self.instructions = [] # list of instructions (operations) carried out by the program
            
        self.num_qubits = int(log2(len(self.q̅)))
        
        
        self.M_k = [] # list of terminating operations of the program as Kraus operators M_k
        
        self.backend = backend
        
    def assign(self, U, variables): # function corresponding to q:=Uq or q̅:=Uq̅
        self.instructions.append((U, variables))

    def to_circuit(self, framework='tequila'): # method to return a quantum circuit defined in tequila or qiskit from the current QProgram object
        if framework == 'tequila':
            qc = tq.QCircuit()
            
            qc_initializer = tq.QCircuit()
            initializer = QuantumCircuit(self.num_qubits) # create an initializer circuit in qiskit before converting it to tequila
            for i in range(self.num_qubits):
                # make sure qubit is initialized on tequila circuit if it remains in state [1, 0]
                qc_initializer += tq.gates.u3(0, 0, 0, i)
            initializer.initialize(self.q̅, range(self.num_qubits))

            transpiled = transpile(initializer, basis_gates=['cx', 'u3']) # convert to qiskit through open qasm
            qasm = transpiled.qasm()
            for instruction in qasm.split('\n'):
                if instruction[:2] == 'u3':
                    ϴ, φ, λ = map(eval, instruction.replace('u3', '').replace('pi', str(pi)).split(' ')[0].replace('(', '').replace(')', '').split(','))
                    q = list(map(int, instruction.replace(';', '').replace('q', '').replace('[', '').replace(']', '').split(' ')[1].split(',')))
                    qc_initializer += tq.gates.u3(theta=ϴ, phi=φ, lambd=λ, target=q)
                elif instruction[:2] == 'cx':
                    q = list(map(int, instruction.replace(';', '').replace('q', '').replace('[', '').replace(']', '').split(' ')[1].split(',')))
                    qc_initializer += tq.gates.CX(control=q[0], target=q[1])
            qc += qc_initializer
            
        elif framework == 'qiskit':
            qc = QuantumCircuit(self.num_qubits)
            
            qc_initializer = QuantumCircuit(self.num_qubits)
            qc_initializer.initialize(self.q̅, range(self.num_qubits))
            qc += qc_initializer
            
        for instruction in self.instructions:
            variables = instruction[1]
            U = instruction[0]
            
            gate_circuit = QuantumCircuit(self.num_qubits)
            gate_circuit.unitary(U, variables)
            transpiled = transpile(gate_circuit, basis_gates=['cx', 'u3']) # for each assignment instruction, reduce the corresponding unitary operation to cx and u3 gates
            
            if framework == 'tequila':
                # get the qiskit gate in open qasm
                qasm = transpiled.qasm()
                for instruction in qasm.split('\n'):
                    if instruction[:2] == 'u3':
                        ϴ, φ, λ = map(eval, instruction.replace('u3', '').replace('pi', str(pi)).split(' ')[0].replace('(', '').replace(')', '').split(','))
                        q = list(map(int, instruction.replace(';', '').replace('q', '').replace('[', '').replace(']', '').split(' ')[1].split(',')))
                        qc += tq.gates.u3(theta=ϴ, phi=φ, lambd=λ, target=q)
                    elif instruction[:2] == 'cx':
                        q = list(map(int, instruction.replace(';', '').replace('q', '').replace('[', '').replace(']', '').split(' ')[1].split(',')))
                        qc += tq.gates.CX(control=q[0], target=q[1])
            elif framework == 'qiskit':
                qc += transpiled
                
        return qc
    
    def to_unitary(self): # method to return the instructions as a single unitary matrix
        qc = QuantumCircuit(self.num_qubits)
        for instruction in self.instructions:
            variables = instruction[1]
            u = instruction[0]
            qc.unitary(u, variables) # map the program to a quantum circuit
        U = Operator(qc).data # get the corresponding unitary matrix
        return U

    def measure(self, q_m): # function corresponding to measuring a set of qubits
        # returns the result of the measurement and modifies the instructions so that they map the initial state to the state after the measurement
        
        # run the program and get the output as a statevector
        qc = self.to_circuit(framework='tequila')
        if self.backend != None:
            state = tq.simulate(qc, backend=self.backend)
            measurement = tq.simulate(qc, samples=1, read_out_qubits=q_m, backend=self.backend).state
        else:
            state = tq.simulate(qc)
            measurement = tq.simulate(qc, samples=1, read_out_qubits=q_m).state
            
        res = list(measurement.keys())[0] # get the collapsed state of the qubit
        
        valid_basis = dict() # dictionary of valid basis states given the measured value of q_m
        norm_sum = 0 # stores the sum of each valid probability amplitude times its conjugate so the calculated state can be normalized
        for key in state.state.keys():
            # check if the bits in the binary form of this basis corresponding to the qubits being measured are equal to the measured states
            if all([('{:0'+str(self.num_qubits)+'d}').format(int(format(int(key), 'b')))[q_m[i]] == str(res[i]) for i in range(len(q_m))]):
                valid_basis[key] = state.state[key]
                norm_sum += (state.state[key]*np.conj(state.state[key]))
                
        # normalize the new vector
        for key in valid_basis.keys():
            valid_basis[key] *= sqrt(1/norm_sum)
            
        # convert the valid_basis dict into a statevector array
        final_state = QubitWaveFunction(valid_basis, self.num_qubits).to_array()
        
        # replace the quantum register with the final state and clear the instructions (reflecting that measurements are non unitary)
        self.q̅ = final_state
        self.instructions = []
        
        return ('{:0'+str(len(q_m))+'d}').format(int(format(int(res), 'b'))) # return the measured state
        
    def while_loop_body(self, q_m, expected_measurement : str, sub_algorithm, prob_threshold=0.01, iter_cutoff=10):
        n = 1 # keeps track of prob of each loop iteration
        iters = 0
        program = self # run the initial program before entering the loop body
        
        # make sure sub_algorithm has the same number of variables as the current program
        if self.num_qubits != program.num_qubits:
            raise ValueError('the passed sub_algorithm has a quantum register with a different length of the current program')
        
        # create a unitary from the sub_algorithm by having each instruction controlled by the qubits in q_m
        circuit_instructions = sub_algorithm.to_circuit(framework='qiskit').data
        sub_algorithm_circuit = QuantumCircuit(self.num_qubits)
        for instruction in circuit_instructions:
            if instruction.operation.name == 'initialize':
                continue # skip over initialization instruction
            try:
                gate = instruction.operation.definition.control(len(q_m)) # controlled instruction
                    
                for qubit in range(len(q_m)):
                    if expected_measurement[qubit] == '0': # flip qubits that are expected to be 0
                        sub_algorithm_circuit.x(q_m[qubit])
                            
                sub_algorithm_circuit.append(gate, q_m + [q for q in list(instruction.qubits)])
                    
                for qubit in range(len(q_m)):
                    if expected_measurement[qubit] == '0':
                        sub_algorithm_circuit.x(q_m[qubit])
                    
            except (CircuitError, ValueError):
                raise ValueError('an instruction was applied to one of the control qubits in the sub algorithm')
        # calculate the unitary of the sub algorithm
        U = Operator(sub_algorithm_circuit).data
        
        # final state that is returned after the loop is carried out
        final_state = self.q̅
        
        while(n > prob_threshold and iters < iter_cutoff):
            # get the probability of the qubits in q_m being equal to the expected_measurement
            qc = program.to_circuit(framework='tequila')
            
            if self.backend != None:
                state = tq.simulate(qc, backend=self.backend)
            else:
                state = tq.simulate(qc)
                
            # get the prob from the prob amplitudes of basis components satisfying the conditional
            prob = 0
            for key in state.state.keys():
                # check if the bits in the binary form of this basis corresponding to the qubits being measured are equal to the measured states
                if all([('{:0'+str(self.num_qubits)+'d}').format(int(format(int(key), 'b')))[q_m[i]] == str(expected_measurement[i]) for i in range(len(q_m))]):
                    prob += abs(state.state[key]*np.conj(state.state[key]))
            n *= prob # update the conditional prob
             
            program = copy.deepcopy(sub_algorithm) # update the sub algorithm
            program.q̅ = state.to_array() # update the sub algorithm's register to the state after this many interations
            
            # modify all operations applied within the sub algorithm (ie. 'program') to be controlled according to the loop conditional
            program.instructions = []
            self.assign(U, range(self.num_qubits)) # add the unitary for this step to the list of instructions     
            iters += 1
            
            # update final_state
            final_state = state.to_array()
        
        return final_state

    def to_superop(self): # function to return the super operator form of the current QProgram
        return kraus2superop([self.to_unitary()])

    def wlp(self, Q): # function to compute the weakest liberal precondition for the current QProgram given a postcondition Q
        if (np.shape(Q)[0] != np.shape(Q)[1] or np.shape(Q)[0] != len(self.q̅)):
            raise ValueError('the passed postcondition has a different shape than the current program\'s unitary')

        S_star = np.conj(self.to_superop()) # the complex conjugate of the QProgram in super operator form: ⟦S⟧*
        dif = np.identity(len(self.q̅)) - Q # the difference (I - Q)
        P_dash = np.identity(len(self.q̅)) - unvec(np.matmul(S_star, vec(dif))) # calculate P' using I - ⟦S⟧*(I-Q)
        return P_dash

    def correct_q_hoare_triple(self, Q, P): # function that checks if the QHoare triple {Q}S{P} is valid
        if (np.shape(Q)[0] != np.shape(Q)[1] or np.shape(Q)[0] != len(self.q̅)):
            raise ValueError('the passed postcondition has a different shape than the current program\'s unitary')

        if (np.shape(Q)[0] != np.shape(P)[1] or np.shape(P)[0] != len(self.q̅)):
            raise ValueError('the passed precondition has a different shape than the current program\'s unitary')

        P_dash = self.wlp(Q)

        # check to see if P ⊑ P' (where ⊑ denotes a Löwner partial order)
        dif = P_dash - P
        if (np.all(np.linalg.eigvals(dif) >= -0.000001)): # the partial order is satisfied
            return True # the program is correct
        else:
            return False # the program is not correct