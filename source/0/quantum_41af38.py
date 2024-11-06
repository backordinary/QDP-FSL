# https://github.com/wuben3125/qrwls/blob/bfd7df8ab90369e377893e2e23e50a38c9e15b85/quantum.py
"""

Code for discrete quantum random walks

"""

### imports
import numpy as np
import random as rand
import qiskit
import qiskit.test.mock
from qiskit.providers.aer.noise import NoiseModel
from common import * # for int_to_binarylist and binarylist_to_int
import glv
### imports


class QuantumRandomWalkHamming:
    
    import numpy as np
    import qiskit
    
    def __init__(self, angle_list):
        """
        angle_list: list of [theta, phi, lambda] triples, starting with index n-1 and ending with 0 (left to right)
        
        """
        self.angle_list = angle_list
        
        self.n = len(angle_list)
       
    
    def p_matrix(self):
        """
        generates the appropriate numpy matrix
        
        """
        matrix = np.ones( (2**self.n, 2**self.n) ) # start with ones, values will be multiplied
        
        for i in range(2**self.n): # rows
            for j in range(2**self.n): # columns
                
                xor_int = i  ^ j
                
                xor_binarylist = int_to_binarylist(xor_int, self.n)
                
                # test pairs n-1,n-2; n-2,n-3; ...; 1,0
                for idx in range(len(xor_binarylist) - 1):
                    
                    power = xor_binarylist[idx] ^ xor_binarylist[idx+1]
                    
                    if power == 1:
                        matrix[i][j] *= np.sin(self.angle_list[idx][0]/2)**2 # sin(theta_idx/2) ^ 2
                    elif power == 0:
                        matrix[i][j] *= np.cos(self.angle_list[idx][0]/2)**2 # cos(theta_idx/2) ^ 2
                    else:
                        raise Exception("Value error")
                        
                # test pair 0,-1
                idx = len(xor_binarylist) - 1
                power = xor_binarylist[idx]
                
                if power == 1:
                    matrix[i][j] *= np.sin(self.angle_list[idx][0]/2)**2
                elif power == 0:
                    matrix[i][j] *= np.cos(self.angle_list[idx][0]/2)**2
                else:
                    raise Exception("Value error")
                
        return matrix
            
        
    def random_step(self, initial_node):
        """
        initial_node: an int as a binary_list
        
        returns the final node as a binary_list
        """
        
        ## instantiate circuit stuff
        qr = qiskit.QuantumRegister(self.n + 1) # nth index qubit is coin 
        cr = qiskit.ClassicalRegister(self.n)
        
        qc = qiskit.QuantumCircuit(qr, cr)
        
        ## initialize registers according to initial_node
        for idx, bit in enumerate(initial_node[-1::-1]):
            if bit == 1:
                qc.x(idx)
            elif bit == 0:
                pass
            else:
                raise Exception("Value error")
        
        
        ## perform coin flip unitary operations and cnots
        # iterate from back to front, since self.angle_list goes from n-1 to 0
        for idx, triple in enumerate(self.angle_list[-1::-1]):
            
            # apply u3 to coin qubit
            qc.u3(triple[0], triple[1], triple[2], self.n) # (theta, phi, lambda, coin_idx)
            
            # apply cnot, with control on coin qubit, target on idx-th qubit 
            qc.cnot(self.n, idx) # (control, target)
        
        # apply measurement operation
        qc.measure(qr[0:self.n], cr[0:self.n]) # exclude coin qubit
        
        ## readout final result, then convert to binary_list
        # use qasm simulator by default
        result = qiskit.execute(qc, backend = qiskit.Aer.get_backend('qasm_simulator'), shots = 1).result()
        counts = result.get_counts()
        
#         print(counts)
        
        # convert dict_keys to list, then get key
        final_node_str = list(counts.keys())[0] # returned as str
        
        final_node_list = [digit_char for digit_char in final_node_str]
        
        final_node_binarylist = [int(digit_char) for digit_char in final_node_list]
        
        return final_node_binarylist
        
    def random_walk(self, initial_node, num_steps):
        """
        initial_node: an int as a binary_list
        num_steps: int
        
        """
        
        current_node = initial_node
        
        for step in range(num_steps):
            
            current_node = self.random_step(current_node)
            
        return current_node
    
    
    
## code for qrw linear solver
def x_component_qrw(angle_list, gamma, b, x_component_index, c, d):
    """
    angle_list: list of [theta, phi, lambda] triples that correspond to coin unitaries
                    listed n-1, n-2, ..., 0 from left to right
        
        for A = 1 - gamma*P,
        
        A @ x = b
        
        A_inv = sum_{s=0,...,inf} (gamma^s * P^s)
        A_inv_c = sum_{s=0,...,c} (gamma^s * P^s)
        
        x_c = A_inv_c @ b
        
    b: 1D list or np.array representing end result vector
    x_component_index: int
    c: int; cutoff 
    d: int; number of times to perform each step - a value from 100 to 10000 can be used 
    
    returns the appropriate index value of x_c (truncated up to error ~ gamma^c)
    """
    qrw = QuantumRandomWalkHamming(angle_list)
    
    prep_node = int_to_binarylist(x_component_index, num_digits = qrw.n)
    
    x_component = 0
    
    for s in range(0, c+1): # number of steps starting at 0
        
        total_sum = 0
        
        for iter in range(d):
            final_node = qrw.random_walk(prep_node, s)
            final_node_int = binarylist_to_int(final_node)
            total_sum += b[final_node_int]
                    
        avg_sum = total_sum / d
        
        avg_sum *= gamma**s
        
        ## add to component value
        x_component += avg_sum
        
    return x_component


class QuantumRandomWalkHamming_Optimized:
    
    import numpy as np
    import qiskit
    
    def __init__(self, angle_list):
        """
        angle_list: list of [theta, phi, lambda] triples, starting with index n-1 and ending with 0 (left to right)
        
        """
        self.angle_list = angle_list
        
        self.n = len(angle_list)
       
    
    def p_matrix(self):
        """
        generates the appropriate numpy matrix
        
        """
        matrix = np.ones( (2**self.n, 2**self.n) ) # start with ones, values will be multiplied
        
        for i in range(2**self.n): # rows
            for j in range(2**self.n): # columns
                
                xor_int = i  ^ j
                
                xor_binarylist = int_to_binarylist(xor_int, self.n)
                
                # test pairs n-1,n-2; n-2,n-3; ...; 1,0
                for idx in range(len(xor_binarylist) - 1):
                    
                    power = xor_binarylist[idx] ^ xor_binarylist[idx+1]
                    
                    if power == 1:
                        matrix[i][j] *= np.sin(self.angle_list[idx][0]/2)**2 # sin(theta_idx/2) ^ 2
                    elif power == 0:
                        matrix[i][j] *= np.cos(self.angle_list[idx][0]/2)**2 # cos(theta_idx/2) ^ 2
                    else:
                        raise Exception("Value error")
                        
                # test pair 0,-1
                idx = len(xor_binarylist) - 1
                power = xor_binarylist[idx]
                
                if power == 1:
                    matrix[i][j] *= np.sin(self.angle_list[idx][0]/2)**2
                elif power == 0:
                    matrix[i][j] *= np.cos(self.angle_list[idx][0]/2)**2
                else:
                    raise Exception("Value error")
                
        return matrix
            
        
    def random_step(self, initial_node):
        """
        initial_node: an int as a binary_list
        
        returns the final node as a binary_list
        """
        
        ## instantiate circuit stuff
        qr = qiskit.QuantumRegister(self.n + 1) # nth index qubit is coin 
        cr = qiskit.ClassicalRegister(self.n)
        
        qc = qiskit.QuantumCircuit(qr, cr)
        
        ## initialize registers according to initial_node
        for idx, bit in enumerate(initial_node[-1::-1]):
            if bit == 1:
                qc.x(idx)
            elif bit == 0:
                pass
            else:
                raise Exception("Value error")
        
        
        ## perform coin flip unitary operations and cnots
        # iterate from back to front, since self.angle_list goes from n-1 to 0
        for idx, triple in enumerate(self.angle_list[-1::-1]):
            
            # apply u3 to coin qubit - ignore trivial gates
            if triple[0] == triple[1] == triple[2] == 0:
                pass
            else:
                qc.u3(triple[0], triple[1], triple[2], self.n) # (theta, phi, lambda, coin_idx)
            
            # apply cnot, with control on coin qubit, target on idx-th qubit 
            qc.cnot(self.n, idx) # (control, target)
        
        # apply measurement operation
        qc.measure(qr[0:self.n], cr[0:self.n]) # exclude coin qubit
        
        ## readout final result, then convert to binary_list
        # use qasm simulator by default
        result = qiskit.execute(qc, backend = qiskit.Aer.get_backend('qasm_simulator'), shots = 1, optimization_level=3).result()
        counts = result.get_counts()
        
#         print(counts)
        
        # convert dict_keys to list, then get key
        final_node_str = list(counts.keys())[0] # returned as str
        
        final_node_list = [digit_char for digit_char in final_node_str]
        
        final_node_binarylist = [int(digit_char) for digit_char in final_node_list]
        
        return final_node_binarylist
        
    def random_walk(self, initial_node, num_steps):
        """
        initial_node: an int as a binary_list
        num_steps: int
        
        """
        
        current_node = initial_node
        
        for step in range(num_steps):
            
            current_node = self.random_step(current_node)
            
        return current_node
    
    
    
## code for qrw linear solver
def x_component_qrw_optimized(angle_list, gamma, b, x_component_index, c, d):
    """
    angle_list: list of [theta, phi, lambda] triples that correspond to coin unitaries
                    listed n-1, n-2, ..., 0 from left to right
        
        for A = 1 - gamma*P,
        
        A @ x = b
        
        A_inv = sum_{s=0,...,inf} (gamma^s * P^s)
        A_inv_c = sum_{s=0,...,c} (gamma^s * P^s)
        
        x_c = A_inv_c @ b
        
    b: 1D list or np.array representing end result vector
    x_component_index: int
    c: int; cutoff 
    d: int; number of times to perform each step - a value from 100 to 10000 can be used 
    
    returns the appropriate index value of x_c (truncated up to error ~ gamma^c)
    """
    qrw = QuantumRandomWalkHamming_Optimized(angle_list)
    
    prep_node = int_to_binarylist(x_component_index, num_digits = qrw.n)
    
    x_component = 0
    
    for s in range(0, c+1): # number of steps starting at 0
        
        total_sum = 0
        
        for iter in range(d):
            final_node = qrw.random_walk(prep_node, s)
            final_node_int = binarylist_to_int(final_node)
            total_sum += b[final_node_int]
                    
        avg_sum = total_sum / d
        
        avg_sum *= gamma**s
        
        ## add to component value
        x_component += avg_sum
        
    return x_component

def reset_glv():
    glv.total_valid_steps = 0
    glv.total_invalid_steps = 0
    glv.total_retried_steps = 0
    

class QuantumRandomWalkHamming_Noisy:
    
    import numpy as np
    import qiskit
    
    def __init__(self, angle_list, backend = 'fake_casablanca', test_invalid_steps = True):
        """
        angle_list: list of [theta, phi, lambda] triples, starting with index n-1 and ending with 0 (left to right)
        
        """
        self.angle_list = angle_list
        
        self.n = len(angle_list)
        
        self.backend = backend ## default value with no input is fake_vigo
        
        self.test_invalid_steps = test_invalid_steps
        
        self.num_valid_steps = 0
        self.num_invalid_steps = 0
        self.num_retried_steps = 0  
         
         
         # set device once
        if self.backend == 'fake_vigo':
            self.device_backend = qiskit.test.mock.FakeVigo()
        elif self.backend == 'fake_melbourne':
            self.device_backend = qiskit.test.mock.FakeMelbourne()
        elif self.backend == 'fake_paris':
            self.device_backend = qiskit.test.mock.FakeParis()
        elif self.backend =='fake_boeblingen':
            self.device_backend = qiskit.test.mock.FakeBoeblingen()
        elif self.backend == 'fake_casablanca':
            self.device_backend = qiskit.test.mock.FakeCasablanca()
        
        else:
            raise Exception("Backend not recognized")
        
        
        self.coupling_map = self.device_backend.configuration().coupling_map
        self.noise_model = NoiseModel.from_backend(self.device_backend)
        self.basis_gates = self.noise_model.basis_gates
       
    
    def p_matrix(self):
        """
        generates the appropriate numpy matrix
        
        """
        matrix = np.ones( (2**self.n, 2**self.n) ) # start with ones, values will be multiplied
        
        for i in range(2**self.n): # rows
            for j in range(2**self.n): # columns
                
                xor_int = i  ^ j
                
                xor_binarylist = int_to_binarylist(xor_int, self.n)
                
                # test pairs n-1,n-2; n-2,n-3; ...; 1,0
                for idx in range(len(xor_binarylist) - 1):
                    
                    power = xor_binarylist[idx] ^ xor_binarylist[idx+1]
                    
                    if power == 1:
                        matrix[i][j] *= np.sin(self.angle_list[idx][0]/2)**2 # sin(theta_idx/2) ^ 2
                    elif power == 0:
                        matrix[i][j] *= np.cos(self.angle_list[idx][0]/2)**2 # cos(theta_idx/2) ^ 2
                    else:
                        raise Exception("Value error")
                        
                # test pair 0,-1
                idx = len(xor_binarylist) - 1
                power = xor_binarylist[idx]
                
                if power == 1:
                    matrix[i][j] *= np.sin(self.angle_list[idx][0]/2)**2
                elif power == 0:
                    matrix[i][j] *= np.cos(self.angle_list[idx][0]/2)**2
                else:
                    raise Exception("Value error")
                
        return matrix
            
        
    def random_step(self, initial_node):
        """
        initial_node: an int as a binary_list
        
        returns the final node as a binary_list, OR returns False if step is "invalid"
        """
        
        ## instantiate circuit stuff
        qr = qiskit.QuantumRegister(self.n + 1) # nth index qubit is coin 
        cr = qiskit.ClassicalRegister(self.n)
        
        qc = qiskit.QuantumCircuit(qr, cr)
        
        ## initialize registers according to initial_node
        for idx, bit in enumerate(initial_node[-1::-1]):
            if bit == 1:
                qc.x(idx)
            elif bit == 0:
                pass
            else:
                raise Exception("Value error")
        
        
        ## perform coin flip unitary operations and cnots
        # iterate from back to front, since self.angle_list goes from n-1 to 0
        for idx, triple in enumerate(self.angle_list[-1::-1]):
            
            # apply u3 to coin qubit
            qc.u(triple[0], triple[1], triple[2], self.n) # (theta, phi, lambda, coin_idx)
            
            # apply cnot, with control on coin qubit, target on idx-th qubit 
            qc.cnot(self.n, idx) # (control, target)
        
        # apply measurement operation
        qc.measure(qr[0:self.n], cr[0:self.n]) # exclude coin qubit
        
        ## readout final result, then convert to binary_list
        # use qasm simulator by default
        
        
        result = qiskit.execute(qc, backend = qiskit.Aer.get_backend('qasm_simulator'),
                                noise_model = self.noise_model, 
                                coupling_map = self.coupling_map, 
                                basis_gates = self.basis_gates, shots = 1).result()
        counts = result.get_counts()
        
#         print(counts)
        
        # convert dict_keys to list, then get key
        final_node_str = list(counts.keys())[0] # returned as str
        
        final_node_list = [digit_char for digit_char in final_node_str]
        
        final_node_binarylist = [int(digit_char) for digit_char in final_node_list]
        
        ## test for invalid quantum random walk
        initial_idx = binarylist_to_int(initial_node)
        incident_idx = binarylist_to_int(final_node_binarylist)

        if self.p_matrix()[initial_idx, incident_idx] == 0:
            
            if self.test_invalid_steps == True:
                self.num_retried_steps += 1
                glv.total_retried_steps += 1
                return False # invalid step
            else:
                self.num_invalid_steps += 1
                glv.total_invalid_steps += 1
                #print('invalid step (ignored)')
        else:
            self.num_valid_steps += 1
            glv.total_valid_steps += 1
            #print('valid step')
        
        return final_node_binarylist
        
    def random_walk(self, initial_node, num_steps):
        """
        initial_node: an int as a binary_list
        num_steps: int
        
        """
        
        current_node = initial_node
        
        step_idx = 0
        while step_idx < num_steps:
            
            step_result = self.random_step(current_node)
            
            if step_result == False:
                #print('invalid step, retrying...')
                pass # redo
            else:
                step_idx += 1
                current_node = step_result # move to next step
                
        return current_node
    
    def reset_step_counts(self):
        self.num_valid_steps = 0
        self.num_invalid_steps = 0
        
        
    def display_step_counts(self):
        print(f"num valid steps: {self.num_valid_steps}")
        print(f"num invalid steps: {self.num_invalid_steps}")
        print(f"num retried steps: {self.num_retried_steps}")


## code for qrw linear solver
def x_component_qrw_noisy(angle_list, gamma, b, x_component_index, c, d, backend = 'fake_vigo', test_invalid_steps=True):
    """
    angle_list: list of [theta, phi, lambda] triples that correspond to coin unitaries
                    listed n-1, n-2, ..., 0 from left to right
        
        for A = 1 - gamma*P,
        
        A @ x = b
        
        A_inv = sum_{s=0,...,inf} (gamma^s * P^s)
        A_inv_c = sum_{s=0,...,c} (gamma^s * P^s)
        
        x_c = A_inv_c @ b
        
    b: 1D list or np.array representing end result vector
    x_component_index: int
    c: int; cutoff 
    d: int; number of times to perform each step - a value from 100 to 10000 can be used 
    
    returns the appropriate index value of x_c (truncated up to error ~ gamma^c)
    """
    
    # print output
    #print(f"x component index: {x_component_index}")
    
    # local variable
    step_counts = np.array([0, 0, 0]) # valid, invliad, retried
    
    qrw = QuantumRandomWalkHamming_Noisy(angle_list, backend, test_invalid_steps=test_invalid_steps)
    
    qrw.reset_step_counts() # zero the step counts
    
    prep_node = int_to_binarylist(x_component_index, num_digits = qrw.n)
    
    x_component = 0
    
    for s in range(0, c+1): # number of steps starting at 0
        
        total_sum = 0
        
        for iter in range(d):
            final_node = qrw.random_walk(prep_node, s)
            final_node_int = binarylist_to_int(final_node)
            total_sum += b[final_node_int]
                    
        avg_sum = total_sum / d
        
        avg_sum *= gamma**s
        
        ## add to component value
        x_component += avg_sum
        
    # display print output
    #qrw.display_step_counts()
    
#    print(f"total_valid_steps: {glv.total_valid_steps}")
#    print(f"total_invalid_steps: {glv.total_invalid_steps}")
#    print(f"total_retried_steps: {glv.total_retried_steps}")

    step_counts[0] = glv.total_valid_steps
    step_counts[1] = glv.total_invalid_steps
    step_counts[2] = glv.total_retried_steps
        
    return x_component, step_counts
    

def permutation_up(n):
    """
    n: int
    
    returns an nxn permutation matrix that shifts elements 
        in a column vector up one space
    """
    
    import numpy as np
    
    perm = np.zeros([n, n])
    
    ## instantiate 1s as if an identity matrix had columns
        # shifted over one to the right
        
    for i in range(n):
        perm[ (i-1)%n , i ] = 1
        
    return perm


def permutation_down(n):
    """
    n: int
    
    returns an nxn permutation matrix that shifts elements 
        in a column vector down one space
    """
    
    import numpy as np
    
    perm = np.zeros([n, n])
    
    ## instantiate 1s as if an identity matrix had columns
        # shifted over one to the left
        
    for i in range(n):
        perm[ (i+1)%n , i ] = 1
        
    return perm

## convert binary string to integer
def bin_to_dec(bin_str):
    
    dec = 0
    
    pow_2 = 0
    
    for i in range(len(bin_str)-1, -1, -1):
        if bin_str[i] == '1':
            dec += 2**pow_2
            
        pow_2 += 1
        
    return dec

def quantum_walk(n, num_steps, coin_unitary, num_shots):
    """
    a 1-D quantum walk with wraparound boundary conditions
    
    n: int - number of qubits s.t. there are 2^n nodes
    num_steps: int
    coin_unitary: matrix 
    
    returns a list of final measured positions
    """
    
    qr = qiskit.QuantumRegister(n+1) # last qubit is for coin
    cr = qiskit.ClassicalRegister(n)
    
    qc = qiskit.QuantumCircuit(qr, cr)
    
    ## instantiate state in the middle, but slightly to the right
    qc.x( qr[n-1] )
    
    ## create coin gate from coin matrix
    coin_gate = qiskit.quantum_info.operators.Operator(coin_unitary)
    coin_gate = coin_gate.to_instruction()
    
    ## create gate for the walking
    proj_0 = np.matrix([[1,0], [0,0]])
    proj_1 = np.matrix([[0,0], [0,1]])
    
    left_walk_matrix = np.kron(proj_0, permutation_up(2**n)) # only multiplies if the coin qubit is |0>
    right_walk_matrix = np.kron(proj_1, permutation_down(2**n)) # only multiplies if the coin qubit is |1>
    
    walk_matrix = left_walk_matrix + right_walk_matrix
    
    walk_gate = qiskit.quantum_info.operators.Operator(walk_matrix)
    walk_gate = walk_gate.to_instruction()
    
    ### instantiate coin register
    
    ## start as |0> --> will drift to left
#     qc.i(qr[n])
    
    ## start as |1> --> will drift to right    
#     qc.x(qr[n]) 

    ## start as |+> --> ?
#     qc.h(qr[n])  
    
    ## start as |-> --> ? 
#     qc.h(qr[n])
#     qc.z(qr[n])
    
    ## start as |i> --> ?
    qc.h(qr[n])
    qc.s(qr[n])
    
    ## start as |-i> --> ?
#     qc.h(qr[n]) 
#     qc.z(qr[n])
#     qc.s(qr[n])
    
    ### perform flipping and walking steps
    for i in range(num_steps):
        ## apply coin gate
#         qc.unitary(coin_gate, qr[n])
        qc.append(coin_gate, [qr[n]])
        
        ## apply walk gate
#         qc.unitary(walk_gate, [qr[i] for i in range(n+1)])
        qc.append(walk_gate, [qr[i] for i in range(n+1)])
    
    ## perform measurement
    qc.measure( [qr[i] for i in range(n)], [cr[i] for i in range(n)])
    
    ## do simulation
    backend_sim = qiskit.Aer.get_backend('qasm_simulator')
    from qiskit import execute
    from qiskit import visualization
    
    job = execute(qc, backend_sim, shots = num_shots)
    results = job.result()
    counts = results.get_counts()
    
    ## convert measurements to decimal
    counts_dec = {}
    for key in counts.keys():
        counts_dec[bin_to_dec(key)] = counts[key]
    
    # convert to list
    counts_dec = [key for key in counts_dec.keys() for i in range(counts_dec[key]) ]
    
    return counts_dec






