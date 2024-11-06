# https://github.com/BILLYZZ/Born_Machine_CSE/blob/4bb6b3b4e313dd08f4795f19f8d7adef82078a2b/MBL_Born/Qiskit_Circuits.py
import qiskit
from qiskit import transpile, assemble
import numpy as np
import torch
import Utils
import Qiskit_Utils


# Using efficient symmetrized trotterization circuit. The quench layers will accumulate since in reality we don't have qrams
class MBL_Circuit:
    # The trotterized MBL circuit the number of layers is changing!! tm is the set time for each quench
    def __init__(self, delta_t, t_m, Jxy, Jzz, n_qubits, n_h_qubits, connections, backend, shots, if_measure):
        self.delta_t = delta_t  # Make this 1.0 by default
        self.t_m = t_m
        self.r = int(t_m/delta_t)
        self.Jxy  = Jxy
        self.Jzz = Jzz
        #self.h_m_q_mat = h_m_q_mat # Growing from 1 by Q to M by Q, for m-th quench, we have one parameter for each qubit q
        self.n_qubits = n_qubits
        self.n_h_qubits = n_h_qubits
        self.hdim = 2**(n_qubits-n_h_qubits)
        self.connections = connections
        self.backend = backend
        self.shots = shots
        self.if_measure = if_measure

        self.all_qubit_list = [i for i in range(n_qubits)]
        self.measure_qubit_list = [i for i in range(n_qubits-n_h_qubits)]
        # Set up the connection outlet for binding dictionary
        self.thetas = [] # This stores all the qiskit variational parameter objects the circuit currently has at quench m
        self.current_m = None # this is updated throughout the quenches

        # The B and C modules
        self.B_gate, self.C_gate = Qiskit_Utils.create_B_C(n_qubits, delta_t, Jx=Jxy, Jy=Jxy, Jz=Jzz, connections=connections)

        self._circuit = qiskit.QuantumCircuit(n_qubits, n_qubits-n_h_qubits) # need n_qubits-n_h_qubits number of classic bits to
        self.init_wall_state()
        
        if if_measure:
            self._circuit.measure(self.measure_qubit_list, self.measure_qubit_list)

        self.t_qc = transpile(self._circuit, # Should transpile this whenever the circuit structure is altered
                         self.backend)


    def init_wall_state(self):
        Qiskit_Utils.Wall_state(self._circuit) #Start from a Neel state 01010101010....or wall state 00001111

    def append_layers_m(self, m): # for the  m-th quench, append a block of trotter_r number of layers
        self.current_m = m
        theta_list_at_m = [qiskit.circuit.Parameter('theta-'+str(m)+'-'+str(q)) for q in range(self.n_qubits)] # variational params
        self.thetas.append(theta_list_at_m)
        

        # First take away the previous measurements
        if self.if_measure:
            self._circuit.remove_final_measurements()

        # Please pre-set self.B_gate and self.C_gates if we don't need to adjust the Jx and Jz coeffs during quenches 
        if self.B_gate is None or self.C_gate is None:
            B_gate, C_gate = Qiskit_Utils.create_B_C(self.n_qubits, self.delta_t, 
                                Jx=self.Jxy, Jy=self.Jxy, Jz=self.Jzz, connections=self.connections) # This will create circuit blocks using the current self.J values
        else:
            B_Jxy, C_gate = self.B_gate, self.C_gate
        
        for i in range(self.r): # To make enough delta_t layers to extend to the required t_m for quench m.
            # The A portion:
            for q in range(self.n_qubits): # for each qubit
                #print('debug i q', (i, q))
                self._circuit.rz(theta_list_at_m[q]*self.delta_t, q)
            # The B portion:
            self._circuit.append(B_gate, self.all_qubit_list) # Only involves Jxy
            # The C portion:
            self._circuit.append(C_gate, self.all_qubit_list)
            # The B portion again:
            self._circuit.append(B_gate, self.all_qubit_list)
            # The A portion again:
            for q in range(self.n_qubits): # for each qubit
                self._circuit.rz(theta_list_at_m[q]*self.delta_t, q)

        if self.if_measure:
            # This is a Qiskit glitch: removing the final measuremnts also removes the classical registers
            #self._circuit.add_register(qiskit.ClassicalRegister(self.n_qubits-self.n_h_qubits))
            self._circuit.measure(self.measure_qubit_list, self.measure_qubit_list)
        self.t_qc = transpile(self._circuit,
                         self.backend)
    # Note: the circuit only has knowledge of the current number of quenches, so theta_vals will have a shape (m, q), m=0 to M
    def bind_dict(self, theta_vals): # Assume theta_values are converted from tensor to numpy array. shape is (m, q)
        return_dict = {}
        for m in range(self.current_m+1): # self.thetas has a len of m
            for q in range(self.n_qubits):
                # Use circuit.Parameter objects as the key, and float numbers as the value
                return_dict[self.thetas[m][q]] = theta_vals[m][q]
        #print(return_dict)
        return return_dict     

    def pdf_sample(self, theta_list):
        if self.if_measure: # if we take measurements, use the measurement results and make histogram
            theta_vals = theta_list.reshape((self.current_m+1, self.n_qubits)).detach().numpy()
            qobj = assemble(self.t_qc,
                            shots=self.shots,
                            parameter_binds = [self.bind_dict(theta_vals)])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            #print(result)
            keys = list(result.keys())
            ints = []
            for k in keys:
                ints.append(int(k, 2))
            ints = torch.tensor(ints)
            inds = torch.argsort(ints) # get the indices of sorting low to high
            counts = torch.FloatTensor(list(result.values()))[inds]
            #print('reorder', ints[inds])
            probabilities = counts / self.shots
            #print('result,', result)
        else: # if we don't measure but directly use the state vector, sample from the sate vector 
            probabilities, state_vec = self.pdf_actual(theta_list).detach().numpy().astype('float64')
            # Then add some sample noise
            probabilities = Utils.prob_from_sample( Utils.sample_from_prob(np.arange(self.hdim), probabilities, self.shots),
                    self.hdim, False )

        return torch.FloatTensor(probabilities), state_vec

    # only works with ''statevector'' simulator
    def pdf_actual(self, theta_list):
        if self.if_measure:
            print('warning: calling this on measured circuit is meaningless')
        theta_vals = theta_list.reshape((self.current_m+1, self.n_qubits)).detach().numpy()
        qobj = assemble(self.t_qc,
                        shots=self.shots,
                        parameter_binds=[self.bind_dict(theta_vals)])
        job = self.backend.run(qobj)
        #print('job result', job.result())
        amps = job.result().get_statevector() # This returns a list of complex numbers
        full_statevec = qiskit.quantum_info.Statevector(amps) # Maybe save and return this state for faster simulations?
        #print(full_statevec)
        if self.n_h_qubits == 0: # Don't use partial trace if we dont' have hidden qubits
            probabilities = amps.real**2 + amps.imag**2
        else:
            partial_density_mat = qiskit.quantum_info.partial_trace(full_statevec, range(self.n_qubits-self.n_h_qubits, self.n_qubits))
            probabilities = np.diagonal(partial_density_mat)

        return torch.FloatTensor(probabilities), full_statevec

    # To calcualte the marginal probabitlites of a particular qubit 
    def marginal(self, qubit, theta_list):
        if self.if_measure:
            print("Can't call this function with measurements!")
        else:
            theta_vals = theta_list.reshape((self.current_m+1, self.n_qubits)).detach().numpy()
            qobj = assemble(self.t_qc,
                            shots=self.shots,
                            parameter_binds=[self.bind_dict(theta_vals)])
            job = self.backend.run(qobj)
            #print('job result', job.result())
            amps = job.result().get_statevector() # This returns a list of complex numbers
            full_statevec = qiskit.quantum_info.Statevector(amps)
            traceout = [i for i in range(self.n_qubits) if i != qubit]
            partial_density_mat = qiskit.quantum_info.partial_trace(full_statevec, traceout)
            probabilities = np.diagonal(partial_density_mat)
        return probabilities


# A more efficient way of running simulations (by loading quantum states)
# This will have a one quench layer and an arbitrary state loader in front!
# Has to use the statevector simulator, i.e., we dont' actually use the qiskit measure operations
class MBL_Circuit_quench:
    def __init__(self, delta_t, t_m, Jxy, Jzz, n_qubits, n_h_qubits, connections, backend):
        self.delta_t = delta_t  # Make this 1.0 by default
        self.t_m = t_m
        self.r = int(t_m/delta_t)
        self.Jxy  = Jxy
        self.Jzz = Jzz
        #self.h_m_q_mat = h_m_q_mat # Growing from 1 by Q to M by Q, for m-th quench, we have one parameter for each qubit q
        self.n_qubits = n_qubits
        self.n_h_qubits = n_h_qubits
        self.hdim = 2**(n_qubits-n_h_qubits)
        self.connections = connections
        self.backend = backend 

        # The B and C modules
        self.B_gate, self.C_gate = None, None

        self._circuit = qiskit.QuantumCircuit(n_qubits, n_qubits-n_h_qubits) # need n_qubits-n_h_qubits number of classic bits to
        # store the measurement
        self.all_qubit_list = [i for i in range(n_qubits)]
        self.measure_qubit_list = [i for i in range(n_qubits-n_h_qubits)]

        #-------------------Parameters and circuit architecture--------------------------
        self.h_vec = [qiskit.circuit.Parameter('h_'+str(i)) for i in range(n_qubits)] # variationa parameters for Jz of each qubit
        #-------------------Architecture Modules:------------------------------------------------
        self.B_gate, self.C_gate = Qiskit_Utils.create_B_C(n_qubits, delta_t, 
                                Jx=Jxy, Jy=Jxy, Jz=Jzz, connections=connections) # Since Jxy and Jzz will be fixed, save these now
        
        #------------------Start building the circuit:-----------------------------------
        self._circuit.initialize(qiskit.quantum_info.Statevector.from_label('0'*n_qubits), self.all_qubit_list)
        
        for i in range(self.r): # To make enough delta_t layers to extend to the required t_m for quench m.
            # The A portion:
            for q in range(self.n_qubits): # for each qubit
                #print('debug i q', (i, q))
                self._circuit.rz(self.h_vec[q]*self.delta_t, q)
            
            # The B portion:
            self._circuit.append(self.B_gate, self.all_qubit_list) # Only involves Jxy
            # The C portion:
            self._circuit.append(self.C_gate, self.all_qubit_list)
            # The B portion again:
            self._circuit.append(self.B_gate, self.all_qubit_list)
            
            # The A portion again:
            for q in range(self.n_qubits): # for each qubit
                self._circuit.rz(self.h_vec[q]*self.delta_t, q)

        self.t_qc = transpile(self._circuit,
                         self.backend) # Only call once, since the architecture doesn't change!
    
    # Quench from an arbitrary state: the input full_statevec is a qisit.quantum_info.StateVector object
    def load_arbitrary_state(self, full_statevec):
        # We assume that .data[0] is the `initialize' module
        self._circuit.data[0][0].params=full_statevec.data # set the list of complex values (amps)
        self.t_qc = transpile(self._circuit,
                         self.backend) # Only call once, since the architecture doesn't change!
        return

    def bind_dict(self, theta_vals): # Assume theta_values are converted from tensor to numpy array. shape is (m, q)
        return_dict = {}
        for q in range(self.n_qubits):
            # Use circuit.Parameter objects as the key, and float numbers as the value
            return_dict[self.h_vec[q]] = theta_vals[q]
        #print(return_dict)
        return return_dict     

    # only works with ''statevector'' simulator
    def pdf_actual(self, theta_list):
        theta_vals = theta_list.detach().numpy()
        qobj = assemble(self.t_qc, parameter_binds=[self.bind_dict(theta_vals)])
        job = self.backend.run(qobj)
        #print('job result', job.result())
        amps = job.result().get_statevector() # This returns a list of complex numbers
        full_statevec = qiskit.quantum_info.Statevector(amps) # Maybe save and return this state for faster simulations?
        if self.n_h_qubits == 0: # Don't use partial trace if we dont' have hidden qubits
            probabilities = amps.real**2 + amps.imag**2
        else:
            partial_density_mat = qiskit.quantum_info.partial_trace(full_statevec, range(self.n_qubits-self.n_h_qubits, self.n_qubits))
            probabilities = np.diagonal(partial_density_mat)

        return torch.FloatTensor(probabilities), full_statevec

    def pdf_sample(self, theta_list):
        probabilities, state_vec = self.pdf_actual(theta_list).detach().numpy().astype('float64')
        # Then add some sample noise
        probabilities = Utils.prob_from_sample( Utils.sample_from_prob(np.arange(self.hdim), probabilities, self.shots),
                self.hdim, False )

        return torch.FloatTensor(probabilities), state_vec





# As a rule of thumb, the circuit class objectis shall not store any value! Make everything variational ---BZ
# class MBL_Circuit_trotter(object):
#     # The trotterized MBL circuit the number of layers is changing!!
#     def __init__(self, delta_t, trotter_r, Jxy, Jzz, n_qubits, n_h_qubits, connections, backend, shots, if_measure):
#         self.dt = delta_t  # Make this 1.0 by default
#         self.r = trotter_r  # Is 5 larage enough for this?
#         self.Jxy  = Jxy
#         self.Jzz = Jzz
#         #self.h_m_q_mat = h_m_q_mat # Growing from 1 by Q to M by Q, for m-th quench, we have one parameter for each qubit q
#         self.n_qubits = n_qubits
#         self.n_h_qubits = n_h_qubits
#         self.hdim = 2**(n_qubits-n_h_qubits)
#         self.connections = connections
#         self.backend = backend
#         self.shots = shots
#         self.if_measure = if_measure

#         self._circuit = qiskit.QuantumCircuit(n_qubits, n_qubits-n_h_qubits) # need n_qubits-n_h_qubits number of classic bits to
#         # store the measurement
#         self.measure_qubit_list = [i for i in range(n_qubits-n_h_qubits)]
#         # Set up the connection outlet for binding dictionary
#         self.thetas = [] # This stores all the qiskit variational parameter objects the circuit currently has at quench m
        
#         self.current_m = None # this is updated throughout the quenches
#         self._circuit.h(range(n_qubits))
#         self.append_layers_m(m=0) # append the first quenching block, this updates self.current_m
#         if if_measure:
#             self._circuit.measure(self.measure_qubit_list, self.measure_qubit_list)

#         self.t_qc = transpile(self._circuit, # Shoul transpile this whenever the circuit structure is altered
#                          self.backend)

#     def append_layers_m(self, m): # for the  m-th quench, append a block of trotter_r number of layers
#         self.current_m = m
#         theta_list_at_m = [qiskit.circuit.Parameter('theta-'+str(m)+'-'+str(q)) for q in range(self.n_qubits)]
#         self.thetas.append(theta_list_at_m)
#         # First take away the previous measurements
#         if self.if_measure:
#             self._circuit.remove_final_measurements()
#         for i in range(self.r):
#             for c in self.connections:
#                 self._circuit.rxx(2*self.Jxy*self.dt/self.r, c[0], c[1])
#                 self._circuit.ryy(2*self.Jxy*self.dt/self.r, c[0], c[1])
#                 self._circuit.rzz(2*self.Jzz*self.dt/self.r, c[0], c[1])
#             for q in range(self.n_qubits): # for each  qubit
#                 self._circuit.rz(2*theta_list_at_m[q]*self.dt/self.r, q)

#         if self.if_measure:
#             # This is a Qiskit glitch: removing the final measuremnts also removes the classical registers
#             #self._circuit.add_register(qiskit.ClassicalRegister(self.n_qubits-self.n_h_qubits))
#             self._circuit.measure(self.measure_qubit_list, self.measure_qubit_list)
#         self.t_qc = transpile(self._circuit,
#                          self.backend)
#     # Note: the circuit only has knowledge of the current number of quenches, so theta_vals will have a shape (m, q), m=0 to M
#     def bind_dict(self, theta_vals): # Assume theta_values are converted from tensor to numpy array. shape is (m, q)
#         return_dict = {}
#         for m in range(len(self.thetas)): # self.thetas has a len of m
#             for q in range(self.n_qubits):
#                 # Use circuit.Parameter objects as the key, and float numbers as the value
#                 return_dict[self.thetas[m][q]] = theta_vals[m][q]
#         #print(return_dict)
#         return return_dict     

#     def pdf_sample(self, theta_list):
#         if self.if_measure: # if we take measurements, use the measurement results and make histogram
#             theta_vals = theta_list.reshape((self.current_m+1, self.n_qubits)).detach().numpy()
#             qobj = assemble(self.t_qc,
#                             shots=self.shots,
#                             parameter_binds = [self.bind_dict(theta_vals)])
#             job = self.backend.run(qobj)
#             result = job.result().get_counts()
#             #print(result)
#             keys = list(result.keys())
#             ints = []
#             for k in keys:
#                 ints.append(int(k, 2))
#             ints = torch.tensor(ints)
#             inds = torch.argsort(ints) # get the indices of sorting low to high
#             counts = torch.FloatTensor(list(result.values()))[inds]
#             #print('reorder', ints[inds])
#             probabilities = counts / self.shots
#             #print('result,', result)
#         else: # if we don't measure but directly use the state vector, sample from the sate vector 
#             probabilities = self.pdf_actual(theta_list).detach().numpy().astype('float64')
#             # Then add some sample noise
#             probabilities = Utils.prob_from_sample( Utils.sample_from_prob(np.arange(self.hdim), probabilities, self.shots),
#                     self.hdim, False )

#         return torch.FloatTensor(probabilities)

#     # only works with ''statevector'' simulator
#     def pdf_actual(self, theta_list):
#         if self.if_measure:
#             print('warning: calling this on measured circuit is meaningless')
#         theta_vals = theta_list.reshape((self.current_m+1, self.n_qubits)).detach().numpy()
#         qobj = assemble(self.t_qc,
#                         shots=self.shots,
#                         parameter_binds=[self.bind_dict(theta_vals)])
#         job = self.backend.run(qobj)
#         #print('job result', job.result())
#         amps = job.result().get_statevector() # This returns a list of complex numbers
#         full_statevec = qiskit.quantum_info.Statevector(amps)
#         #print(full_statevec)
#         if self.n_h_qubits == 0: # Don't use partial trace if we dont' have hidden qubits
#             probabilities = amps.real**2 + amps.imag**2
#         else:
#             partial_density_mat = qiskit.quantum_info.partial_trace(full_statevec, range(self.n_qubits-self.n_h_qubits, self.n_qubits))
#             probabilities = np.diagonal(partial_density_mat)
#         #print('amps', amps)
#         #probabilities = amps.real**2 + amps.imag**2
#         #print('probabilities', probabilities)
#         return torch.FloatTensor(probabilities)

class MLP_Circuit(object):# multilayer parameterized circuit
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    def __init__(self, n_qubits, n_layers, connections, backend, shots, if_measure):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.hdim = 2**n_qubits
        self.n_layers = n_layers
        self.connections = connections
        self.if_measure = if_measure
        self.num_param = n_layers*n_qubits*3
        # theta has the folllowing addres: layer, qubit, one of the three rotations
        self.thetas = []
        for l in range(n_layers):
            layer_thetas = []
            for q in range(n_qubits):
                layer_qubit_thetas = [] # to wrap and pass into the layer() function
                for r in range(3):
                    theta = qiskit.circuit.Parameter('theta-'+str(l)+'-'+str(q)+'-'+str(r))
                    layer_qubit_thetas.append(theta)
                layer_thetas.append(layer_qubit_thetas)
            
            self.build_layer(layer_thetas, self.connections)
            self.thetas.append(layer_thetas)
    
        if if_measure:
            self._circuit.measure_all()
        # ---------------------------
        self.backend = backend
        self.shots = shots
        self.t_qc = transpile(self._circuit,
                         self.backend)
        
    # Create load stuff into each layer
    def build_layer(self, layer_thetas, connections): # This is the generic naive multipalyer parameterized circuit
        for i in range(self.n_qubits):
            self._circuit.rz(layer_thetas[i][0], i)
            self._circuit.rx(layer_thetas[i][1], i)
            self._circuit.rz(layer_thetas[i][2], i)
        # Then the entangle
        for c in connections: # each element in connections is a tuple pair
            self._circuit.cnot(c[0], c[1])
        self._circuit.barrier()
    
    # Create the binding dictionary for parameter assignments
    # Input theta_vals is a nested array of float numbers, the hierarchy is the same as self.thetas
    def bind_dict(self, theta_vals):
        return_dict = {}
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                for r in range(3):
                    # Use circuit.Parameter objects as the key, and float numbers as the value
                    return_dict[self.thetas[l][q][r]] = theta_vals[l][q][r]
        return return_dict            
        
    def pdf_sample(self, theta_list):
        if self.if_measure: # if we take measurements, use the measurement results and make histogram
            theta_vals = theta_list.reshape((self.n_layers, self.n_qubits, 3)).detach().numpy()
            qobj = assemble(self.t_qc,
                            shots=self.shots,
                            parameter_binds = [self.bind_dict(theta_vals)])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            keys = list(result.keys())
            ints = []
            for k in keys:
                ints.append(int(k, 2))
            ints = torch.tensor(ints)
            inds = torch.argsort(ints) # get the indices of sorting low to high
            counts = torch.FloatTensor(list(result.values()))[inds]
            #print('reorder', ints[inds])
            probabilities = counts / self.shots
            #print('result,', result)
        else: # if we don't measure but directly use the state vector, sample from the sate vector 
            probabilities = self.pdf_actual(theta_list).detach().numpy().astype('float64')
            # Then add some sample noise
            probabilities = Utils.prob_from_sample( Utils.sample_from_prob(np.arange(self.hdim), probabilities, self.shots),
                    self.hdim, False )

        return torch.FloatTensor(probabilities)
    # only works with ''statevector'' simulator
    def pdf_actual(self, theta_list):
        theta_vals = theta_list.reshape((self.n_layers, self.n_qubits, 3)).detach().numpy()
        qobj = assemble(self.t_qc,
                        shots=self.shots,
                        parameter_binds = [self.bind_dict(theta_vals)])
        job = self.backend.run(qobj)
        #print('job result', job.result())
        amps = job.result().get_statevector()
        #print('amps', amps)
        probabilities = amps.real**2 + amps.imag**2
        
        return torch.FloatTensor(probabilities)

    # This generate() function is obsolete
    def generate(self, theta_vals, batch_size): # To do: reshape theta_list into theta_vals
        qobj = assemble(self.t_qc,
                        shots=batch_size,
                        parameter_binds = [self.bind_dict(theta_vals)])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        keys = list(result.keys()) # A list of strings
        ints = []
        for k in keys: # string to int
            ints.append(int(k, 2))
        inds = torch.argsort(ints) # get the indices of sorting low to high
        counts = torch.FloatTensor(list(result.values()))[inds]
        keys = np.array(keys)[inds] # array of strings
        patterns = torch.FloatTensor([self.split(k) for k in keys]) # m by n 
        print('result', result)
        return torch.FloatTensor(torch.repeat_interleave(patterns, counts, axis=0)).detach()

    def split(self, word):
        return np.array([float(char) for char in word])
    # def mmd_loss(self, theta_vals):
    #     if self.if_measure: # In this case, use the sampled wave function
    #         prob = self.pdf_sample(theta_vals)
    #     else:
    #         prob = self.pdf_actual(theta_vals)
    #     return self.Kernel.mmd_loss(prob, self.p)

    # def mcr_loss(self, px, py, num_sample):

    
	        

    