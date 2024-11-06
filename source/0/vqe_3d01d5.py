# https://github.com/heineaabo/QuantumAlgorithms/blob/9ef038b5b2b721efada8360e76382b055a1fd38b/quantum_algorithms/vqe.py
import numpy as np
import qiskit as qk
from qiskit.extensions.standard import *
from .tools import print_state,get_state_count
from .ansatz import UnitaryCoupledCluster,RYRZ,RY,RYpairing,UCC5
from .algorithm import QuantumAlgorithm
from .attributes import QuantumComputer
from .optimizers import RyGradient,QuantumGradientDescent

class VQE(QuantumAlgorithm):
    def __init__(self,
                 hamiltonian,
                 optimizer,
                 ansatz = 'UCCSD',
                 ansatz_depth = 1,
                 options = {}):
        """
        Input:
            hamiltonian  (class) - Hamiltonian with system information
                                   and circuit_list.
            optimizer    (class) - Optimization method.
            ansatz       (class) - Ansatz.
            ansatz_depth (class) - Ansatz depth of Trotter expansion.

        """
        if isinstance(hamiltonian,dict):
            self.n_qubits = hamiltonian['l']
            self.n_fermi = hamiltonian['n']
            self.circuit_list = hamiltonian['circuit']
            self.conv = 1 if hamiltonian.get('conv') == None else hamiltonian['conv'] # Occupation convention
        else:
            self.n_qubits = hamiltonian.l
            self.n_fermi = hamiltonian.n
            self.circuit_list = hamiltonian.circuit_list('vqe')
            self.conv = hamiltonian.conv # Occupation convention

        super().__init__(self.n_qubits,options)
        #assert self.ibmq
        # Unitary Coupled Cluster ansatz
        self.ansatz_string = ansatz
        if isinstance(ansatz,str):
            if ansatz[:3].upper() == 'UCC':
                #self.ansatz = UCC5(self.n_fermi,self.n_qubits,
                #                   ansatz[3:].upper(),depth=ansatz_depth)
                if self.ibmq:
                    self.ansatz = UCC5(self.n_fermi,self.n_qubits,
                                       ansatz[3:].upper(),depth=ansatz_depth)
                else:
                    self.ansatz = UnitaryCoupledCluster(self.n_fermi,
                                                        self.n_qubits,
                                                        ansatz[3:].upper(),
                                                        depth=ansatz_depth)
                if isinstance(hamiltonian,dict):
                    self.theta = hamiltonian.theta
                else:
                    self.theta = self.ansatz.new_parameters(hamiltonian.h,
                                                            hamiltonian.v)
            elif ansatz.upper() == 'RYPAIRING':
                self.ansatz = RYpairing(self.n_fermi,self.n_qubits)
                self.theta = self.ansatz.new_parameters()
            elif ansatz.upper() == 'RY':
                self.ansatz = RY(self.n_fermi,self.n_qubits)
                self.theta = self.ansatz.new_parameters()
            elif ansatz.upper() == 'RYRZ':
                self.ansatz = RYRZ(self.n_fermi,self.n_qubits,depth=ansatz_depth)
                self.theta = self.ansatz.new_parameters()
            else:
                def no_ansatz(t,qc,qb):
                    return qc
                self.ansatz = no_ansatz

        # Custom ansatz
        else:
            self.ansatz = ansatz
            self.theta = 0

        # For optimization
        self.optimizer = optimizer
        if isinstance(optimizer,RyGradient):
            self.optimizer.set_vqe(self)
        #elif isinstance(optimizer,QuantumGradientDescent):
        else:
            self.optimizer.set_loss_function(self.expval)
            #self.optimizer.set_gradient_function(self.gradient)
        #elif optimizer != None:
        #    self.optimizer.set_loss_function(self.expval)

        # For counting states
        self.legal = 0
        self.illegal = 0
        self.evals = 0

        # For plotting progression from optimization
        self.energies = []

    def expval(self,theta=None,callback=True,p_mes=False):
        if theta is None:
            theta = self.theta
        E = 0
        # Prepare qiskit circuit and registers.
        qb = qk.QuantumRegister(self.n_qubits)
        cb = qk.ClassicalRegister(self.n_qubits)
        qc = qk.QuantumCircuit(qb,cb)
        # Prepare ansatz to be reused.
        qc_ansatz = self.ansatz(theta,qc,qb)
        for i,pauli_string in enumerate(self.circuit_list):
            # Skip constant term
            if len(pauli_string) == 0:
                E += pauli_string.factor
                continue
            # New circuit
            qc = qk.QuantumCircuit(qb,cb)
            # Apply measurement transformation
            qc = pauli_string.prepare(qc,qb) 
            # Combine circuit with ansatz circuit
            #print(qc)
            qc = qc_ansatz + qc
            # Measure circuit and add expectation value
            measurement = self.measure(qc,qb,cb)
            if self.meas_fitter != None:
                measurement = self.meas_fitter.filter.apply(measurement)
            if p_mes:
                print(qc)
                print(measurement)
            E += pauli_string.expectation(measurement,self.shots)
        if callback:
            if self.prnt:
                #print('⟨E⟩ = {}, θ = {}'.format(E,theta))
                print('⟨E⟩ = {}'.format(E))
            self.energies.append(E)
            self.evals += 1
        if self.ibmq:
            np.save('.ibmq_energies.npy',np.asarray(self.energies))
            thetaz = np.load('.ibmq_theta.npy')
            thetaz = np.append(thetaz,theta)
            np.save('.ibmq_theta.npy',thetaz)
        return E

    def optimize(self,theta=None):
        if theta == None:
            theta = self.theta
        params = self.optimizer(theta)
        if not isinstance(params,(list,tuple,np.ndarray)):
            np.asarray(params)
        return params

    def get_mean(self,theta,N=None,M=10):
        """
        N: Number of shots.
        M: Number of expectation value calculations.
        """
        old_shots = self.shots
        if N == None:
            N = self.shots
        self.shots = N
        measurements = np.zeros(M)
        for i in range(M):
            measurements[i] = self.expval(theta)
        self.shots = old_shots
        return np.mean(measurements),np.var(measurements)

    def get_state_coeff(self,theta):
        # Get FCI state configuration
        from itertools import combinations
        states = []
        states_int = []
        for state in combinations(range(self.n_qubits),self.n_fermi):
            states_int.append([orb for orb in state])
        states_str = [['0' for j in range(self.n_qubits)] for i in states_int]
        
        for state,inds in zip(states_str,states_int):
            for ind in inds:
                state[ind] = '1'
            states.append(''.join(state)[::-1])
        qb = qk.QuantumRegister(self.n_qubits)
        cb = qk.ClassicalRegister(self.n_qubits)
        qc = qk.QuantumCircuit(qb,cb)
        qc = self.ansatz(theta,qc,qb)
        measurement = self.measure(qc,qb,cb)
        coef = np.zeros(len(states))
        for i,state in enumerate(states):
            if measurement.get(state) == None:
                coef[i] = 0
            else:
                coef[i] = measurement[state]
        return coef/self.shots


    def print_circ_info(self):
        theta = self.theta
        # Prepare qiskit circuit and registers.
        qb = qk.QuantumRegister(self.n_qubits)
        cb = qk.ClassicalRegister(self.n_qubits)
        qc = qk.QuantumCircuit(qb,cb)
        qc = self.ansatz(theta,qc,qb)
        measurement = self.measure(qc,qb,cb)
        print('For ansatz {} with {} parameters:'.format(self.ansatz_string,len(theta)))
        print('\t- Circuit depth: {} \n\t- CNOT gates: {} \n\t- Single qubit gates: {}'.format(qc.depth(),qc.count_ops()['cx'],qc.size()-qc.count_ops()['cx']))

    def optimize_gradient(self,
                          theta,
                          max_iters=200,
                          max_evals=200,
                          step_length=1e-01, # Step length
                          tol=1e-05):
        self.energies = []
        qc,qb,qa,cb,ca = None,None,None,None,None
        new_theta = theta
        qb = qk.QuantumRegister(self.n_qubits)
        cb = qk.ClassicalRegister(self.n_qubits)
        for i in range(max_iters):
            new_theta = theta
            E_inter = self.expval(new_theta)
            print('<E> =',E_inter)
            grad = np.zeros_like(theta)
            for d_j in range(len(theta)):
                im = 0
                for pauli_string in self.circuit_list:
                    qubit_list = []
                    # New circuit
                    qc = qk.QuantumCircuit(qb,cb)
                    # Prepare ancilla qubit with Hadamard
                    qa = qk.QuantumRegister(1)
                    ca = qk.ClassicalRegister(1)
                    qc.add_register(qa,ca)
                    qc.h(qa[0])
                    # Apply measurement transformation
                    qc = pauli_string.prepare(qc,qb,qa=qa) 
                    # Finish Hadamard test
                    qc.h(qa[0])
                    qc.rx(np.pi/2,qa[0])
                    # Measure circuit and add expectation value
                    measurement = self.measure(qc,qa,ca)
                    im += self.ancilla_expectation(measurement)
                #print('Updating theta {} -> {} - {} = {}'.format(d_j,new_theta[d_j],im,new_theta[d_j]-im*step))
                new_theta[d_j] += im*step_length
            theta = new_theta
            print('Iteration {} finished!'.format(i),new_theta)
        return theta

