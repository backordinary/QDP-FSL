# https://github.com/heineaabo/QuantumAlgorithms/blob/9ef038b5b2b721efada8360e76382b055a1fd38b/quantum_algorithms/ansatz/ucc_nonopt.py
import numpy as np
import qiskit as qk

class UnitaryCoupledCluster:
    """
    Unitary Coupled Cluster ansatz, with optimized singles and doubles circuits.
    Jordan-Wigner transformed operators.
    Assumes the convention:
        ❘0⟩ - occupied orbital.
        ❘1⟩ - unoccupied orbital.

    Input:
            n (int): Number of particles.    
            l (int): Number of spin orbitals.
        trunc (str): CC truncation.
                     - 'S' for singles.
                     - 'D' for doubles.
                     - 'SD' for singles and doubles.
        depth (int): Depth of Trotter approximation. 
        occupied   : Convention for occupied orbital.
    """

    def __init__(self,n,l,trunc='SD',depth=1,occupied=1):
        self.n = n
        self.l = l
        self.trunc = trunc
        self.depth = depth
        self.occupied = occupied
        self.mp2 = False # MP2 intial parameters

    def __call__(self,theta,qc,qb,qa=None,i=None):
        if 'S' not in self.trunc and 'D' not in self.trunc:
            raise ValueError('Need to spesify correct UCC type')
        qc = self.prepare_Hartree_state(qc,qb)
        theta = self.sort_theta(theta)
        for rho in range(self.depth):
            if self.mp2:
                rho = 0
            if 'S' in self.trunc:
                qc = self.UCCS(qc,qb,theta[0],rho,qa=qa,d_j=i)
            if 'D' in self.trunc:
                qc = self.UCCD(qc,qb,theta[1],rho,qa=qa,d_j=i)
        return qc

    def new_parameters(self,h=[],v=[]):
        """
        Define new parameters. If one-body and two-body integrals 
        will return the MP2 cluster amplitudes. Else return random parameteres.
        """
        if len(h)!=0 and len(v)!=0:
            self.mp2 = True
        S = 0
        D = 0
        params = []
        if 'S' in self.trunc:
            for i in range(self.n):
                for a in range(self.n,self.l):
                    S += 1
                    if self.mp2:
                        params.append(0)

        if 'D' in self.trunc:
            for i in range(self.n):
                for j in range(i+1,self.n):
                    for a in range(self.n,self.l):
                        for b in range(a+1,self.l):
                            D += 1
                            if self.mp2:
                                t = (v[i,j,b,a]-v[i,j,a,b])/(h[i,i]+h[j,j]-h[a,a]-h[b,b])
                                params.append(t)
        self.num_S = S
        self.num_D = D
        if self.mp2:
            return np.asarray(params)
        else:
            return 2*np.pi*np.random.randn((self.num_S+self.num_D)*self.depth)

    def prepare_Hartree_state(self,qc,qb):
        """
        Prepares Hartree-Fock state. Assume initialized to ❘0...0⟩ before 
        Example for n=2 and l=4 -> ❘1100⟩
        """
        # Reference state
        if self.occupied: #❘1...0...⟩
            for k in range(self.n):
                qc.x(qb[k])
        elif not self.occupied: #❘0...1...⟩
            for k in range(self.n,self.l):
                qc.x(qb[k])
        return qc

    def sort_theta(self,theta):
        """
        Sort parameters to two channels for singles and doubles.
        TODO: 
            - Sort for depth > 1.
        """
        singles = self.num_S
        doubles = self.num_D
        thetaS = []
        thetaD = []
        if self.mp2:
            assert len(theta) == (singles+doubles)
            thetaS.append(theta[:singles])
            thetaD.append(theta[singles:])
        else:
            assert len(theta) == (singles+doubles)*self.depth
            for i in range(self.depth):
                thetaS.append(theta[(i*singles):(i+1)*singles])
                thetaD.append(theta[(self.depth*singles + (i)*doubles):\
                                    (self.depth*singles + (i+1)*doubles)])
        new_theta = [thetaS,thetaD]
        return new_theta

    def UCCS(self,qc,qb,theta,rho,qa=None,d_j=-1):
        """
        Optimized UCC singles implementation.
        """
        n_qubits = self.l
        n_fermi = self.n
        #theta = theta.reshape(n_fermi,n_qubits-n_fermi)

        t = 0
        # UCCS
        for i in range(n_fermi):
            for a in range(n_fermi,n_qubits):
                if d_j == t:
                    qc = self.dUCCS(qc,qb,qa,theta[rho][t],i,a)
                    continue
                # Pauli-z
                for k in range(i+1,a):
                    qc.z(qb[k])
                # Pauli strings
                qc.h(qb[i])
                qc.rx(np.pi/2,qb[i])
                qc.h(qb[a])

                qc.cx(qb[i],qb[a])

                #qc.rz(theta[i,a-n_fermi]/2,qb[a])
                qc.rz(theta[rho][t]/2,qb[a])

                qc.cx(qb[i],qb[a])

                qc.rx(-np.pi/2,qb[i])
                qc.h(qb[i])
                qc.h(qb[a])

                qc.h(qb[i])
                qc.h(qb[a])
                qc.rx(np.pi/2,qb[a])

                qc.cx(qb[i],qb[a])

                qc.rz(-theta[rho][t]/2,qb[a])

                qc.cx(qb[i],qb[a])

                qc.h(qb[i])
                qc.rx(-np.pi/2,qb[a])
                qc.h(qb[a])

                t += 1

        return qc

    def UCCD(self,qc,qb,theta,rho,qa=None,d_j=-1):
        """
        Optimized UCC doubles implementation.
        qa -> Ancilla qubit register
        d_j -> parameter index of parameter to diffrentiate
        """
        n_qubits = self.l
        n_fermi = self.n
        #theta = theta.reshape(n_fermi,n_fermi,n_qubits-n_fermi,n_qubits-n_fermi)

        t = 0
        # UCCD
        for i in range(n_fermi):
            for j in range(i+1,n_fermi):
                for a in range(n_fermi,n_qubits):
                    for b in range(a+1,n_qubits):
                        if d_j == t+self.num_S:
                            qc = self.dUCCD(qc,qb,qa,theta[rho][t],i,j,a,b)
                            continue

                        # Pauli-z
                        for k in range(i+1,j):
                            qc.z(qb[k])
                        for l in range(a+1,b):
                            qc.z(qb[l])

                        #TERM1
                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.h(qb[a])
                        qc.h(qb[b])
                        qc.rx(np.pi/2,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])

                        qc.rz(-theta[rho][t]/8,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])


                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.h(qb[a])
                        qc.rx(-np.pi/2,qb[b])
                        qc.h(qb[b])

                        #TERM2

                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.h(qb[a])
                        qc.rx(np.pi/2,qb[a])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])

                        qc.rz(-theta[rho][t]/8,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])


                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.rx(-np.pi/2,qb[a])
                        qc.h(qb[a])
                        qc.h(qb[b])


                        #TERM3

                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.rx(np.pi/2,qb[j])
                        qc.h(qb[a])
                        qc.h(qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])

                        qc.rz(theta[rho][t]/8,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])


                        qc.h(qb[i])
                        qc.rx(-np.pi/2,qb[j])
                        qc.h(qb[j])
                        qc.h(qb[a])
                        qc.h(qb[b])

                        #TERM4

                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.rx(np.pi/2,qb[j])
                        qc.h(qb[a])
                        qc.rx(np.pi/2,qb[a])
                        qc.h(qb[b])
                        qc.rx(np.pi/2,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])

                        qc.rz(-theta[rho][t]/8,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])


                        qc.h(qb[i])
                        qc.rx(-np.pi/2,qb[j])
                        qc.h(qb[j])
                        qc.rx(-np.pi/2,qb[a])
                        qc.h(qb[a])
                        qc.rx(-np.pi/2,qb[b])
                        qc.h(qb[b])

                        #TERM5

                        qc.h(qb[i])
                        qc.rx(np.pi/2,qb[i])
                        qc.h(qb[j])
                        qc.h(qb[a])
                        qc.h(qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])

                        qc.rz(theta[rho][t]/8,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])


                        qc.rx(-np.pi/2,qb[i])
                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.h(qb[a])
                        qc.h(qb[b])

                        #TERM6

                        qc.h(qb[i])
                        qc.rx(np.pi/2,qb[i])
                        qc.h(qb[j])
                        qc.h(qb[a])
                        qc.rx(np.pi/2,qb[a])
                        qc.h(qb[b])
                        qc.rx(np.pi/2,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])

                        qc.rz(-theta[rho][t]/8,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])


                        qc.rx(-np.pi/2,qb[i])
                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.rx(-np.pi/2,qb[a])
                        qc.h(qb[a])
                        qc.rx(-np.pi/2,qb[b])
                        qc.h(qb[b])

                        #TERM7

                        qc.h(qb[i])
                        qc.rx(np.pi/2,qb[i])
                        qc.h(qb[j])
                        qc.rx(np.pi/2,qb[j])
                        qc.h(qb[a])
                        qc.h(qb[b])
                        qc.rx(np.pi/2,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])

                        qc.rz(theta[rho][t]/8,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])


                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.rx(-np.pi/2,qb[i])
                        qc.h(qb[a])
                        qc.rx(-np.pi/2,qb[j])
                        qc.rx(-np.pi/2,qb[b])
                        qc.h(qb[b])

                        #TERM8

                        qc.h(qb[i])
                        qc.rx(np.pi/2,qb[i])
                        qc.h(qb[j])
                        qc.rx(np.pi/2,qb[j])
                        qc.h(qb[a])
                        qc.rx(np.pi/2,qb[a])
                        qc.h(qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])

                        qc.rz(theta[rho][t]/8,qb[b])

                        qc.cx(qb[i],qb[b])
                        qc.cx(qb[j],qb[b])
                        qc.cx(qb[a],qb[b])


                        qc.rx(-np.pi/2,qb[i])
                        qc.h(qb[i])
                        qc.h(qb[j])
                        qc.h(qb[a])
                        qc.rx(-np.pi/2,qb[j])
                        qc.h(qb[b])
                        qc.rx(-np.pi/2,qb[a])
                        
                        t += 1

        return qc


    def dUCCS(self,qc,qb,qa,theta_t,i,a):
        """
        Optimized gradient UCC singles implementation.
        qa -> ancilla qubit register
        theta_t - the t-th parameter from the derivative
        """
        n_qubits = self.l
        n_fermi = self.n

        # Pauli-z
        for k in range(i+1,a):
            qc.z(qb[k])
        # Pauli strings
        qc.h(qb[i])
        qc.rx(np.pi/2,qb[i])
        qc.h(qb[a])

        qc.cx(qb[i],qb[a])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(theta_t/2,qa[0],qb[a])

        qc.cx(qb[i],qb[a])

        qc.rx(-np.pi/2,qb[i])
        qc.rx(np.pi/2,qb[a])

        qc.cx(qb[i],qb[a])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(-theta_t/2,qa[0],qb[a])

        qc.cx(qb[i],qb[a])

        qc.h(qb[i])
        qc.rx(-np.pi/2,qb[a])
        qc.h(qb[a])

        return qc

    def dUCCD(self,qc,qb,qa,theta_t,i,j,a,b):
        """
        Optimized gradient UCC doubles implementation.
        qa -> ancilla qubit register
        theta_t - the t-th parameter from the derivative
        """
        n_qubits = self.l
        n_fermi = self.n

        # Pauli-z
        for k in range(i+1,j):
            qc.z(qb[k])
        for l in range(a+1,b):
            qc.z(qb[l])

        # Pauli strings
        qc.h(qb[i])
        qc.h(qb[j])
        qc.h(qb[a])
        qc.rx(np.pi/2,qb[a])
        qc.h(qb[b])

        qc.cx(qb[i],qb[b])
        qc.cx(qb[j],qb[b])
        qc.cx(qb[a],qb[b])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(theta_t/8,qa[0],qb[b]) #(xxyx)
        
        qc.cx(qb[a],qb[b])
        qc.cx(qb[j],qb[b])

        qc.rx(np.pi/2,qb[j])
        qc.rx(-np.pi/2,qb[a])

        qc.cx(qb[j],qb[b])
        qc.cx(qb[a],qb[b])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(-theta_t/8,qa[0],qb[b]) #(xyxx)
        
        qc.cx(qb[j],qb[b])
        qc.cx(qb[i],qb[b])

        qc.rx(np.pi/2,qb[i])
        qc.rx(-np.pi/2,qb[j])
        
        qc.cx(qb[i],qb[b])
        qc.cx(qb[j],qb[b])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(-theta_t/8,qa[0],qb[b]) #(yxxx)

        qc.cx(qb[a],qb[b])
        qc.cx(qb[j],qb[b])

        qc.rx(np.pi/2,qb[j])
        qc.rx(np.pi/2,qb[a])

        qc.cx(qb[j],qb[b])
        qc.cx(qb[a],qb[b])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(-theta_t/8,qa[0],qb[b]) #(yyyx)
        
        qc.cx(qb[a],qb[b])
        qc.cx(qb[j],qb[b])
        qc.cx(qb[i],qb[b])

        qc.rx(-np.pi/2,qb[i])
        qc.rx(np.pi/2,qb[b])

        qc.cx(qb[i],qb[b])
        qc.cx(qb[j],qb[b])
        qc.cx(qb[a],qb[b])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(theta_t/8,qa[0],qb[b]) #(xyyy)

        qc.cx(qb[j],qb[b])
        qc.cx(qb[i],qb[b])

        qc.rx(np.pi/2,qb[i])
        qc.rx(-np.pi/2,qb[j])
        
        qc.cx(qb[i],qb[b])
        qc.cx(qb[j],qb[b])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(theta_t/8,qa[0],qb[b]) #(yxyy)

        qc.cx(qb[a],qb[b])
        qc.cx(qb[j],qb[b])

        qc.rx(np.pi/2,qb[j])
        qc.rx(-np.pi/2,qb[a])

        qc.cx(qb[j],qb[b])
        qc.cx(qb[a],qb[b])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(-theta_t/8,qa[0],qb[b]) #(yyxy)

        qc.cx(qb[j],qb[b])
        qc.cx(qb[i],qb[b])

        qc.rx(-np.pi/2,qb[i])
        qc.rx(-np.pi/2,qb[j])
        
        qc.cx(qb[i],qb[b])
        qc.cx(qb[j],qb[b])

        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.s(qb[b])
        qc.cx(qa[0],qb[b])
        qc.crz(theta_t/8,qa[0],qb[b]) #(xxxy)

        qc.cx(qb[a],qb[b])
        qc.cx(qb[j],qb[b])
        qc.cx(qb[i],qb[b])

        qc.h(qb[i])
        qc.h(qb[j])
        qc.h(qb[a])
        qc.rx(-np.pi/2,qb[b])
        qc.h(qb[b])
        
        return qc

