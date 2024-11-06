# https://github.com/heineaabo/QuantumAlgorithms/blob/9ef038b5b2b721efada8360e76382b055a1fd38b/quantum_algorithms/ansatz/ryrz.py
import numpy as np
import qiskit as qk

from .AND_gates import *
class RYRZ:
    """
    """

    def __init__(self,n,l,occupied=1,pair=True,depth=1):
        self.n = n
        self.l = l
        self.occupied = occupied
        self.depth = depth

    def __call__(self,theta,qc,qb,qa=None,i=None):
        """
        Pairing ansatz. For 2 particles and 4 orbitals.
        """
        #if theta == None:
        #    theta = self.new_parameters()
        theta = self.sort_theta(theta)
        qc = self.prepare_Hartree_state(qc,qb)
        #if self.n == 2 and self.l == 3:
        #    qc = self.n2l3(theta,qc,qb)
        if self.n == 2 and self.l == 4:
            qc = self.n2l4(theta,qc,qb)
        elif self.n == 4 and self.l == 8:
            qc = self.n4l8(theta,qc,qb)
        else:
            print('No ansatz action')
        return qc

    def n2l4(self,theta,qc,qb):
        ### Rotation layer
        for j in range(self.depth):
            for i in range(self.n):
                qc.u3(theta[i,2*j],0,0,qb[i]) # Ry gate
                qc.u1(theta[i,2*j+1],qb[i])     # Rz gate
                if i == 0:
                    # For coupling map
                    #qc.swap(qb[1],qb[2])
                    #qc.cx(qb[0],qb[2])
                    #qc.swap(qb[1],qb[2])
                    qc.cx(qb[i],qb[(i+1)%2])
        ### Singles excitations
        for i in range(self.n,self.l,2):
            for j in range(self.depth):
                qc.u1(-0.5*theta[i,-2*j-1],qb[i])     # Rz gate
                qc.u3(-0.5*theta[i,-2*j-2],0,0,qb[i]) # Ry gate
        for i in range(self.n):
            qc.cx(qb[i],qb[self.n])
        for i in range(self.n,self.l,2):
            for j in range(self.depth):
                qc.u3(0.5*theta[i,2*j],0,0,qb[i]) # Ry gate
                qc.u1(0.5*theta[i,2*j+1],qb[i])     # Rz gate
        for i in range(self.n):
            qc.cx(qb[i],qb[self.n+1])
        ### Doubles excitation
        qc = AND_00(qc,qb,[0,1],2)
        qc.cx(qb[self.n],qb[self.n+1])# CNOT gate
        return qc

    def n2l3(self,theta,qc,qb):
        print(theta)
        for j in range(self.depth):
            qc.u3(theta[0,2*j],0,0,qb[0]) # Ry gate
            qc.u1(theta[0,2*j+1],qb[0])     # Rz gate
            qc.cx(qb[0],qb[1])
            qc.u3(theta[1,2*j],0,0,qb[1]) # Ry gate
            qc.u1(theta[1,2*j+1],qb[1])     # Rz gate

        qc.cx(qb[0],qb[2])
        qc.cx(qb[1],qb[2])

        ## Doubles
        qc = AND_00(qc,qb,[0,1],2)

        qc.cx(qb[2],qb[1])
        qc.cx(qb[2],qb[0])


        ### fix double
        #qc.cx(qb[2],qb[0])
        #qc = AND_10(qc,qb,[2,0],1)
        #qc.cx(qb[2],qb[0])
        #qc.h(qb[1])
        #qc.cx(qb[2],qb[0])
        #qc = AND_10(qc,qb,[2,0],1)
        #qc.cx(qb[2],qb[0])
        #qc.cx(qb[1],qb[2])
        #qc = AND_10(qc,qb,[1,2],0)
        #qc.cx(qb[1],qb[2])
        return qc

    def n4l8(self,theta,qc,qb):
        for j in range(self.depth):
            qc.u3(theta[0,2*j],0,0,qb[0]) # Ry gate
            qc.u1(theta[0,2*j+1],qb[0])     # Rz gate
            qc.cx(qb[0],qb[2])
            qc.u3(theta[1,2*j],0,0,qb[2]) # Ry gate
            qc.u1(theta[1,2*j+1],qb[2])     # Rz gate

        for j in range(self.depth):
            qc.u1(-0.5*theta[3,-2*j-1],qb[4])     # Rz gate
            qc.u3(-0.5*theta[3,-2*j-2],0,0,qb[4]) # Ry gate
        qc.cx(qb[0],qb[4])
        qc.cx(qb[2],qb[4])
        for j in range(self.depth):
            qc.u3(0.5*theta[3,2*j],0,0,qb[4]) # Ry gate
            qc.u1(0.5*theta[3,2*j+1],qb[4])     # Rz gate

        qc.cx(qb[0],qb[6])
        qc.cx(qb[2],qb[6])
        qc.cx(qb[4],qb[6])

        # Doubles
        qc.cx(qb[4],qb[6])
        qc = AND_00(qc,qb,[0,2],4)
        qc.cx(qb[4],qb[6])

        # Flip opposite spin qubits
        # Occ1
        qc.x(qb[0])
        qc.cx(qb[0],qb[1])
        qc.x(qb[0])
        # Occ2
        qc.x(qb[2])
        qc.cx(qb[2],qb[3])
        qc.x(qb[2])
        # Vir1
        qc.cx(qb[4],qb[5])
        # Vir2
        qc.cx(qb[6],qb[7])
        return qc

    def new_parameters(self):
        """
        New initial parameters.
        """
        N = 0
        if self.l % 2 == 0:
            N = (self.n+self.l)*self.depth
        else:
            N = 2*self.n*self.depth
        return np.zeros(N)
        #return 2*np.pi*np.random.randn((self.n+self.l)*self.depth)

    def prepare_Hartree_state(self,qc,qb):
        """
        Prepares Hartree-Fock state. Assume initialized to ❘0...0⟩ before 
        Example for n=2 and l=4 -> ❘1100⟩
        """
        # Reference state
        if self.occupied: #❘1...0...⟩
            for k in range(self.n):
                #qc.x(qb[k])
                qc.u3(np.pi,0,np.pi,qb[k])
        else: #❘0...1...⟩
            for k in range(self.n,self.l):
                #qc.x(qb[k])
                qc.u3(np.pi,0,np.pi,qb[k])
        return qc

    def sort_theta(self,theta):
        assert len(theta)%2 == 0
        return theta.reshape((int(len(theta)/(2*self.depth)),2*self.depth)) 


