# https://github.com/Priyanshusinhaa/Super-Circuit/blob/934ee4891d95a067aaf71169d83acabef2068eaa/Circuit7/circuit7.py
from qiskit import *

theta = [i for i in range(788)]

class Circuit7:
    def __init__(self, qubits, layer, thetaList) -> None:
        self.qubits = qubits
        self.layer = layer -1
        self.theta = thetaList
        
        nbgates_0_last = self.__countOdd(self.layer + 1)*2
        nbtheta = (nbgates_0_last*2) + (self.qubits-2)*((self.layer+1)*2)
        
        if (len(self.theta) > nbtheta):
            print('Given theta list is Greater than Required gates')
        elif (len(self.theta) == nbtheta):
            pass
        else:
            print("Number of Elements in theta list provided isn't equal to Gates required in Ansatz. Please check thetaList")
            raise
        
        if (self.qubits < 4) or (self.qubits%2 != 0):
            print('Qubit must be greater than 4 and qubit must be even')
            raise
        
        if ((self.layer + 1) <= 2) or ((self.layer+1)%2 != 1):
            print('Layer should always be greater than 2 and layer must be odd number')
        
        if type(self.qubits) != type(int):
            self.qubits = int(self.qubits)
        else:
            print("Please Check You're Entering Right DataType in place of 'Qubits' parameter")
            raise

        self.qc = QuantumCircuit(self.qubits)
        
        nbgates_0_last = self.__countOdd(self.layer + 1)*2
        nbgates_all = (self.layer+1)*2
        
        nbgates_at_each_layer = []
        for i in range(0, self.qubits):
            nbgates_at_each_layer.append(nbgates_all)
        
        for i in range(0, self.layer + 1):
            if i%2 == 0: # apply first layer with entanglement
                j = 0
                ent = 0
                while j < self.qubits:
                    self.qc.ry(self.theta[self.__addAllAtIndex(nbgates_at_each_layer, j) + i*2], j)
                    self.qc.rz(self.theta[self.__addAllAtIndex(nbgates_at_each_layer, j) + i *2+ 1], j)   
                    j += 1
                if i != self.layer:
                    while ent < self.qubits:
                        self.qc.swap(ent, ent+1)
                        ent += 2
            else: 
                j = 0
                ent = 1
                while j < self.qubits:
                    self.qc.ry(self.theta[self.__addAllAtIndex(nbgates_at_each_layer, j) + i*2], j)
                    self.qc.rz(self.theta[self.__addAllAtIndex(nbgates_at_each_layer, j) + i*2 + 1], j)
                    j += 1
                self.qc.i(0)
                while ent < self.qubits -1:
                    self.qc.swap(ent, ent+1)
                    ent += 2
                self.qc.i(self.qubits-1)
    def __countOdd(self, num):
        count = 0
        for i in range(0, num + 1):
            if i%2 == 1:
                count += 1
            else:
                pass
        return count

    def __addAllAtIndex(self, mylist, index):
        a = mylist
        num = 0
        for i in range(0, index):
            num += a[i]
        return num
    
    def draw(self):
        print(self.qc)

    def getCircuit(self):
        return self.qc

    def statevector(self):
        simu_sv = BasicAer.get_backend('statevector_simulator')
        sv = execute(self.qc, simu_sv).result().get_statevector()
        if type == 'print':
            print(sv)
        return sv
    
    def counts(self, type = None):
        simu_sv = BasicAer.get_backend('statevector_simulator')
        counts = execute(self.qc, simu_sv).result().get_counts()
        if type == 'print':
            print(counts)
        return counts
        

def main():
    Circuit7(4, 5, theta).draw()
    pass

if __name__ == '__main__':
    main()