# https://github.com/panmx9909/MLPhys/blob/360c36a3145561b0a7b26c4972ebb097e2901a5e/QC/noNoise/QCutils.py
from multiprocessing import set_forkserver_preload
import numpy as np
from qiskit import *
from qiskit.tools.monitor import job_monitor
#import qiskit.tools.jupyter
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import copy

class QuGateWithGrad:
    rDict = {'Rx': 0.5, 'Ry':0.5, 'Rz':0.5, 'Rxx': 0.5, 'Ryy': 0.5,'Rzz': 0.5}

    qiskitGateDict = {'Rx':QuantumCircuit.rx, 'Ry':QuantumCircuit.ry,'Rz':QuantumCircuit.rz,
    'Rxx':QuantumCircuit.rxx,'Ryy':QuantumCircuit.ryy,'Rzz':QuantumCircuit.rzz}

    def __init__(self,symbol: str, registerNum: tuple):
        self.qiskitGate = self.qiskitGateDict[symbol]
        self.r = self.rDict[symbol]
        
        self.registerNum = registerNum
        self.symbol = symbol
        self.gradNum = 0 # gradNum -1/+1 add or minus r
    
    
    def put(self,theta) -> tuple:
        
        return self.qiskitGate, theta + self.gradNum*( np.pi/ (4* self.r))
    

    def setGrad(self,gradNum):

        assert gradNum == -1 or gradNum == 1

        self.gradNum = gradNum



class QuCircWithGrad:

    simShots = 1000

    def __init__(self,N : int):
        self.thetas = []
        self.gates = []
        self.regiNums = []
        self.isConstructed = False
        self.qc = None
        self.N  = N
    
    def add_gate(self,symbol: str,registerNum: tuple,newTheta = None):
        '''
        add a gate with a random theta
        '''
        if type(registerNum) is int:
            registerNum = tuple([registerNum])
        
        if not newTheta:
            newTheta = np.random.random() * 2 * np.pi
        
        newGate = QuGateWithGrad(symbol,registerNum)
        self.regiNums.append(registerNum)
        self.gates.append(newGate)
        self.thetas.append(newTheta)

    def genGradCirc(self,gradIdx : int):
        """
        generate two circuit with plus and minus gradient
        """
        newCircPlus = copy.deepcopy(self)
        newCircMinus = copy.deepcopy(self)

        newCircPlus.gates[gradIdx].setGrad(1)
        newCircMinus.gates[gradIdx].setGrad(-1)

        newCircPlus.construct()
        newCircMinus.construct()

        return newCircPlus,newCircMinus
    
    def genGradCircList(self):
        gradCircs = []
        for i in range(self.size()):
            gradCircs.append(self.genGradCirc(i))
        return gradCircs
    
    def getThetaGrad(self):
        """
        return an array, gradients of the corresponding parameters
        """
        r = 0.5
        circList = self.genGradCircList()
        thetaGrad = np.zeros(self.size())

        for i in range(self.size()):
            circPlus, circMinus = circList[i]
            grad1 = r * (circPlus.calc() - circMinus.calc())
            assert np.abs(np.imag(grad1)) < 0.001
            thetaGrad[i] = grad1

        return thetaGrad
            

    def construct(self):
        '''
        construct the circuit using added gates and append measure
        '''
        N = self.N
        self.isConstructed = True

        self.q = QuantumRegister(N)
        self.c = ClassicalRegister(N)
        self.qc = QuantumCircuit(self.q,self.c)

        for i in range(len(self.gates)):
            gate = self.gates[i]
            theta = self.thetas[i]
            registerNum = self.regiNums[i]
            kitGate, para = gate.put(theta)
            kitGate(self.qc,para,*registerNum)
        
        self.qc.measure(self.q,self.c)
    
    def getCircuit(self):
        assert self.isConstructed, "Circuit not constructed"
        return self.qc
    
    def size(self):
        """
        return the number of gates in the circuit
        """
        return len(self.gates)

    def updateThetas(self,newThetas):
        self.thetas = newThetas
        self.construct()
    
    def setCostFun(self,f):
        """
        set a cost function that have:
            input from the result of a circuit
            output a scalar
        """
        self.costFun = f
    
    def calc(self):

        backend = Aer.get_backend('statevector_simulator')
        job = execute(self.qc, backend, shots=self.simShots)
        res = job.result().get_counts(self.qc)
        cost = self.costFun(res)
        
        return cost