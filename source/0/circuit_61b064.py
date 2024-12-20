# https://github.com/damazz/HQCA/blob/8c28c49dc46640791806b123ac218d0d96b55152/hqca/core/_circuit.py
from qiskit import execute
from abc import ABC,abstractmethod
from qiskit.circuit import Parameter
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit
from math import pi


class Circuit(ABC):
    @abstractmethod
    def __init__(self,
            quantstore,
            Nq=None,
            _name=False,
            ):
        self.qs = quantstore
        if type(Nq)==type(None):
            self.Nq = self.qs.Nq_tot
        else:
            self.Nq = Nq
        self.q = QuantumRegister(self.Nq,name='q')
        self.c = ClassicalRegister(self.Nq,name='c')
        self.name = _name
        if _name==False:
            self.qc = QuantumCircuit(self.q,self.c)
        else:
            self.qc = QuantumCircuit(self.q,self.c,name=_name)

    @abstractmethod
    def apply(self,
            Instruct,
            ):
        for var,fxn in Instruct.gates:
            fxn(self,*var)

    @abstractmethod
    def tomography(self,
            Instruct,
            ):
        for var,fxn in Instruct.generate_tomography(self):
            fxn(self,*var)


class GenericCircuit(Circuit):

    def __init__(self,**kwargs):
        Circuit.__init__(self,**kwargs)
        self.swap = {i:i for i in range(self.Nq)}
        self.sl = []

    def apply(self,**kwargs):
        Circuit.apply(self,**kwargs)

    def h(self,q):
        self.qc.s(self.q[q])
        self.qc.sx(self.q[q])
        self.qc.s(self.q[q])

    def s(self,q):
        self.qc.rz(pi/2,self.q[q])

    def sx(self,q):
        # square root of X gate
        self.qc.sx(self.q[q])

    def si(self,q):
        self.qc.rz(-pi/2,self.q[q])

    def Cx(self,q,p):
        self.qc.cx(self.q[q],self.q[p])

    def Cz(self,q,p):
        self.qc.cz(self.q[q],self.q[p])

    def Rx(self,q,val):
        self.qc.rx(val,self.q[q])

    def barrier(self,q,p):
        self.qc.barrier(self.q[q],self.q[p])

    def Ry(self,q,val):
        self.qc.ry(val,self.q[q])

    def Rz(self,q,val):
        self.qc.rz(val,self.q[q])

    def x(self,q):
        self.qc.x(self.q[q])

    def y(self,q):
        self.qc.rz(-pi/2,self.q[q])
        self.qc.x(self.q[q])
        self.qc.rz(+pi/2,self.q[q])

    def z(self,q):
        self.qc.rz(pi,self.q[q])

    def U3(self,q,theta,phi,lamb):
        self.qc.u3(theta,phi,lamb,self.q[q])

    def Sw(self,q,p):
        #self.qc.swap(self.q[q],self.q[p])
        self.swap[p]=q
        self.swap[q]=p
        self.sl.append([p,q])

    def tomography(self,**kwargs):
        Circuit.tomography(self,**kwargs)
