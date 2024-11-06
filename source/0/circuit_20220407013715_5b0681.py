# https://github.com/Amai-RusseMitsu/discrete-time-crystal-ch/blob/127689538c88aad9d20035f78cf86581e3c546bf/.history/setup/circuit_20220407013715.py
import numpy as np
from qiskit import QuantumCircuit

class DTC_Circ:

    def __init__(self,
                 N=None,
                 J=None,
                 t=None,
                 dt=None,
                 lamb=None,
                 omega=None,
                 h=None
                 ):
        self.parameters = {
            "N": N,
            "J": J,
            "t": t,
            "dt": dt,
            "lamb": lamb,
            "omega": omega,
            "h": h,
        }
        self.dtc_circ = QuantumCircuit(N, N, name='N=' + str(N) +
                                       ',J=' + str(J) +
                                       ',h=' + str(h) +
                                       ',lamb=' + str(lamb) +
                                       ',omega=' + str(omega) +
                                       ',dt=' + str(dt) +
                                       ',t=' + str(t)
                                       )
        self.N = N
        self.J = J
        self.lamb = lamb
        self.omega = omega
        self.h = h
        self.t = t
        self.dt = dt

    def prepare(self):
        for i in range(self.N):
            self.dtc_circ.h(i)
            self.dtc_circ.s(i)

    def measure(self):
        for i in range(self.N):
            self.dtc_circ.sdg(i)
            self.dtc_circ.h(i)
            self.dtc_circ.measure(i, i)

    def hamiltonian(self, t):

        def coeff(h, t, omega):
            return -h*np.cos(omega*t/2)**2

        # H0 terms with sigma^y and sigma^z
        for i in range(self.N):
            self.dtc_circ.ry(2*self.lamb*self.dt, i)
            self.dtc_circ.rz(2*self.lamb*self.dt, i)

        self.dtc_circ.barrier()
        # H1 terms with sigma^z
        for i in range(0, self.N-1, 2):
            self.dtc_circ.cx(i, i+1)
            self.dtc_circ.rz(-2*self.J*self.dt, i+1)
            self.dtc_circ.cx(i, i+1)

        for i in range(1, self.N-1, 2):
            self.dtc_circ.cx(i, i+1)
            self.dtc_circ.rz(-2*self.J*self.dt, i+1)
            self.dtc_circ.cx(i, i+1)

        self.dtc_circ.barrier()
        # H2 time dependent terms with sigma^x
        for i in range(self.N):
            self.dtc_circ.rx(2*coeff(self.h, t , self.omega)*self.dt, i)

    def generate_circuit(self):
        self.prepare()
        for i in range(int(self.t//self.dt)):
            self.hamiltonian(i*self.dt)
        self.measure()