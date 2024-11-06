# https://github.com/lslattery11/quantum_kernel/blob/56b6a34d5b6e416296abf94d72b9c7f67559bf53/code/feature_maps/HeisenbergZZMultiScale.py
import numpy as np
from qiskit import QuantumCircuit,QuantumRegister
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit import Parameter,ParameterVector

class HZZMultiscaleFeatureMap(BlueprintCircuit):
    """
    Feature map for time evolution of the 1D Heisenberg model. H = (need to think over appropraite choice). Time evolution is given by first order Suzuki-Trotter.
    Currently implemented with OBC as in Ruslan's original paper but I think PBC makes more sense for long-range interaction
    so that each feature can be equally weighted in the Hamiltonian.
    """
    def __init__(
        self,
        feature_dim: int,
        n_trotter: int,
        init_state: str,
        init_state_seed: int,
        lam0 : float = 1.0,
        lam1 : float = 1.0,
        lam2 : float = 1.0,
        h_layer: int = 1,
        name: str = 'HZZMulti'
    ) -> None:
        super().__init__(name=name)
        self._num_qubits = feature_dim
        self._feature_dimension = feature_dim
        self._support_parameterized_circuit = True
        self._init_state=init_state
        self._init_state_seed=init_state_seed

        #self.scaling_factor= 2 * evo_time/n_trotter
        self.lam0=lam0
        self.lam1=lam1
        self.lam2=lam2
        self.h_layer=h_layer
        #self.evo_time=evo_time
        self.n_trotter=n_trotter

        self.add_register(QuantumRegister(self._num_qubits,'q'))

    def _build(self):
        super()._build()
        qc=self.construct_circuit(self.qregs)
        inst=qc.to_instruction()
        qargs=qc.qubits
        cargs=qc.clbits
        self._data=[(inst,qargs,cargs)]
        self._update_parameter_table(inst)
        return

    def _check_configuration(self) -> bool:
        valid=False
        if self._num_qubits>=self._feature_dimension:
            valid=True
        return valid

    def construct_circuit(self,qr=None,inverse=False):
        params = ParameterVector(name='x', length=self._feature_dimension)
        qr=self.qregs[0]
        qc=QuantumCircuit(qr)


        qc=self.construct_init_state(qc,qr)
        for layer in range(self.n_trotter):
            #h layer
            if self.h_layer==1 or layer==0:
                for q1 in range(self._feature_dimension):
                    qc.h(qr[q1])
            #Z data layer
            for q1 in range(self._feature_dimension):
                phase=self.lam0*(2*params[q1])
                qc.p(phase,qr[q1])
            #ZZ data layer (maybe the ordering is wrong?)
            for q1 in range(self._feature_dimension):
                for q2 in range(q1 + 1, self._feature_dimension):
                    #phase=2*(np.pi-self.lam1*params[q1])*(np.pi-self.lam1*params[q2]))
                    phase=2*(self.lam1*params[q1])*(self.lam1*params[q2])
                    qc.cx(qr[q1],qr[q2])
                    qc.p(phase,qr[q2])
                    qc.cx(qr[q1],qr[q2])

            #J1 non-data layer
            if self.lam2 != 0:
                for q1 in range(self._feature_dimension):
                    q2=q1+1
                    if q2 >= self._num_qubits:
                        continue
                    #decay of long range interactions
                    # XX
                    qc.h([qr[q1],qr[q2]])
                    qc.cx(qr[q1],qr[q2])
                    qc.rz(self.lam2, qr[q2])
                    qc.cx(qr[q1],qr[q2])
                    qc.h([qr[q1],qr[q2]])
                    # YY
                    qc.rx(np.pi/2, [qr[q1],qr[q2]])
                    qc.cx(qr[q1],qr[q2])
                    qc.rz(self.lam2, qr[q2])
                    qc.cx(qr[q1],qr[q2])
                    qc.rx(np.pi/2, [qr[q1],qr[q2]])
                    # ZZ
                    qc.cx(qr[q1],qr[q2])
                    qc.rz(self.lam2, qr[q2])
                    qc.cx(qr[q1],qr[q2])
            
        return qc


    def construct_init_state(self,qc,qr):
        np.random.seed(self._init_state_seed)
        init_state=self._init_state
        if init_state == 'Haar_random':
            # pick and fix Haar-random single-qubit state
            random_params = np.random.uniform(size=(len(qr), 3))
            for i,qreg in enumerate(qr):
                qc.rx(2 * random_params[i][0], qreg)
                qc.ry(2 * random_params[i][1], qreg)
                qc.rz(2 * random_params[i][2], qreg)
        elif init_state == 'Basis_random':
            # random computational basis state
            for i,qreg in enumerate(qr):
                if np.random.randint(0, 2):
                    qc.x(qreg)
        elif init_state == 'X_random':
            # random X rotation on each qubit basis state
            random_params = np.random.uniform(size=(self._feature_dimension))
            for i,qreg in enumerate(qr):
                if np.random.randint(0, 2):
                    qc.rx(random_params[i], qreg)
        elif init_state=='zero':
            pass
        else:
            raise ValueError(f"Unknown initial state init_state={init_state}")
        return qc



    