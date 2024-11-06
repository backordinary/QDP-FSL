# https://github.com/AidanGG/mps_tomo/blob/aa766e36ba9c22700e09495089a7b52313ba9209/mps_tomo/wip/StateTomographyProduction.py
import numpy as np
import qiskit
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer
from qiskit import BasicAer, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits
from qiskit.ignis.verification.tomography import StateTomographyFitter
from qiskit.providers.aer import noise
from qiskit.providers.aer.extensions import *
from qiskit import IBMQ


class tomographicConstruction:
    """Class to implement the tomographic circuits necessary for MPS reconstruction

    Initialisation:
    ------------------------------
    R : int
        ansatz bond dimension
    nQ : int
        number of qubits in the circuit
    initial_circuit : qiskit QuantumCircuit object
        Circuit which produces the state which is the object of characterisation
    backend : IBMQ backend
        Hardware on which the circuits are to be run. Defaults to Aer.get_backend('qasm_simulator')
    """

    def __init__(
        self, R, nQ, initial_circuit, backend=Aer.get_backend("qasm_simulator")
    ):
        self.R = R
        self.nQ = nQ
        self.initial_circuit = initial_circuit
        self.backend = backend

    def collect_noisy_tomography_data(
        self, noise_model, coupling_map, returnFull=False, method="cvx"
    ):

        K = int(np.ceil(np.log2(self.R)) + 1)
        nDMs = self.nQ - K + 1
        DM_list = []
        q = self.initial_circuit.qregs[0]

        # Get the basis gates for the noise model
        basis_gates = noise_model.basis_gates

        # Select the QasmSimulator from the Aer provider
        simulator = Aer.get_backend("qasm_simulator")

        for i in range(nDMs):
            tomography_circuits = state_tomography_circuits(
                self.initial_circuit, q[i : i + K]
            )
            result = execute(
                tomography_circuits,
                simulator,
                noise_model=noise_model,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                shots=8192,
            ).result()
            state_tom = StateTomographyFitter(result, tomography_circuits)
            DM = state_tom.fit(method=method)
            DM_list.append(DM)
        if returnFull:
            tomography_circuits = state_tomography_circuits(self.initial_circuit, q)
            result = execute(
                tomography_circuits,
                simulator,
                noise_model=noise_model,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                shots=8192,
                backend_options={"method": "density_matrix"},
            ).result()
            state_tom = StateTomographyFitter(result, tomography_circuits)
            full_DM = state_tom.fit(method=method)
        if returnFull:
            return (DM_list, full_DM)
        else:
            return DM_list

    def collect_true_noisy_dm(self, noise_model, coupling_map):

        K = int(np.ceil(np.log2(self.R)) + 1)
        nDMs = self.nQ - K + 1
        DM_list = []
        q = self.initial_circuit.qregs[0]

        # Get the basis gates for the noise model
        basis_gates = noise_model.basis_gates

        # Select the QasmSimulator from the Aer provider
        simulator = Aer.get_backend("qasm_simulator")

        self.initial_circuit.snapshot_density_matrix("final")

        result = execute(
            self.initial_circuit,
            Aer.get_backend("qasm_simulator"),
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
        ).result()

        tmp = np.array(
            result.data(0)["snapshots"]["density_matrix"]["final"][0]["value"]
        )
        DM = tmp[:, :, 0] + 1.0j * tmp[:, :, 1]
        return DM

    def collect_tomography_data(self):
        K = int(np.ceil(np.log2(self.R)) + 1)
        nDMs = self.nQ - K + 1
        DM_list = []
        q = self.initial_circuit.qregs[0]
        for i in range(nDMs):
            tomography_circuits = state_tomography_circuits(
                self.initial_circuit, q[i : i + K]
            )
            job = execute(
                tomography_circuits, self.backend, shots=8192, optimization_level=0
            )
            result = job.result()
            state_tom = StateTomographyFitter(result, tomography_circuits)
            DM = state_tom.fit(method="cvx")
            DM_list.append(DM)
        self.DM_list = DM_list
        return DM_list


def create_GHZ(nQ):
    q = QuantumRegister(nQ)
    circuit = QuantumCircuit(q)
    circuit.h(q[0])
    for i in range(nQ - 1):
        circuit.cx(q[i], q[i + 1])
    return circuit


def W_state(nQ):
    def B_p(circ, p, q1, q2):
        theta = np.arcsin(np.sqrt(p))
        circ.u3(-theta, 0, 0, q2)
        circ.cx(q1, q2)
        circ.u3(theta, 0, 0, q2)
        circ.cx(q2, q1)

    q = QuantumRegister(nQ)
    # c = ClassicalRegister(nQ)
    # circuit = QuantumCircuit(q,c)
    circuit = QuantumCircuit(q)
    circuit.x(q[0])
    for i in range(nQ - 1):
        B_p(circuit, 1 / (nQ - i), q[i], q[i + 1])
    # circuit.measure(q,c)
    return circuit


def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)
