# https://github.com/sergiogh/tsp_mayhem/blob/d1c5d77c22e283254900eb174ce2d308def46ffd/solvers/vqe.py
import numpy as np
from qiskit import Aer, IBMQ
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.circuit.library import TwoLocal, RealAmplitudes
from qiskit.optimization.applications.ising import tsp
from qiskit.optimization.applications.ising.common import sample_most_likely

class Vqe:

    def calculate(self, G, cost_matrix, starting_node = 0):

        # Create nodes array for the TSP solver in Qiskit
        coords = []
        for node in G.nodes:
            coords.append(G.nodes[node]['pos'])

        tsp_instance = tsp.TspData(name = "TSP", dim = len(G.nodes), coord = coords, w = cost_matrix)

        qubitOp, offset = tsp.get_operator(tsp_instance)
        print("Qubits needed: ", qubitOp.num_qubits)
        #print(qubitOp.print_details())

        #backend = Aer.get_backend('statevector_simulator')
        backend = Aer.get_backend('qasm_simulator')

        # Use real backend
        #IBMQ.load_account()
        #provider = IBMQ.get_provider('ibm-q')
        #from qiskit.providers.ibmq import least_busy
        #backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits > qubitOp.num_qubits and not x.configuration().simulator ))
        #print(backend.name())

        quantum_instance = QuantumInstance(backend)

        optimizer = SPSA(maxiter=400)
        #optimizer = COBYLA(maxiter=200, rhobeg=0.3, tol=0.1, disp=True)
        ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=4, entanglement='full')
        ra = RealAmplitudes(qubitOp.num_qubits, reps=2)
        vqe = VQE(operator=qubitOp, var_form=ry, optimizer=optimizer, quantum_instance=quantum_instance)

        result = vqe.run(quantum_instance)

        x = sample_most_likely(result.eigenstate)

        if(tsp.tsp_feasible(x)):
            z = tsp.get_tsp_solution(x)
            print('solution:', z)
            return z
        else:
            print('no solution:', x)
            return []


    def iterations(_eval_count, param_set, means, estimator_error):
        print(_eval_count)
        print(param_set)
        print(means)
        print(estimator_error)
