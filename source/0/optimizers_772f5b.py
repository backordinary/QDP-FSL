# https://github.com/pchung39/quantum_variational_eigensolver/blob/339dcb928ed1f1624d56d91552031fcbe5a2e2d6/optimizers.py
import numpy as np
import functools
from qiskit import *
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel, pauli_error, depolarizing_error
from skquant.opt import minimize
from vqe import VQE
import json


class ClassicalOptimizer(object):

    def __init__(self):
        self.measurements = []
        self.angle_list = []

    def define_noise_model(self, error_prob, qubits, gates):

        angle_range = np.linspace(0, 2 * np.pi, 20)
        noise_model = NoiseModel()
        error = depolarizing_error(error_prob, qubits)
        noise_model.add_all_qubit_quantum_error(error, gates)

        return noise_model

    def process_results(self, results, optimize=False):
    
        if optimize:
            self.angle_list.append([r[1] for r in results])
            self.measurements.append([r[0] for r in results])
        else:
            self.angle_list.append([key for key, value in results["estimated_energy"].items()])
            self.measurements.append([results["estimated_energy"][key] for key, value in results["estimated_energy"].items()])


    def run_vqe(self,angle_range, noise_model=None):


        # Run once plain
        vqe = VQE(angle_range=angle_range)
        vqe.find_ground_state()
        results = vqe.return_vqe_info()

        self.process_results(results)

        # Run with noise model
        next_vqe = VQE(angle_range=angle_range, noise=noise_model)
        next_vqe.find_ground_state()
        next_results = next_vqe.return_vqe_info()

        self.process_results(next_results)
        
    def run_optimizer(self, angle_range, noise_model, optimizer, budget=100):
        

        bounds = np.array([[0, np.pi]], dtype=float)
        initial_point = np.array([0], dtype=float)

        # Run once plain for baseline 
        obj = VQE(angle_range=np.linspace(0, 2 * np.pi, 20))
        result, history = minimize(obj.get_energy_state, initial_point, bounds, budget, method=optimizer)
        self.process_results(results=history, optimize=True)

        # Run with noise model
        obj = VQE(angle_range=np.linspace(0, 2 * np.pi, 20), noise=noise_model)
        result, history = minimize(obj.get_energy_state, initial_point, bounds, budget, method=optimizer)
        self.process_results(results=history, optimize=True)


    def create_graph(self):

        plt.plot(self.angle_list[0], self.measurements[0], color='blue', label='No noise')
        plt.plot(self.angle_list[1], self.measurements[1], color='red', label="30% noise")
        plt.xlabel('Angle (radians)') 
        plt.ylabel('Energy state') 
        plt.xticks([])
        plt.legend()
        plt.show()

