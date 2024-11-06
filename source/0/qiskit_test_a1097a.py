# https://github.com/GuyPardo/experiment-manager/blob/3d2375e237de16c8e24d351a31756ffb148e6fc8/qiskit_test.py
import numpy as np
import importlib
from qiskit import QuantumCircuit

from dataclasses import dataclass
import experiment_manager
from qiskit import Aer
importlib.reload(experiment_manager)
from experiment_manager import *
import os
import sys
sys.path.append(os.path.abspath(r"G:\My Drive\guy PHD folder\git repos\qiskit-utils"))
import circuit_utils as cu
import Aer_sim_utils as au


@dataclass
class Noise:
    T1: float
    T2: float


class T1Experiment(QiskitExperimentDensityMat):
    def get_circ(self, config:Config):
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.delay(duration = config.delay.value, unit = config.delay.units)
        circ.save_density_matrix()
        noise = Noise(config.T1.value, 2*config.T1.value)
        trans_circ = au.get_transpiled(circ, config.backend.value, noise)

        return trans_circ

    def get_observables(self, config:Config, density_matrix):
        populations = density_matrix.probabilities() # an array
        output_config = Config(Parameter('populations', populations), Parameter('param', 1.0))
        return output_config







###########################################

'''
config = Config(Parameter('noise', [Noise(t1,2*t1) for t1 in np.linspace(0.001,100e-6, 10) ]),
                Parameter('delay', np.linspace(0,100e-6,100), 's'),
                Parameter('backend', Aer.get_backend('aer_simulator')))
                '''


config = Config(Parameter('T1',  np.linspace(0.0001e-6,100e-6, 30), 's'),
                Parameter('delay', np.linspace(10e-6,300e-6,30), 's'),
                Parameter('backend', Aer.get_backend('aer_simulator')))

exp = T1Experiment()
#a = Child()

#a.one_d()

#print(exp.get_circ(config).draw())

#job = exp.one_dimensional_job(config)

exp.sweep(config)

#res = exp.wait_result(job)

exp.labber_read(config)