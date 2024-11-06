# https://github.com/oliverfunk/quantum-natural-gradient/blob/911a1cece8e78cae637a356b059a7f9649110347/QNG.py
import qiskit.aqua.components.variational_forms as vf
import math
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.utils.run_circuits import find_regs_by_name
from qiskit.quantum_info import Pauli
from qiskit import Aer, QuantumCircuit
import numpy as np
from qiskit.aqua.components.optimizers.cg import Optimizer

from qiskit import Aer

# lib from Qiskit Aqua
from qiskit.aqua import QuantumInstance  # , Operator
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP, POWELL, NELDER_MEAD

# lib from Qiskit Aqua Chemistry
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

import logging as logger


class QNG(Optimizer):
    _C0 = 2 * np.pi * 0.1

    CONFIGURATION = {
        'name': 'QNG',
        'description': 'Quantum Natural Gradient',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qng_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 20
                },
                'disp': {
                    'type': 'boolean',
                    'default': False
                },
                'eta': {
                    'type': 'number',
                    'default': 0.001
                },
                'c0': {
                    'type': 'number',
                    'default': _C0
                },
                'c1': {
                    'type': 'number',
                    'default': 0.1
                },
                'c2': {
                    'type': 'number',
                    'default': 0.602
                },
                'c3': {
                    'type': 'number',
                    'default': 0.101
                },
                'c4': {
                    'type': 'number',
                    'default': 0
                },
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'disp', 'eta'],
        'optimizer': ['local']
    }

    def __init__(self,
                 num_qbits,
                 ry_depth,
                 max_trials=1000,
                 eta=0.1,
                 ):

        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v

        self.num_qbits = num_qbits
        self.ry_depth = ry_depth

        self._max_trials = max_trials

        c0 = 2 * np.pi * 0.1
        c1 = 0.1
        c2 = 0.602
        c3 = 0.101
        c4 = 0
        self._parameters = np.array([c0, c1, c2, c3, c4])

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)
        theta = np.array(initial_point)
        obj_fun = objective_function
        last_avg = 1
        save_steps = 1

        self._calibration(objective_function, initial_point, 25)

        eta = self._options['eta']

        theta_plus_save = []
        theta_minus_save = []
        cost_plus_save = []
        cost_minus_save = []
        theta_best = np.zeros(theta.shape)
        for k in range(self._max_trials):
            # SPSA Parameters

            c_spsa = float(self._parameters[1]) / np.power(k + 1, self._parameters[3])

            delta = 2 * aqua_globals.random.randint(2, size=np.shape(theta)[0]) - 1
            # plus and minus directions
            theta_plus = theta + c_spsa * delta
            theta_minus = theta - c_spsa * delta
            # cost function for the two directions
            if self._max_evals_grouped > 1:
                cost_plus, cost_minus = obj_fun(np.concatenate((theta_plus, theta_minus)))
            else:
                cost_plus = obj_fun(theta_plus)
                cost_minus = obj_fun(theta_minus)
            # derivative estimate
            gradient = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)

            # Calc QGT
            QGT = self.determine_QGT(theta)

            inv_QGT = np.real(np.linalg.pinv(QGT))

            with open('SPSA_cost.txt', 'a') as f:
                f.write('{}\n'.format(objective_function(theta)))

            print(objective_function(theta))

            # updated theta
            theta = theta - eta * inv_QGT @ gradient

            # saving
            if k % save_steps == 0:
                logger.debug('Objective function at theta+ for step # {}: {:.7f}'.format(k, cost_plus))
                logger.debug('Objective function at theta- for step # {}: {:.7f}'.format(k, cost_minus))
                theta_plus_save.append(theta_plus)
                theta_minus_save.append(theta_minus)
                cost_plus_save.append(cost_plus)
                cost_minus_save.append(cost_minus)
                # logger.debug('objective function at for step # {}: {:.7f}'.format(k, obj_fun(theta)))

            if k >= self._max_trials - last_avg:
                theta_best += theta / last_avg
        # final cost update
        cost_final = obj_fun(theta_best)
        logger.debug('Final objective function is: %.7f' % cost_final)

        return [cost_final, theta_best, cost_plus_save, cost_minus_save,
                theta_plus_save, theta_minus_save]

    def _calibration(self, obj_fun, initial_theta, stat):
        """Calibrates and stores the SPSA parameters back.

        SPSA parameters are c0 through c5 stored in parameters array

        c0 on input is target_update and is the aimed update of variables on the first trial step.
        Following calibration c0 will be updated.

        c1 is initial_c and is first perturbation of initial_theta.

        Args:
            obj_fun (callable): the function to minimize.
            initial_theta (numpy.array): initial value for the variables of
                obj_fun.
            stat (int) : number of random gradient directions to average on in
                the calibration.
        """

        target_update = self._parameters[0]
        initial_c = self._parameters[1]
        delta_obj = 0
        logger.debug("Calibration...")
        for i in range(stat):
            if i % 5 == 0:
                logger.debug('calibration step # {} of {}'.format(str(i), str(stat)))
            delta = 2 * np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
            theta_plus = initial_theta + initial_c * delta
            theta_minus = initial_theta - initial_c * delta
            if self._max_evals_grouped > 1:
                obj_plus, obj_minus = obj_fun(np.concatenate((theta_plus, theta_minus)))
            else:
                obj_plus = obj_fun(theta_plus)
                obj_minus = obj_fun(theta_minus)
            delta_obj += np.absolute(obj_plus - obj_minus) / stat

        self._parameters[0] = target_update * 2 / delta_obj \
                              * self._parameters[1] * (self._parameters[4] + 1)

        # logger.debug('Calibrated SPSA parameter c0 is %.7f' % self._parameters[0])

    def determine_QGT(self, theta):
        QGT = []
        p_str = 'I' * self.num_qbits

        for layer_idx in range(0, self.ry_depth):
            for qb_idx in range(self.num_qbits):
                pauli_string = list(p_str)
                pauli_string[qb_idx] = 'Y'
                "".join(pauli_string)

                circ = self.get_subcricuit(theta, layer_idx)

                QGT.append(1 - self.calc_expectataion(pauli_string, circ) ** 2)

        QGT = np.diag(QGT)

        return QGT

    def calc_expectataion(self, pauli_str, sub_circuit):
        qubit_op = WeightedPauliOperator([[1, Pauli.from_label(pauli_str)]])
        sv_mode = False

        qi = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=1000, seed_simulator=100,
                             seed_transpiler=2)

        if qi.is_statevector:
            sv_mode = True

        # Make sure that the eval quantum/ classical registers in the circuit are named 'q'/'c'
        qc = qubit_op.construct_evaluation_circuit(statevector_mode=sv_mode,
                                                   wave_function=sub_circuit,
                                                   qr=find_regs_by_name(sub_circuit, 'q'),
                                                   use_simulator_operator_mode=True)

        result = qi.execute(qc)
        avg, std = qubit_op.evaluate_with_result(statevector_mode=sv_mode,
                                                 result=result,
                                                 use_simulator_operator_mode=True)

        return avg

    def get_subcricuit(self, theta_params, layer):
        assert layer >= 0

        if layer == 0:
            return QuantumCircuit(self.num_qbits)

        circ = vf.RY(num_qubits=self.num_qbits, depth=layer, skip_final_ry=True)
        p = theta_params[0:self.num_qbits * layer]

        return circ.construct_circuit(parameters=p)

