# https://github.com/mhasan08/quantum-machine-learning/blob/0e3302c70fccfad6da2a49e1b88ebc88428917ed/qml.py

'''
    @author Munawar Hasan <munawar.hasan@nist.gov>
'''

import os
import exceptions
import constants
import utils
import numpy as np
from math import sqrt

from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators import Operator
import qiskit.aqua as qa
from qiskit import Aer
from qiskit import execute

from progressbar import ProgressBar

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Model:
    def __init__(self, num_of_qubits, U):

        self.logger = utils.Logger()
        if num_of_qubits is None:
            raise exceptions.GenericException(constants.ExceptionMessages.ERR_QUBITS)
        self.num_of_qubits = num_of_qubits
        if U is not None:

            if type(U) is list:
                U = np.array(U)
            self.U = U
            n, m = self.U.shape

            if n != m:
                raise exceptions.SquareMatrixException(m, n)
            elif n != 2**num_of_qubits:
                raise exceptions.UnitaryMatrixQubitsException(m, n, num_of_qubits)
        else:
            self.logger.__log__(self.__class__.__name__)
            self.logger.__log__("\t", constants.WarningMessages.WR, constants.WarningMessages.WR_UNITARY)

            if constants.Environment.OCT in os.environ[constants.Environment.PATH] and False:
                from oct2py import octave
                u_operator = octave.unitary(num_of_qubits, nout=1)
                self.U = u_operator
            else:
                u_operator = random_unitary(num_of_qubits)
                self.U = u_operator
        self.pred_index = None
        self.logger.__log__("\t", str(self.U))

    def get_metrics(self, y_true, y_pred):

        self.logger.__log__(constants.Messages.METRICS)

        predicted_labels = list()
        pbar = ProgressBar()

        for index in pbar(range(len(y_pred))):
            yp = y_pred[index]
            yt = y_true[index]

            if yp in list(self.pred_index.values()):
                predicted_labels.append(list(self.pred_index.keys())[list(self.pred_index.values()).index(yp)])
            else:
                predicted_labels.append(yt ^ 1)

        _acc = accuracy_score(y_true=y_true, y_pred=predicted_labels)
        _f1 = f1_score(y_true=y_true, y_pred=predicted_labels)
        _recall = recall_score(y_true=y_true, y_pred=predicted_labels)
        _precision = precision_score(y_true=y_true, y_pred=predicted_labels)
        dict_metrics = {
            'acc': _acc,
            'f1_score': _f1,
            'recall': _recall,
            'precision': _precision
        }

        return dict_metrics

    @classmethod
    def __fidelity_loss(cls):
        # todo
        pass

    @classmethod
    def __generate_hilbert_space(cls, x):
        temp = list()
        for item in x:
            proba = item
            alpha_beta = np.array([sqrt(float(1) - proba), sqrt(proba)])
            temp.append(alpha_beta)

        if len(temp) == 1:
            return temp[0].tolist()
        else:
            kron_product = np.kron(temp[0], temp[1])
            for i in range(2, len(temp)):
                kron_product = np.kron(kron_product, temp[i])

            return kron_product.tolist()

    @classmethod
    def __quantum_module(cls, num_of_qubits, state_vector, U, qubits):
        custom = qa.components.initial_states.Custom(
            num_qubits=num_of_qubits, state_vector=state_vector
        )
        circuit = custom.construct_circuit()
        cx = Operator(U)
        circuit.unitary(cx, qubits)
        circuit.measure_all()

        simulator = Aer.get_backend(constants.QISKIT.SIMULATOR)
        result = execute(circuit, backend=simulator).result()

        prob_list = [0] * (2**num_of_qubits)
        circuit_prob = result.get_counts(circuit)
        for k, v in circuit_prob.items():
            list_index = int(k, 2)
            prob_list[list_index] = v
        max_index = prob_list.index(max(prob_list))
        prob_list = [x / sum(prob_list) for x in prob_list]
        return prob_list, max_index

    def compute(self, x=None, pred=True, pred_index=None):
        if pred:
            if pred_index is None:
                self.logger.__log__(self.__class__.__name__)
                self.logger.__log__(constants.Messages.DELIMITER, constants.WarningMessages.WR, constants.WarningMessages.WR_PRED_INDEX)
                pred_index = {0: 0, 1: 2**self.num_of_qubits-1}
                self.logger.__log__(constants.Messages.DELIMITER, constants.Messages.PRED_INDEX, str(pred_index))
            else:
                if type(pred_index) is not dict:
                    raise exceptions.GenericException(constants.ExceptionMessages.ERR_PRED_INDEX)
                else:
                    self.logger.__log__(constants.Messages.DELIMITER, constants.Messages.PRED_INDEX, str(pred_index))
            self.pred_index = pred_index
        else:
            self.logger.__log__(constants.Messages.DELIMITER, constants.Messages.PREDICTION_IS_SET_FALSE)

        if x is None:
            raise exceptions.InputException()

        qubits = [i for i in range(self.num_of_qubits)]
        predicted_probabilities = list()
        predicted_labels = list()
        pbar = ProgressBar()
        for i in pbar(range(x.shape[0])):
            sample = x[i]
            custom_state = self.__generate_hilbert_space(sample)

            prob_list, max_index = self.__quantum_module(self.num_of_qubits, custom_state, self.U, qubits)

            predicted_probabilities.append(prob_list)
            predicted_labels.append(max_index)

        if pred:
            return predicted_labels
        else:
            return predicted_probabilities, predicted_labels

    def compute_step(self, x=None):
        # todo
        pass