# https://github.com/maximss635/SPBPU-qnn-labs/blob/cbdd35dd65aea9513c6ad9565ba1d9fdce978ee1/scripts/quantum_model.py
from qiskit import IBMQ, Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from time import time
from notify_bot import *


class QModel:
  def __init__(self, n_epochs, n_qubits):
    self.q_instance = QuantumInstance(Aer.get_backend('aer_simulator'), 
                                      shots=n_epochs)

    self.__n_epochs = n_epochs

    time_start = time()
    self.qnn = TwoLayerQNN(n_qubits, quantum_instance=self.q_instance)
    print('Time constructed qnn: {}'.format(time() - time_start))

    self.q_classifier = NeuralNetworkClassifier(self.qnn, 
                                                optimizer=COBYLA(),
                                                callback=self.__callback)
    
    print('n_cubits={}'.format(self.qnn.num_qubits))
    print('Simulator={}'.format(self.q_instance.backend))
    print('n_epochs={}'.format(n_epochs))


  def fit(self, X, y):
    time_start = time()
    self.__file_train_logs = open('last_train.log', 'w')

    self.__X_tain_size = X.shape[0]

    self.q_classifier.fit(X, y)
    
    self.__file_train_logs.close()
    print('Time training: {}'.format(time() - time_start))


  def __callback(self, weights, obj_func_eval):
    self.__file_train_logs.write('obj_func_eval={}, weights={}\n' \
        .format(obj_func_eval, weights))


  def test(self, X_test, y_test):
    y_pred = self.q_classifier.predict(X_test)
    
    # {-1, 1} -> {0, 1}
    y_pred = (y_pred + 1) // 2
    y_pred = y_pred.reshape(len(y_pred))

    report = self.__make_report(y_test, y_pred)
    print(report)
    send_telegram(report)

  
  def __make_report(self, y_true, y_pred):
    n = len(y_true)

    assert(n == len(y_pred))

    fp, fn, tp, tn = 0, 0, 0, 0

    for i in range(n):
      y1 = y_true[i]
      y2 = y_pred[i]

      if y1 == y2:
        if y1 == 0:
          tn += 1
        else:
          tp += 1
      else:
        if y1 == 0:
          fp += 1
        else:
          fn += 1

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    error_1 = fp / n
    error_2 = fn / n

    text = 'Параметры:' + '\n' + \
      'Эмулятор: {}'.format(self.q_instance.backend) + '\n' + \
      'Число признаков (кубит): {}'.format(self.qnn.num_qubits) + '\n' + \
      'Размер выбоки: тренировочная - {}, тестовая - {}' \
        .format(self.__X_tain_size, len(y_pred)) + '\n' + \
      'Число эпох: {}'.format(self.__n_epochs) + '\n\n' + \
      'Метрики:\n' + \
      'Ошибка первого рода: {}'.format(error_1) + '\n' + \
      'Ошибка второго рода: {}'.format(error_2) + '\n' + \
      'Accuracy: {}'.format(accuracy) + '\n' + \
      'Precision: {}'.format(precision) + '\n' + \
      'Recall: {}'.format(recall)

    return text
