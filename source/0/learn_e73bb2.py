# https://github.com/konakaji/aae/blob/7b99ea7ab6c6e336c147bef22b9d83a0cb016ca7/demo/learn.py
import math
import qiskit, qulacs
from qwrapper.circuit import QulacsCircuit, QiskitCircuit
from aae.extention.data_learning import DataLearning
from aae.extention.aae import AAETrainingMethod, PositiveAAETrainingMethod, AAEUtil

DEMO_FILENAME = "demo"
N_SHOT = 200


def load():
    print("----load----")
    data_learning = DataLearning(n_qubit=3, layer=5, type="qulacs")
    data_learning.load("demo.model")
    print(data_learning.get_state_vector())

    data_learning = DataLearning(n_qubit=3, layer=5, type="qiskit")
    data_learning.load("demo.model")
    print(data_learning.get_state_vector())
    # add gates for quantum algorithm-------
    # qc.xxx()
    # ---------------------------------------
    qc = QulacsCircuit(3)
    data_learning.add_data_gates(qc)
    qc.post_select(0, 1)
    samples = qc.get_state_vector()
    print(samples)


def learn():
    print("----learn----")
    data_learning = DataLearning(n_qubit=3, layer=5, type="qulacs")
    training_method = AAETrainingMethod(iteration=30, idblock=False)
    array = [0] * 4
    array[2] = 1
    data_learning.learn(array, training_method=training_method)
    print(data_learning.get_state_vector())
    data_learning.save_model("demo.model")
    data_learning.save_cost_transition("cost.txt")


def get_count(job_result):
    return job_result.get_counts().items()


def load_positive():
    print("----load positive----")
    n_qubit = 3
    data_learning = DataLearning(n_qubit=n_qubit, layer=2)
    data_learning.load("positive.model")
    print(data_learning.get_state_vector())


def learn_positive():
    print("----learn positive----")
    n_qubit = 3
    training_method = PositiveAAETrainingMethod(iteration=20)
    data_learning = DataLearning(n_qubit=n_qubit, layer=2)
    result = data_learning.learn(normalize([0.1, 0.3, 0.4, 0.2, 0.4, 0.5, 0.3, 0.2]), training_method=training_method)
    print(result)
    data_learning.save_model("positive.model")
    data_learning.save_cost_transition("positive.txt")


def normalize(state_array):
    norm = 0
    for v in state_array:
        norm = norm + v * v
    return [v / math.sqrt(norm) for v in state_array]


if __name__ == '__main__':
    learn()
    # load()
    # learn_positive()
    # load_positive()
