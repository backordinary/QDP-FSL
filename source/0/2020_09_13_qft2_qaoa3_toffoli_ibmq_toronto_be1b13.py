# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/compiler/experiments/toronto_2020-09-13/2020-09-13_QFT2_QAOA3_Toffoli_ibmq_toronto.py
import pickle
from qiskit import IBMQ
from experiments.execute import run_experiments
from experiments.convert_error_information import value_to_ratio


def run(xtalk_path=None):
    # define backend
    backend_name = 'ibmq_toronto'
    IBMQ.load_account()
    # reservation
    provider = IBMQ.get_provider(
        hub='ibm-q-keio', group='keio-internal', project='reservations')
    backend = provider.get_backend(backend_name)

    # get crosstalk prop
    if xtalk_path is None:
        epc_day_path = "/Users/Yasuhiro/Documents/aqua/gp/errors_information/toronto_from20200903/xtalk_data_daily/epc/2020-09-12.pickle"
        epc_day = _pickle_load(epc_day_path)
        crosstalk_prop = value_to_ratio(epc_day, threshold=1.0)
    else:
        crosstalk_prop = _pickle_load(xtalk_path)

    # benchmarking circuit
    multi_circuit_components = {'QFT_2': 1,
                                'QAOA_3': 1,
                                'Toffoli': 1,
                                }
    jobfile_dir = "/Users/Yasuhiro/Documents/aqua/gp/experiments/jobfiles/ibmq_toronto/"
    circ = run_experiments(
        jobfile_dir, multi_circuit_components=multi_circuit_components, backend=backend, crosstalk_prop=crosstalk_prop, shots=8192)


def _pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


if __name__ == "__main__":
    xtalk_path = "/Users/Yasuhiro/Documents/aqua/gp/errors_information/toronto_from20200903/xtalk_data_daily/ratio/2020-09-12.pickle"
    run(xtalk_path)
