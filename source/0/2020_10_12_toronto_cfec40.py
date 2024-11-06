# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/compiler/experiments/2020-10-12_toronto.py
from qiskit import IBMQ
from experiments.execute import run_experiments
from experiments.convert_error_information import value_to_ratio
from experiments.pickle_tools import pickle_dump, pickle_load


def run(multi_circuit_components, xtalk_path=None, reservation=False):
    """ 
    Args: 
        multi_circuit_components : benchmarking circuits 
    """

    # define backend
    backend_name = 'ibmq_toronto'
    IBMQ.load_account()
    if reservation:
        provider = IBMQ.get_provider(
            hub='ibm-q-keio', group='keio-internal', project='reservations')
    else:
        provider = IBMQ.get_provider(
            hub='ibm-q-keio', group='keio-internal', project='keio-students')
    backend = provider.get_backend(backend_name)

    #  crosstalk prop
    if xtalk_path is None:
        epc_path = "/Users/Yasuhiro/Documents/aqua/gp/errors_information/toronto_from20200903/xtalk_data_daily/epc/2020-10-11.pickle"
        epc_dict = pickle_load(epc_path)
        crosstalk_prop = value_to_ratio(epc_dict)
    else:
        crosstalk_prop = pickle_load(xtalk_path)

    jobfile_dir = "/Users/Yasuhiro/Documents/aqua/gp/experiments/jobfiles/ibmq_toronto/2020-10-12/"
    circ = run_experiments(
        jobfile_dir, multi_circuit_components=multi_circuit_components, backend=backend, crosstalk_prop=crosstalk_prop, shots=8192)


if __name__ == "__main__":
    """
    クロストークの影響を含む量子ビット組みを用いた実験。
    """
    multi_circuit_components_list = [
        {'Toffoli': 1, 'QAOA_3': 1},
        {'Toffoli': 1, 'QAOA_4': 1},
        {'Toffoli_SWAP': 1, 'QAOA_3': 1},
        {'Toffoli_SWAP': 1, 'QAOA_4': 1},
        {'Fredkin': 1, 'Toffoli': 1},
        {'Fredkin': 1, 'Toffoli_SWAP': 1},
        {'QFT_2': 1, 'Toffoli': 1},
        {'QFT_2': 1, 'Toffoli_SWAP': 1},
        {'QFT_3': 1, 'Toffoli': 1},
        {'QFT_3': 1, 'Toffoli_SWAP': 1},
        {'QFT_2': 1, 'Fredkin': 1, 'Toffoli': 1},
        {'QFT_2': 1, 'Fredkin': 1, 'Toffoli_SWAP': 1},
        {'QFT_3': 1, 'Fredkin': 1, 'Toffoli': 1},
        {'QFT_3': 1, 'Fredkin': 1, 'Toffoli_SWAP': 1},
    ]

    for multi_circuit_components in multi_circuit_components_list:
        run(multi_circuit_components)
