# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/test/test_multi_transpile.py
from datetime import datetime, timezone
from pprint import pprint

from qiskit.compiler import transpile
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit
from qiskit.providers.models import BackendProperties
from qiskit.test.mock import FakeToronto
from qiskit.providers.models.backendproperties import Nduv, Gate

from experiments.compiler import CompilerBenchmark
from experiments.error_info_converter import value_to_ratio
from experiments.utils import pickle_load, pickle_dump
from multitasking_transpiler.palloq.compiler import multitasking_transpile


def make_qubit_with_error(readout_error):
    """Create a qubit for BackendProperties"""
    calib_time = datetime(
        year=2019, month=2, day=1, hour=0, minute=0, second=0, tzinfo=timezone.utc
    )
    return [
        Nduv(name="T1", date=calib_time, unit="µs", value=100.0),
        Nduv(name="T2", date=calib_time, unit="µs", value=100.0),
        Nduv(name="frequency", date=calib_time, unit="GHz", value=5.0),
        Nduv(name="readout_error", date=calib_time, unit="", value=readout_error),
    ]


def transpile_with_basegates(circ, initial_layout=None):
    basis_gates = ["u1", "u2", "u3", "cx"]
    backend = FakeToronto()

    transpiled_circ = transpile(
        circ,
        backend=backend,
        basis_gates=basis_gates,
        initial_layout=initial_layout,
        optimization_level=3,
    )
    return transpiled_circ


def test_noiseadaptive_multitask_layout():
    IBMQ.load_account()
    provider = IBMQ.get_provider(
        hub="ibm-q-keio", group="keio-internal", project="keio-students"
    )
    backend = provider.backends.ibmq_toronto

    calib_time = datetime(
        year=2020, month=8, day=1, hour=0, minute=0, second=0, tzinfo=timezone.utc
    )

    bprop = backend.properties(datetime=calib_time)

    ###################################################################
    # backend = None
    # calib_time = datetime(year=2020, month=8, day=1,
    #                       hour=0, minute=0, second=0, tzinfo=timezone.utc)
    # qubit_list = []
    # ro_errors = [0.01]*6
    # for ro_error in ro_errors:
    #     qubit_list.append(make_qubit_with_error(ro_error))
    # p01 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.1)]
    # p03 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.3)]
    # p12 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.3)]
    # p14 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.2)]
    # p35 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.3)]
    # p45 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.1)]
    # p25 = [Nduv(date=calib_time, name='gate_error', unit='', value=0.1)]
    # g01 = Gate(name="CX0_1", gate="cx", parameters=p01, qubits=[0, 1])
    # g03 = Gate(name="CX0_3", gate="cx", parameters=p03, qubits=[0, 3])
    # g12 = Gate(name="CX1_2", gate="cx", parameters=p12, qubits=[1, 2])
    # g14 = Gate(name="CX1_4", gate="cx", parameters=p14, qubits=[1, 4])
    # g35 = Gate(name="CX3_5", gate="cx", parameters=p35, qubits=[3, 5])
    # g45 = Gate(name="CX4_5", gate="cx", parameters=p45, qubits=[4, 5])
    # g25 = Gate(name="CX2_5", gate="cx", parameters=p25, qubits=[2, 5])
    # 
    # gate_list = [g01, g03, g12, g14, g35, g45, g25]
    # 
    # bprop = BackendProperties(
    #     last_update_date=calib_time, backend_name="test_backend",
    #     qubits=qubit_list, backend_version="1.0.0", gates=gate_list,
    #     general=[])
    ###################################################################

    crosstalk_prop = {
        (1, 4): {(7, 10): 2.007855196208965},
        (2, 3): {(5, 8): 2.2562876383458272},
        (5, 8): {(2, 3): 4.407297774712314},
        (7, 10): {(1, 4): 1.6495182758869775, (12, 15): 4.460931414407089},
        (10, 12): {
            (4, 7): 2.3646024088589743,
            (13, 14): 5.158153837151825,
            (15, 18): 3.4005240793851117,
        },
        (11, 14): {(12, 13): 1.5815379245637133},
        (12, 13): {(11, 14): 1.9429829606099185},
        (12, 15): {(7, 10): 3.3721649065903803, (13, 14): 3.4767073109961886},
        (13, 14): {
            (8, 11): 1.9294023975707386,
            (10, 12): 3.786805218079993,
            (12, 15): 6.021972074875656,
        },
        (15, 18): {(10, 12): 1.9467061167117026},
    }

    circ_list = [
        random_circuit(1, 2, measure=True, seed=1),
        random_circuit(3, 5, measure=True, seed=1),
        random_circuit(2, 3, measure=True, seed=1),
    ]
    circ_list = transpile_with_basegates(circ_list)

    # combine circuits
    multi_programming_circuit = multitasking_transpile(
        multi_circuits=circ_list,
        backend=backend,
        backend_properties=bprop,
        crosstalk_prop=crosstalk_prop,
    )

    if isinstance(multi_programming_circuit, list):
        for circ in multi_programming_circuit:
            print("################## " + circ.name + " ##################")
            print(circ)
    else:
        print(
            "################## "
            + multi_programming_circuit.name
            + " ##################"
        )
        print(multi_programming_circuit)


def test_transpile_on_regurated_hw():
    multi_circuit_components = {"QFT_2": 1, "Toffoli": 3}
    jobfile_dir = "/Users/Yasuhiro/Documents/aqua/gp/experiments/test/test_jobfile/"
    benchmark = CompilerBenchmark(backend_name="ibmq_toronto", reservations=False)

    xtalk_row_value = pickle_load(
        "/Users/Yasuhiro/Documents/aqua/gp/errors_information/toronto_from20201030/xtalk_data_daily/epc/2020-10-30.pickle"
    )
    xtalk_prop = value_to_ratio(xtalk_row_value, threshold=1.25)
    pprint(xtalk_prop)

    multi_circ = benchmark.run(multi_circuit_components, jobfile_dir, xtalk_prop)

    for circ in multi_circ:
        print("#################################################################")
        print(circ)


if __name__ == "__main__":

    # test_noiseadaptive_multitask_layout()
    test_transpile_on_regurated_hw()
