# https://github.com/FredericSauv/qc_optim/blob/c3229a5d336cf8efd084f5036352a094e6923ac0/tests/test_utilities.py
"""
Tests for utilities
"""

import pytest

import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Measure
from qiskit.utils import QuantumInstance
from qiskit.aqua.utils.backend_utils import is_simulator_backend

from qcoptim.ansatz import RandomAnsatz, TrivialAnsatz
from qcoptim.utilities import (
    add_random_measurements,
    RandomMeasurementHandler,
    transpile_circuit,
    make_quantum_instance,
)
from qcoptim.utilities.pytket import compile_for_backend

_TEST_IBMQ_BACKEND = 'ibmq_santiago'
_TRANSPILERS = ['instance', 'pytket']


def test_make_quantum_instance():
    """
    ensure can make sim/device quantum instances and noisy sims from devices
    """

    # device instance
    instance = make_quantum_instance(
        _TEST_IBMQ_BACKEND,
        simulate_ibmq=False,
    )
    assert not is_simulator_backend(instance.backend)

    # simulated device instance
    instance = make_quantum_instance(
        _TEST_IBMQ_BACKEND,
        simulate_ibmq=True,
    )
    assert is_simulator_backend(instance.backend)

    # sim instance
    instance = make_quantum_instance(
        'aer_simulator',
        simulate_ibmq=False,
    )
    assert is_simulator_backend(instance.backend)

    # this combination of args used to cause crash
    instance = make_quantum_instance(
        'aer_simulator',
        simulate_ibmq=False,
    )
    assert is_simulator_backend(instance.backend)


@pytest.mark.parametrize("device", ['ibmq_santiago', 'ibmq_manhattan'])
def test_pytket_compile_for_backend(device):
    """
    Pytket previously had bugs that meant compilation did not work on large
    IBMQ devices and for circuits with parameter objects. So, here we test two
    different devices (one large, one small) and use an ansatz that contains
    parameters
    """
    instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    ansatz = RandomAnsatz(2, 2)

    # this should do nothing to circ
    circ = ansatz.circuit
    circ2 = compile_for_backend(instance.backend, circ)
    assert circ2 == circ

    ibmq_instance = make_quantum_instance(device)

    # should work for an array and preserve names
    circs = [
        circ.copy(),
        circ.copy(),
    ]
    circs[0].name = 'test1'
    circs[1].name = 'test2'
    t_circs = compile_for_backend(ibmq_instance.backend, circs)
    assert len(t_circs) == 2
    assert t_circs[0].name == 'test1'
    assert t_circs[0] != circs[0]
    assert t_circs[1].name == 'test2'
    assert t_circs[1] != circs[1]

    # check parameters are still in circuit
    for param in ansatz.params:
        assert param in t_circs[0].parameters
        assert param in t_circs[1].parameters

    # full test would submit to instance and make sure it executes, but don't
    # want to do that because it'll be very slow


@pytest.mark.parametrize("method", _TRANSPILERS)
def test_transpile_circuit(method):
    """ """
    num_qubits = 4

    ibmq_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)
    ansatz = RandomAnsatz(num_qubits, 3)
    circ = ansatz.circuit

    t_circ, t_map = transpile_circuit(circ, ibmq_instance, method)
    assert t_circ != circ
    for qubit in range(num_qubits):
        assert qubit in t_map.keys()

    # check parameters are still in circuit
    for param in ansatz.params:
        assert param in t_circ.parameters


def test_add_random_measurements_active_qubits():
    """ """
    circ = RandomAnsatz(4, 3).circuit

    # test different values of active_qubits
    for active_qubits in [None, [1], [2, 3]]:
        new_circs = add_random_measurements(
            circ, 10, active_qubits=active_qubits)
        assert len(new_circs) == 10

        for mcirc in new_circs:

            # identify measured qubits
            measured_qubits = set()
            for instruction in mcirc.data:
                if isinstance(instruction[0], Measure):
                    measured_qubits.add(instruction[1][0].index)

            if active_qubits is None:
                test_set = set(range(circ.num_qubits))
            else:
                test_set = set(active_qubits)
            assert measured_qubits == test_set


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_random_measurement_handler(transpiler):
    """ """
    num_random = 10
    seed = 0

    def circ_name(idx):
        return 'test_circ'+f'{idx}'

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)

    ansatz = RandomAnsatz(2, 2, strict_transpile=True)
    rand_meas_handler = RandomMeasurementHandler(
        ansatz,
        transpile_instance,
        num_random,
        transpiler=transpiler,
        seed=seed,
        circ_name=circ_name,
    )

    point = np.ones(ansatz.nb_params)
    circs = rand_meas_handler.circuits(point)
    assert len(circs) == num_random
    assert circs[0].name == 'test_circ0'

    # simple check that __init__ is using `active_qubits` arg of
    # add_random_measurements as expected, by counting number of measurements
    for mcirc in circs:
        # identify measured qubits
        measured_qubits = []
        for instruction in mcirc.data:
            if isinstance(instruction[0], Measure):
                measured_qubits.append(instruction[1][0].index)
        assert len(measured_qubits) == 2

    # test that no circuits are returned on second request
    assert len(rand_meas_handler.circuits(point)) == 0

    # test reset
    rand_meas_handler.reset()
    assert len(rand_meas_handler.circuits(point)) == num_random
    assert rand_meas_handler.circuits(point) == []

    # change circuit name func
    rand_meas_handler.circ_name = lambda idx: 'HaarRandom' + f'{idx}'

    # test that new point releases lock
    point2 = 2. * np.ones(ansatz.nb_params)
    circs2 = rand_meas_handler.circuits(point2)
    assert len(circs2) == num_random
    assert circs2[0].name == 'HaarRandom0'


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_random_measurement_handler_trivial_ansatz(transpiler):
    """
    Test special case of ansatz with no parameters
    """
    num_random = 10
    seed = 0

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)

    circ = QuantumCircuit(2)
    rand_meas_handler = RandomMeasurementHandler(
        TrivialAnsatz(circ), transpile_instance, num_random,
        transpiler=transpiler, seed=seed,
    )

    circs = rand_meas_handler.circuits([])
    assert len(circs) == num_random
    assert rand_meas_handler.circuits([]) == []


@pytest.mark.parametrize("transpiler", _TRANSPILERS)
def test_random_measurement_handler_2d_point(transpiler):
    """
    Test correct behaviour with array of points
    """
    num_random = 10
    num_points = 3
    seed = 0

    transpile_instance = make_quantum_instance(_TEST_IBMQ_BACKEND)

    ansatz = RandomAnsatz(2, 2)
    rand_meas_handler = RandomMeasurementHandler(
        ansatz, transpile_instance, num_random, transpiler=transpiler,
        seed=seed,
    )

    points = np.random.random(num_points*ansatz.nb_params)
    points = points.reshape((num_points, ansatz.nb_params))
    circs = rand_meas_handler.circuits(points)
    assert len(circs) == num_random*num_points
    assert rand_meas_handler.circuits(points) == []

    # change one of points, should unlock
    points[0, :] = 2
    circs = rand_meas_handler.circuits(points)
    assert len(circs) == num_random*num_points
    assert rand_meas_handler.circuits(points) == []
