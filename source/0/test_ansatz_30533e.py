# https://github.com/FredericSauv/qc_optim/blob/d30cd5d55d89a9ce2c975a8f8891395e94e763f0/tests/test_ansatz.py
"""
"""

from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance

from qcoptim.ansatz import TrivialAnsatz, AnsatzFromCircuit
from qcoptim.utilities import make_quantum_instance, bind_params


def test_ansatz_from_function():
    """ """
    circ = QuantumCircuit(4)
    param_0 = Parameter('R0')
    param_1 = Parameter('R1')
    circ.ry(param_1, 0)
    circ.ry(param_0, 2)
    circ.cx(0, 1)
    circ.cx(2, 3)
    circ.ry(param_1, 1)
    circ.cx(1, 2)

    ansatz = AnsatzFromCircuit(circ)
    assert param_0 in ansatz.params
    assert param_1 in ansatz.params

    # simple test of expected behaviour of binding params
    circ2 = circ.copy()
    circ2 = circ2.bind_parameters(dict(zip([param_0, param_1], [0, 1])))
    assert bind_params(ansatz.circuit, [0, 1], ansatz.params)[0] == circ2

    # ansatz and original circ should be decoupled
    assert ansatz.circuit == circ
    circ.measure_all()
    assert ansatz.circuit != circ


def test_ansatz_transpile():
    """
    Transpile code is in the BaseAnsatz class, but that is not directly
    instanceable. TrivialAnsatz is the simplest derived class so that is used
    """
    # simple circuit linking all qubits
    circ = QuantumCircuit(4)
    circ.h(0)
    circ.cx(0, 1)
    circ.h(2)
    circ.cx(2, 3)
    circ.cx(1, 2)
    ansatz = TrivialAnsatz(circ, strict_transpile=False)

    sim_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    ibmq_instance = make_quantum_instance('ibmq_santiago')

    # should raise AttributeError
    try:
        _ = ansatz.transpiler_map
        assert False, 'should have raised error'
    except AttributeError:
        pass

    # test these all work
    _ = ansatz.transpiled_circuit(ibmq_instance, method='instance',
                                  enforce_bijection=True)
    _ = ansatz.transpiler_map
    _ = ansatz.transpiled_circuit(ibmq_instance, method='pytket',
                                  enforce_bijection=True)
    _ = ansatz.transpiler_map

    # these should fail
    ansatz.strict_transpile = True
    try:
        ansatz.transpiled_circuit(ibmq_instance, method='instance',)
        assert False, 'should have raised error'
    except ValueError:
        pass
    try:
        ansatz.transpiled_circuit(sim_instance, method='pytket',)
        assert False, 'should have raised error'
    except ValueError:
        pass
