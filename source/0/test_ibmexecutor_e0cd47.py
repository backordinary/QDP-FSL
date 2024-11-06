# https://github.com/albaaparicio12/TFG-App/blob/38a589b4d9a96d0d71fe61cdd89093d2ce827021/tests/test_IBMExecutor.py
from extended.IBMExecutor import IBMExecutor
from qiskit import IBMQ

TOKEN = "5fdc8febffc863044dd6a9595abbc9b49a87a1bc36a3869f2fe970493c17173b2eb591d52cf85f0e658084ffb4bfec9daf779f886f7042f23e819069b2957d64"


def test_authenticated_false(mocker):
    mocker.patch('extended.IBMExecutor.IBMQ.stored_account', return_value={})
    mocker.patch('extended.IBMExecutor.IBMExecutor.validate_token', return_value=True)
    ibm = IBMExecutor("token1", 100)
    assert ibm.authenticated == False
    mocker.resetall()


def test_authenticated_true(mocker):
    mocker.patch('extended.IBMExecutor.IBMQ.stored_account', return_value={"token1": 'cuenta1'})
    mocker.patch('extended.IBMExecutor.IBMExecutor.validate_token', return_value=True)
    ibm = IBMExecutor("token1", 100)
    assert ibm.authenticated == True
    mocker.resetall()


def test_validate_token_correct(mocker):
    mocker.patch('extended.IBMExecutor.IBMQ.stored_account', return_value={"token1": 'cuenta1'})
    mocker.patch('extended.IBMExecutor.IBMExecutor.log_in_IBMQ', return_value=True)
    ibm = IBMExecutor("token1", 100)
    assert ibm.validate_token("token1") == True
    mocker.resetall()


def test_validate_token_not_correct(mocker):
    mocker.patch('extended.IBMExecutor.IBMQ.stored_account', return_value={})
    mocker.patch('extended.IBMExecutor.IBMExecutor.log_in_IBMQ', return_value=True)
    try:
        ibm = IBMExecutor("token1", 100)
        assert ibm.validate_token("token1") == False
        assert False
    except:
        assert True
    mocker.resetall()


def test_log_in_IBMQ(mocker):
    mocker.patch('extended.IBMExecutor.IBMExecutor.validate_token', return_value=True)

    ibm = IBMExecutor(TOKEN, 1)
    ibm.log_in_IBMQ(TOKEN)
    assert ibm.authenticated == True

    assert ibm.log_in_IBMQ(TOKEN) is not None
    IBMQ.delete_account()
    mocker.resetall()


def test_create_backend_n_executions_less_than_0():
    ibm = IBMExecutor(TOKEN, -1)
    try:
        ibm.create_backend()
        assert False
    except:
        assert True


def test_create_backend_n_executions_more_than_20000():
    ibm = IBMExecutor(TOKEN, 20001)
    try:
        ibm.create_backend()
        assert False
    except:
        assert True


def test_create_backend_n_executions():
    ibm = IBMExecutor(TOKEN, 10)
    backend, quantum_instance = ibm.create_backend()

    assert backend.configuration().n_qubits >= 2
    assert backend.configuration().simulator == False
    assert backend.status().operational == True
    assert quantum_instance.run_config.shots == 10
