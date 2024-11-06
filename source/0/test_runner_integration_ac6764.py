# https://github.com/ogorodnikov/m1/blob/a2f259167767e0fcee49a2584f2dc7b67c24d956/tests/integration/test_runner_integration.py
import os
import pytest

from qiskit import QuantumCircuit

from core.runner import Runner
from core.dynamo import Dynamo

from test_runner import set_mocks
from test_runner import runner
from test_runner import test_task
from test_runner import get_test_task


def test_start_stop(runner):
    
    runner.start()
    assert os.environ.get('RUNNER_STATE') == 'Running'
    
    runner.stop()
    assert os.environ.get('RUNNER_STATE') == 'Stopped'


def test_execute_task(runner, monkeypatch, stub):
    
    monkeypatch.setattr(Runner, "monitor_job", stub)
    
    test_circuit = QuantumCircuit()
    test_backend = runner.simulator_backend
    
    runner.execute_task(task_id=None, circuit=test_circuit, backend=test_backend)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings("ignore::::69")
def test_plot_statevector_figure(runner, test_run_result):
    test_statevector = test_run_result.get_statevector(decimals=0)
    
    runner.plot_statevector_figure(statevector=test_statevector, task_id=None)
        

###   Fixtures   ###

@pytest.fixture(scope="module", autouse=True)
def set_mocks(mock, get_test_task, stub):
    
    mock(Dynamo, "__init__", stub)
    mock(Dynamo, "add_task", stub)
    mock(Dynamo, "add_status_update", stub)
    mock(Dynamo, "update_task_attribute", stub)
    mock(Dynamo, "move_figure_to_s3", stub)
    
    mock(Dynamo, "get_next_task", get_test_task)
    

@pytest.fixture
def runner(run_config):
    return Runner(db=Dynamo())
    
    
@pytest.fixture
def test_run_result():
    
    class TestRunResult:
        # def get_counts(self):
        #     return {'test_state': 0}
            
        def get_statevector(self, decimals):
            return [0, 0, 1, 0]
                
    return TestRunResult()