# https://github.com/mar-be/master-thesis-code/blob/79c102a2b685eb1c54ace12cb6df31f32d4a51ae/quantum_execution_job.py
from typing import Any, Dict, Optional
from qiskit import QuantumCircuit
from qiskit.result import Result
from uuid import uuid4
from enum import Enum



class Execution_Type(Enum):
    raw = 0
    aggregation = 1
    partition = 2

class QuantumExecutionJob():

    def __init__(self, circuit:QuantumCircuit, shots:int, type:Execution_Type=Execution_Type.raw, config:Dict={}, **kwargs) -> None:
        self.id = uuid4().hex
        self.circuit = circuit
        self.shots = shots
        self.type = type
        self.config = config
        self._result:Optional[Result] = None
        self.result_prob:Optional[Dict] = None
        self.__dict__.update(kwargs)

    @property
    def result(self) -> Optional[Result]:
        return self._result

    @result.setter
    def result(self, result:Optional[Result]):
        if result:
            self._result = result
            cnts = result.data()['counts']
            self.result_prob = dict({key:value/self.shots for key, value in cnts.items()})
        else:
            self._result = None


    

if __name__ == "__main__":
    vJob = QuantumExecutionJob(None, test = "test")

    print(vJob.test)
    vJob.hallo = 1234
    print(vJob.hallo)
    print(vJob.__dict__)