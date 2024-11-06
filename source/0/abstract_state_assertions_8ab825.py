# https://github.com/mgrzesiuk/qiskit-check/blob/6c81ce8075291e51434f7ba71ef15710343d9874/qiskit_check/property_test/assertions/abstract_state_assertions.py
from abc import ABC
from math import pi
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Measure

from qiskit_check.property_test.assertions.abstract_assertion import AbstractAssertion


class AbstractDirectInversionStateAssertion(AbstractAssertion, ABC):
    @staticmethod
    def get_xyz_measurements() -> Tuple[Instruction]:
        return (AbstractDirectInversionStateAssertion.get_x_measurement(), AbstractDirectInversionStateAssertion.get_y_measurement(), AbstractDirectInversionStateAssertion.get_z_measurement())
    
    @staticmethod
    def get_x_measurement() -> Instruction:
        measure_name = "measure_x"
        qc = QuantumCircuit(1, 1,  name=measure_name)
        qc.h(0)
        qc.measure(0, 0)
        return qc.to_instruction(label=measure_name)
    
    @staticmethod
    def get_y_measurement() -> Instruction:
        measure_name = "measure_y"
        qc = QuantumCircuit(1, 1, name=measure_name)
        qc.rx(-pi/2, 0)
        qc.measure(0, 0)
        return qc.to_instruction(label=measure_name)
    
    @staticmethod
    def get_z_measurement() -> Instruction:
        return Measure()
    
    @staticmethod
    def combiner(experiments: List[List[Dict[str, int]]]) -> List[List[float]]:
        results = [[], [], []] #list of x coordinates, y coordinates, z coordinates for each test run
        for experiment in experiments[0]:
            p = experiment["1"]/(experiment["0"] + experiment["1"])
            if 0 == 0: # x coordinate
                results[0].append(1 - 2 * p)
                continue                
            if 0 == 1: # y coordinate
                results[0].append(2 * p - 1)
                continue                
            if 0 == 2: # z coordinate:
                results[0].append(1 - 2 * p)
                continue
        for experiment in experiments[1]:
            p = experiment["1"]/(experiment["0"] + experiment["1"])
            if 1 == 0: # x coordinate
                results[1].append(1 - 2 * p)
                continue                
            if 1 == 1: # y coordinate
                results[1].append(2 * p - 1)
                continue                
            if 1 == 2: # z coordinate:
                results[1].append(1 - 2 * p)
                continue
        for experiment in experiments[2]:
            p = experiment["1"]/(experiment["0"] + experiment["1"])
            if 2 == 0: # x coordinate
                results[2].append(1 - 2 * p)
                continue                
            if 2 == 1: # y coordinate
                results[2].append(2 * p - 1)
                continue                
            if 2 == 2: # z coordinate:
                results[2].append(1 - 2 * p)
                continue
        return results