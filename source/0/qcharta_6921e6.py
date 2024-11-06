# https://github.com/JonasReichhardt/qCharta/blob/1251915d5a182375797863f60b0441de0b68f37c/src/qCharta.py
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout
from qiskit import QuantumRegister

import random

from sabre import Sabre
from coupling import distance_dict


class qCharta(TransformationPass):

    def __init__(self,
                 coupling_map, seed,  layout_option = 'trivial'):
        super().__init__()
        self.coupling_map = coupling_map
        self.seed = seed
        self.initial_mapping: Layout
        random.seed(seed)
        self.layout_option = layout_option

    def create_random_layout(self,dag):
        nr_qbits = len(self.coupling_map.physical_qubits)

        layout_arr = list(range(0,nr_qbits))
        random.shuffle(layout_arr)
        layout = Layout.from_intlist(layout_arr,*dag.qregs.values())

        return layout

    def create_heuristic_layout(self,dag):
        analysis = self.gate_analysis(dag)

        # place most used node in the center
        layout_dict = dict.fromkeys(range(0,len(self.coupling_map.physical_qubits)))
        layout_dict[31] = analysis[0][0]

        distance = 1
        position = 0
        for qbit in dag.qubits:
            candidates = distance_dict[distance]
            if layout_dict[candidates[position]] is None:
                layout_dict[candidates[position]] = qbit
                position = position+1
                if len(candidates) == position:
                    position = 0
                    distance = distance+1
                    if len(distance_dict.items()) == distance:
                        break

        return Layout(layout_dict)

    def gate_analysis(self, dag):
        # analyse the circuit to indentify the most used logical qbit 
        analysis = {}
        for gate in dag.two_qubit_ops():
            qbit1 = gate.qargs[0]
            qbit2 = gate.qargs[1]
            try:
                analysis[qbit1] = analysis[qbit1]+1
            except KeyError:
                analysis[qbit1] = 1
                
            try:
                analysis[qbit2] = analysis[qbit2]+1
            except KeyError:
                analysis[qbit2] = 1

        # sort qbits by usage in 2 qbit operations
        sorted_logical_qbits = sorted(analysis.items(), key=lambda x: x[1],reverse=True)

        return sorted_logical_qbits

    # not sure if this function will be needed 
    def hotspot_anaysis(self, dag, analysis):
        hot_qbit = max(analysis, key=analysis.get)
        hot_gates = {}
        for gate in dag.two_qubit_ops():
            if(gate.qargs[0].index == hot_qbit):
                try:
                    hot_gates[gate.qargs[1].index] = hot_gates[gate.qargs[1].index]+1
                except KeyError:
                    hot_gates[gate.qargs[1].index] = 1
            if(gate.qargs[1].index == hot_qbit):
                try:
                    hot_gates[gate.qargs[0].index] = hot_gates[gate.qargs[0].index]+1
                except KeyError:
                    hot_gates[gate.qargs[0].index] = 1
        return hot_qbit, hot_gates

    def run(self, dag):
        # filll up a "reserve" register
        reg = QuantumRegister(len(self.coupling_map.physical_qubits) - len(dag.qubits), 'r')
        dag.add_qreg(reg)

        if self.layout_option == 'trivial':
            init_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        elif self.layout_option == 'random':
            init_layout = self.create_random_layout(dag)
        elif self.layout_option == 'heuristic':
            init_layout = self.create_random_layout(dag)
            init_layout = self.create_heuristic_layout(dag)
        
        self.initial_layout = init_layout.copy()

        sabre = Sabre(self.coupling_map)
        return sabre.sabre_swap(dag.front_layer(), init_layout, dag, self.coupling_map)[0]