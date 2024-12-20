# https://github.com/peachnuts/Multiprogramming/blob/c539b1b8d4739546909caf813f6aa8d26b6435c2/src/hardware/IBMQHardwareArchitecture.py
# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (02/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
#               Siyuan Niu (<siyuan.niu@lirmm.fr>)
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================

import logging
import typing as ty
import pickle
from pathlib import Path
import networkx as nx
import numpy as np
from hardware.HardwareArchitecture import HardwareArchitecture
from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit.dagcircuit import DAGNode

logger = logging.getLogger("hardware.IBMQHardwareArchitecture")

def cnot_error_rate_function(vertex, hardware):
    source, sink = vertex
    return hardware.get_link_error_rate(source, sink)

def swap_number(*_) -> float:
    return 1.0

def readout_error_rate_function(qubit_index, hardware):
    return hardware.get_qubit_readout_error(qubit_index)

class IBMQHardwareArchitecture(HardwareArchitecture):
    _hardware_directory: Path = Path(
        __file__
    ).parent/ "architectures_saved_data"

    @staticmethod
    def _get_value(value: float, unit: str):
        # We want time in nanoseconds
        if unit == "us":
            return value * 10 ** 3
        elif unit == "ns":
            return value

        # We want everything in Hertz
        elif unit == "GHz":
            return value * 10 ** 9
        elif unit == "":
            return value
        else:
            logger.error(f"Unsupported unit: '{unit}'")
            exit(1)

    @staticmethod
    def _get_link_properties_dict(backend_properties, source: int, sink: int):
        for gate in backend_properties.gates:
            if gate.gate != "cx":
                continue
            if len(gate.qubits) != 2:
                continue
            if gate.qubits[0] == source and gate.qubits[1] == sink:
                # Return the dictionary
                properties = {
                    param.name: IBMQHardwareArchitecture._get_value(
                        param.value, param.unit
                    )
                    for param in gate.parameters
                }
                return properties

    @staticmethod
    def _get_qubit_properties_dict(qubit_index, qubit_properties, gates_properties):
        properties = dict()
        for prop in qubit_properties[qubit_index]:
            properties[prop.name] = IBMQHardwareArchitecture._get_value(
                prop.value, prop.unit
            )
        for gate in gates_properties:
            # Filter the gates that are not interesting
            if len(gate.qubits) > 1:
                continue
            if gate.qubits[0] != qubit_index:
                continue
            properties[gate.gate] = {
                param.name: IBMQHardwareArchitecture._get_value(param.value, param.unit)
                for param in gate.parameters
            }
        return properties

    @staticmethod
    def _get_backend(backend_name: str):
        from qiskit import IBMQ
        from qiskit.providers.ibmq.exceptions import (
            IBMQAccountCredentialsNotFound,
            IBMQAccountMultipleCredentialsFound,
            IBMQAccountCredentialsInvalidUrl,
        )
        logger.info("Loading IBMQ account.")
        try:
            IBMQ.load_account()
            provider = IBMQ.get_provider(
                hub='ibm-q-france', group='univ-montpellier', project='default'
            )
        except(
                IBMQAccountMultipleCredentialsFound,
                IBMQAccountCredentialsNotFound,
                IBMQAccountCredentialsInvalidUrl,
        ):
            logger.error(
                "WARNING: No valid IBMQ credentials found on disk.\n"
                "You must store your credentials using IBMQ.save_account(token, url).\n"
                "For now, there's only access to local simulator backends..."
            )
            exit(1)
        logger.info("Connected to IBMQ account!")
        logger.info(f"Getting backend '{backend_name}'.")
        matching_backends = provider.backends(backend_name)
        if not matching_backends:
            logger.error(f"No backend matching the search '{backend_name}'.")
            exit(1)
        backend = matching_backends[0]
        return backend

    def __init__(
            self,
            backend_name: str,
            weight_func: ty.Callable[
                [ty.Tuple[int, int], nx.classes.reportviews.OutEdgeView], float
            ] = None,
            incoming_graph_data=None,
            **kwargs,
    ):
        """
        The architecute of any IBMQ hardware
        :param backend_name: The name of IBMQ hardware
        :param weight_func: a function taking an edge identifier and all the edges of
            the architecture in parameters and that return the cost associated to this
            edge. If None, the function returns the execution time of the CNOT gate on
            the given link.
        :param incoming_graph_data: forwarded to :py:method:`networkx.Digraph.__init__`.
        :param kwargs: forwarded to :py:method:`networkx.Digraph.__init__`.
        """
        super().__init__(incoming_graph_data, **kwargs)
        self._ignored_gates = {"barrier"}
        if weight_func is None:
            # Default to the cost of a CNOT
            weight_func = swap_number

        self._weight_func = weight_func
        self.name = backend_name
        backend = IBMQHardwareArchitecture._get_backend(backend_name)
        backend_configuration = backend.configuration()
        qubit_number = backend_configuration.n_qubits

        self._coupling_graph = backend_configuration.coupling_map

        backend_properties = backend.properties()
        qubit_properties = backend_properties.qubits
        gate_properties = backend_properties.gates
        qubit_indices = list()
        # Add the qubits with their properties (error rates?)
        for qubit_index in range(qubit_number):
            added_qubit_index = self.add_qubit(
                **IBMQHardwareArchitecture._get_qubit_properties_dict(
                    qubit_index, qubit_properties, gate_properties
                )
            )
            qubit_indices.append(added_qubit_index)
        # Add the links between qubits
        for source, sink in self._coupling_graph:
            self.add_link(
                source,
                sink,
                **IBMQHardwareArchitecture._get_link_properties_dict(
                    backend_properties, source, sink
                ),
            )
        # Update the links with a default function
        self.update_link_weights()


    @property
    def hardware_coupling_graph(self):
        return self._coupling_graph


    def get_link_error_rate(self, source: int, sink: int) -> float:
        return self.edges[source, sink]['gate_error']

    def get_qubit_readout_error(self, qubit_index: int) -> float:
        return self.nodes[qubit_index]['readout_error']

    def get_single_qubit_error(self, qubit_index: int) -> float:
        return (self.nodes[qubit_index]['id']['gate_error']
               + self.nodes[qubit_index]['u1']['gate_error']
               + self.nodes[qubit_index]['u2']['gate_error'] +
               + self.nodes[qubit_index]['u3']['gate_error']) / 4

    def update_link_weights(self, distance_matrix: np.ndarray=None):
        """
        Updates the weight on each qubit link
        """
        weight_attributes = dict()
        if distance_matrix is None:
            for edge in self.edges:
                weight_attributes[edge] = self._weight_func(edge,self)
        else:
            for edge in self.edges:
                i = edge[0]
                j = edge[1]
                weight_attributes[edge] = distance_matrix.item(i,j)

        nx.set_edge_attributes(self, weight_attributes, name="weight")

    @property
    def weight_function(self):
        return self._weight_func

    @weight_function.setter
    def weight_function(self, value):
        self._weight_func = value
        self.update_link_weights()

    def get_link_execution_time(self, source: int, sink: int) -> float:
        """Returns the execution time of the CNOT gate between source and sink in \
        nano-seconds."""
        return self.edges[source, sink]["gate_length"]

    def is_ignored_operation(self, op: DAGNode) -> bool:
        return op.name in self._ignored_gates

    def can_natively_execute_operation(
        self, op: DAGNode, current_mapping: ty.Dict[Qubit, int]
    ) -> bool:
        if self.is_ignored_operation(op):
            return True

        if len(op.qargs) == 1:
            # If this is a 1-qubit operation, then the hardware can always execute it
            # natively.
            return True
        elif len(op.qargs) == 2:
            # q1, q2 = initial_mapping[op.qargs[0]], initial_mapping[op.qargs[1]]
            # inverse_mapping = {val: key for key, val in current_mapping.items()}
            # control, target = inverse_mapping[q1], inverse_mapping[q2]
            source, sink = current_mapping[op.qargs[0]], current_mapping[op.qargs[1]]
            # source = mapping[op.qargs[0]]
            # sink = mapping[op.qargs[1]]
            return (source, sink) in self.edges
        else:
            logger.error(
                f"Found invalid operation acting on {len(op.qargs)} qubits. "
                f"Ignoring the operation {op.name} and exiting."
            )
            exit(1)

    def save(self, hardware_name: str):
        filepath = (
            IBMQHardwareArchitecture._hardware_directory / f"{hardware_name}.archdata"
        )
        with open(str(filepath), "wb") as f:
            logger.info(f"Saving IBMQHardwareArchitecture instance in '{filepath}'.")
            pickle.dump(self, f)

    @staticmethod
    def load(hardware_name: str) -> "IBMQHardwareArchitecture":
        filepath = (
            IBMQHardwareArchitecture._hardware_directory / f"{hardware_name}.archdata"
        )
        with open(str(filepath), "rb") as f:
            logger.info(f"Loading IBMQHardwareArchitecture instance from '{filepath}'.")
            return pickle.load(f)











