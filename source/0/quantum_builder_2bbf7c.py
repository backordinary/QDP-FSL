# https://github.com/swapnil2me/chatlab/blob/fe887c25254f54401c94957945a1a844aee488da/quantum_builder.py
from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
import io

from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere

from config import cwq_style

gate_dict = {'h':{'alias':['h','hadamard'],
                  'num_gate':1
                },
             'not':{'alias':['n','negate'],
                    'num_gate':1
                    },
             'cnot':{'alias':['cnot','controlled not'],
                    'num_gate':2
                    },
             'tofolli':{'alias':['tofi'],
                    'num_gate':3
                    },
             'swap':{'alias':['swap'],
                    'num_gate':2
                    },
             'identity':{'alias':['i'],
                    'num_gate':1
                    },
             'reset':{'alias':['r','reset'],
                    'num_gate':1
                    },
             'measure':{'alias':['measure','m'],
                    'num_gate':2
                    }
            }

def confirm_gate_name(gate_name):
    if gate_name in gate_dict.keys():
        return gate_name
    else:
        for i in gate_dict.keys():
            if gate_name in gate_dict[i]["alias"]:
                gate_name = i
                break
    return gate_name


class Quantum_Circuit(object):
    """docstring for Quantum_Circuit."""

    def __init__(self, name="cwq"):
        self.name = name
        self.circut = None
        self.num_qbits = None
        self.num_cbits = None
        self.return_msg = None
        self.error_msg = None
        self.meas_ckt = None
        self.result = None

    def restore_default(self):
        self.circut = None
        self.return_msg = None
        self.error_msg = None
        self.meas_ckt = None
        self.result = None

    def _clear_error(self):
        self.error_msg = None

    def init_ckt(self, num_qbits=2, num_cbits=2):
        self.circut = QuantumCircuit(num_qbits,num_cbits)
        self.num_qbits = num_qbits
        self.num_cbits = num_cbits

    def init_measure_block(self):
        self._clear_error
        if self.num_qbits == self.num_cbits:
            temp = QuantumCircuit(self.num_qbits, self.num_cbits)
            temp.measure(list(range(self.num_qbits)), list(range(self.num_cbits)))
            self.meas_ckt = self.circut + temp
        else:
            self.error_msg = "Qunatum and classic bits must be equal in number for measurement"

    async def add_gate(self, gate_name, bit_list_from = None, bit_list_to = None, message = None):
        self._clear_error
        gate_name = confirm_gate_name(gate_name)
        if gate_dict.get(gate_name).get('num_gate') == 1:
            getattr(self.circut, gate_name)(bit_list_from)
            if message != None:
                print("Sending message to channel")
                await message.channel.send('Added {0} to the Qubit {1}'.format(gate_name.upper( ),  bit_list_from) )
        elif gate_dict.get(gate_name).get('num_gate') == 2:
            getattr(self.circut, gate_name)(bit_list_from, bit_list_to)
            if message != None:
                print("Sending message to channel")
                await message.channel.send('Added {0} from Qubit {1} to {2}'.format(gate_name.upper(), bit_list_from, bit_list_to))
        else:
            self.error_msg = "Gate not currently supported"

    def measure_classic(self):
        backend = BasicAer.get_backend('statevector_simulator')
        result = execute(self.meas_ckt, backend).result()
        self.result  = result.get_statevector(self.meas_ckt)

    def measure_q(self):
        backend = BasicAer.get_backend('statevector_simulator')
        result = execute(self.circut, backend).result()
        self.result  = result.get_statevector(self.circut)

    def qsphere(self,where_to):
        if where_to == "classic":
            self.measure_classic()
            fig = plot_state_qsphere(self.result)
        elif where_to == 'before_classic':
            self.measure_q()
            fig = plot_state_qsphere(self.result)
        fig.set_facecolor('white')
        output = io.BytesIO()
        return fig, output

    def multi_vector(self,where_to,title=None):
        if where_to == "classic":
            self.measure_classic()
            fig = plot_bloch_multivector(self.result, title=title)
        elif where_to == 'before_classic':
            self.measure_q()
            fig = plot_bloch_multivector(self.result, title=title)
        fig.set_facecolor('white')
        output = io.BytesIO()
        return fig, output

    def open_ckt(self):
        if self.meas_ckt == None:
            fig = self.circut.draw("mpl",style=cwq_style)
        else:
            fig = self.meas_ckt.draw("mpl",style=cwq_style)
        output = io.BytesIO()
        return fig, output
