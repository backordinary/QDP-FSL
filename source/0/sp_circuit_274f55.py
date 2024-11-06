# https://github.com/r08222011/Qympy/blob/95b1674dbc106a9112f1e2116a00bca5b68119bd/src/qympy/quantum_circuit/sp_circuit.py
import qiskit
import sympy as sp
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger
from qympy.quantum_circuit import sp_gate

class Circuit:

    def __init__(self, num_qubits):
        '''Symbolic quantum circuit with Sympy'''
        # initialization
        self.num_qubits     = num_qubits
        self.initial_state  = sp.Matrix([1]+[0]*(2**num_qubits-1))
        self.final_state    = None
        # gates record
        self.gate_record    = []    # record of sequence of applied gates
        self.theta_set      = set() # set of strings of parameters 'theta'
        # qiskit plotting
        self.qiskit_circuit = qiskit.circuit.QuantumCircuit(num_qubits)
        self.qiskit_params  = {}

    def evolve(self):
        '''Evolve the circuit to compute the final quantum state (final_state)'''
        # initialize the quantum state
        self.final_state = self.initial_state
        # start computing matrices product (sequence of applied gates)
        gate_buffer = [sp.eye(2) for _ in range(self.num_qubits)]
        for i in range(len(self.gate_record)):
            if self.gate_record[i].gate_type == "SingleQubitGate":
                # postpone the tensor product of single-qubit gates until two-qubit gates appear
                wire = self.gate_record[i].wire
                gate_buffer[wire] = self.gate_record[i].matrix * gate_buffer[wire]
            elif self.gate_record[i].gate_type == "TwoQubitGate":
                # apply gate_buffer gates first then apply two-qubit gates
                wire1 = self.gate_record[i].wire1
                wire2 = self.gate_record[i].wire2
                submatrix = self.gate_record[i].get_submatrix(wire1, wire2)
                wire_min = min(wire1, wire2)
                wire_max = max(wire1, wire2)
                full_matrix = TensorProduct(sp.eye(2**wire_min), submatrix, sp.eye(2**(self.num_qubits-wire_max-1)))
                self.final_state = TensorProduct(*gate_buffer) * self.final_state
                self.final_state = full_matrix * self.final_state
                gate_buffer = [sp.eye(2) for _ in range(self.num_qubits)]
            else:
                raise TypeError("Unknown gate type : {self.gate_record[i].gate_type}")
        self.final_state = TensorProduct(*gate_buffer) * self.final_state

    def measure(self, qubit, basis):
        '''Single qubit measurement only'''
        # check basis and evolution of final_state
        assert basis in ["X", "Y", "Z"], f"Basis should be X or Y or Z, not {basis}"
        if self.final_state == None: self.evolve()
        # start measurement
        basis_dict = {
            "X": sp.Matrix([[0,1],[1,0],]),
            "Y": sp.Matrix([[0,-sp.I],[sp.I,0],]),
            "Z": sp.Matrix([[1,0],[0,-1],]),
        }
        full_matrix = TensorProduct(sp.eye(2**qubit), basis_dict[basis], sp.eye(2**(self.num_qubits-qubit-1)))
        return (Dagger(self.final_state) * full_matrix * self.final_state)[0]

    def draw(self, *args, **kwargs):
        '''Draw quantum circuit with Qiskit'''
        return self.qiskit_circuit.draw(*args, **kwargs)

    def __add__(self, other):
        '''Concatenate two quantum circuits, note that this method is not commutative'''
        new_circuit = Circuit(max(self.num_qubits, other.num_qubits))
        new_circuit.qiskit_circuit = self.qiskit_circuit.compose(other.qiskit_circuit)
        new_circuit.gate_record    = self.gate_record + other.gate_record
        new_circuit.theta_set      = self.theta_set.union(other.theta_set)
        # qiskit Parameters will show error if a Parameter has already created
        new_circuit.qiskit_params.update(self.qiskit_params)
        new_circuit.qiskit_params.update(other.qiskit_params)
        return new_circuit
    def __call__(self, x):
        '''Substitute inputs variables with x'''
        if self.final_state == None: self.evolve()
        subs = {}
        for i in range(len(x)):
            subs[sp.Symbol(f"inputs_{i}", real=True)] = x[i]
        self.final_state = self.final_state.evalf(subs=subs)
        return self
    def _check_wire(self, wire):
        '''Check whether the wire of a gate applied is valid'''
        num_qubits = self.num_qubits
        if wire >= num_qubits:
            raise ValueError(f"Wire {wire} is not between 0 to {num_qubits-1} with only {num_qubits} qubit(s).")
    def h(self, wire):
        self._check_wire(wire)
        self.qiskit_circuit.h(wire)
        self.gate_record.append(sp_gate.H(wire))
    def x(self, wire):
        self._check_wire(wire)
        self.qiskit_circuit.x(wire)
        self.gate_record.append(sp_gate.X(wire))
    def y(self, wire):
        self._check_wire(wire)
        self.qiskit_circuit.y(wire)
        self.gate_record.append(sp_gate.Y(wire))
    def z(self, wire):
        self._check_wire(wire)
        self.qiskit_circuit.z(wire)
        self.gate_record.append(sp_gate.Z(wire))
    def swap(self, wire1, wire2):
        self._check_wire(wire1)
        self._check_wire(wire2)
        self.qiskit_circuit.swap(wire1, wire2)
        self.gate_record.append(sp_gate.SWAP(wire1, wire2))
    def cx(self, wire1, wire2):
        # wire 1 is control qubit
        # wire 2 is target qubit
        self._check_wire(wire1)
        self._check_wire(wire2)
        self.qiskit_circuit.cx(wire1, wire2)
        self.gate_record.append(sp_gate.CX(wire1, wire2))
    def cz(self, wire1, wire2):
        # wire 1 is control qubit
        # wire 2 is target qubit
        self._check_wire(wire1)
        self._check_wire(wire2)
        self.qiskit_circuit.cz(wire1, wire2)
        self.gate_record.append(sp_gate.CZ(wire1, wire2))
    def _add_param_gate(self, gate_name, theta, wires):
        gate_qiskit = getattr(self.qiskit_circuit, gate_name)
        gate_spgate = getattr(sp_gate, gate_name.upper())
        if type(theta) == str:
            if theta in self.theta_set:
                gate_qiskit(self.qiskit_params[theta], *wires)
            else:
                qiskit_param = qiskit.circuit.Parameter(theta)
                gate_qiskit(qiskit_param, *wires)
                self.theta_set.add(theta)
                self.qiskit_params[theta] = qiskit_param
            self.gate_record.append(gate_spgate(sp.Symbol(theta, real=True), *wires))
        elif type(theta) in [int, float, complex]:
            gate_qiskit(theta, *wires)
            self.gate_record.append(gate_spgate(theta, *wires))
        else: # it would expected to be type from Sympy expressions
            gate_qiskit(0, *wires)
            self.gate_record.append(gate_spgate(theta, *wires))
    def rx(self, theta, wire):
        self._check_wire(wire)
        self._add_param_gate("rx", theta, wires=[wire])
    def ry(self, theta, wire):
        self._check_wire(wire)
        self._add_param_gate("ry", theta, wires=[wire])
    def rz(self, theta, wire):
        self._check_wire(wire)
        self._add_param_gate("rz", theta, wires=[wire])
    def rxx(self, theta, wire1, wire2):
        self._check_wire(wire1)
        self._check_wire(wire2)
        self._add_param_gate("rxx", theta, wires=[wire1, wire2])
    def ryy(self, theta, wire1, wire2):
        self._check_wire(wire1)
        self._check_wire(wire2)
        self._add_param_gate("ryy", theta, wires=[wire1, wire2])
    def rzz(self, theta, wire1, wire2):
        self._check_wire(wire1)
        self._check_wire(wire2)
        self._add_param_gate("rzz", theta, wires=[wire1, wire2])
    def crx(self, theta, wire1, wire2):
        self._check_wire(wire1)
        self._check_wire(wire2)
        self._add_param_gate("crx", theta, wires=[wire1, wire2])
    def cry(self, theta, wire1, wire2):
        self._check_wire(wire1)
        self._check_wire(wire2)
        self._add_param_gate("cry", theta, wires=[wire1, wire2])
    def crz(self, theta, wire1, wire2):
        self._check_wire(wire1)
        self._check_wire(wire2)
        self._add_param_gate("crz", theta, wires=[wire1, wire2])