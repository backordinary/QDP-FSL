# https://github.com/Alvoras/qtap/blob/be58e501467dd16706e121685ea7bc6808b623a0/lib/circuit/circuit.py
from lib.circuit.grid import CircuitGrid
from lib.circuit.grid_model import CircuitGridModel
from lib.utils.constants import MAX_COLUMNS, NUM_SHOTS, FRETS_COLOR_MAP
from lib.circuit.node_types import GATE_MAPPING
import lib.circuit.node_types as NODE_TYPES
from colorama import Style, Back

from math import ceil, degrees
import numpy as np
from qiskit import BasicAer, execute, ClassicalRegister

from copy import deepcopy


class Circuit:
    def __init__(self, qbit_qty, bar, height=20, width=20):
        self.height = height
        self.width = width
        self.qbit_qty = qbit_qty
        self.bar = bar

        self.circuit_grid_model = CircuitGridModel(self.qbit_qty, MAX_COLUMNS)
        self.circuit_grid = CircuitGrid(self.qbit_qty, MAX_COLUMNS, self.circuit_grid_model)

        self.grid_delta = 4
        self.y_padding = (self.height - (self.grid_delta * self.qbit_qty)) // 2

    def render(self, last_measured={}):
        lines = []

        for i in range(self.y_padding):
            lines.append(" " * self.width)

        measured = self.measure()
        probas = [0] * self.bar.tracks_qty

        for key, value in measured.items():
            probas[int(key, 2)] = int(value)

        lines += self.make_circuit()
        lines += self.make_proba(last_measured, probas)
        return lines

    def make_proba(self, last_measured, probas):
        lines = []
        line = []
        ref_symbols = []

        for idx, s in enumerate(self.bar.tracks_symbols):
            # Needed because of the color chars, not printed but present in the math
            # Used to calculate padding for centering
            ref_symbols.append(s)

            color = FRETS_COLOR_MAP[idx]
            symbol_proba = probas[idx]
            proba_bar = " "

            if symbol_proba == 0:
                s = "-"*self.qbit_qty
            else:
                if 10 <= symbol_proba < 20:
                    proba_bar = "▁"
                elif 20 <= symbol_proba < 30:
                    proba_bar = "▂"
                elif 30 <= symbol_proba < 40:
                    proba_bar = "▃"
                elif 40 <= symbol_proba < 50:
                    proba_bar = "▄"
                elif 50 <= symbol_proba < 60:
                    proba_bar = "▅"
                elif 60 <= symbol_proba < 70:
                    proba_bar = "▆"
                elif 70 <= symbol_proba < 80:
                    proba_bar = "▇"
                elif 80 <= symbol_proba < 90:
                    proba_bar = "█"
                elif symbol_proba >= 90:
                    proba_bar = "▉"

            last_measured_keys = list(last_measured.keys())
            if symbol_proba in last_measured_keys:
                line.append(f"{Back.WHITE}{Style.BRIGHT}{color}{s} {proba_bar}{Style.RESET_ALL}")
            else:
                line.append(f"{Style.BRIGHT}{color}{s} {proba_bar}{Style.RESET_ALL}")

        lines.append("  ".join(line))
        padding = ((self.width - len("   ".join(ref_symbols))) // 2) - 1

        # lines.append("   ".join(s for s in self.bar.tracks_symbols).center(self.width))
        lines.append((" " * self.qbit_qty).join(f"{str(p)}%".ljust(4) for p in probas))
        lines[0] = f"{' ' * padding}{lines[0]}"
        lines[1] = f"{' ' * padding}{lines[1]}"

        return lines

    def make_circuit(self):
        line_idx = ceil(self.grid_delta / 2)
        lines = []
        allowed_trace_connectors = [GATE_MAPPING[NODE_TYPES.CTRL_TOP_WIRE],
                                   GATE_MAPPING[NODE_TYPES.CTRL_BOTTOM_WIRE],
                                   GATE_MAPPING[NODE_TYPES.H],
                                   GATE_MAPPING[NODE_TYPES.NOT_GATE],
                                   GATE_MAPPING[NODE_TYPES.TRACE]]

        for wire in range(self.qbit_qty):
            for _ in range(self.grid_delta):
                lines.append(" " * self.width)

            offset = (wire * self.grid_delta) + line_idx
            lines[offset] = "─" * self.width

            for col in range(MAX_COLUMNS):
                c = self.render_gate(wire, col).center(5)

                left_padding = (col * (self.width // MAX_COLUMNS)) + (self.width // MAX_COLUMNS) // 2
                lines[offset] = lines[offset][:left_padding - 2] + c + lines[offset][left_padding + 3:]

                line_offset = offset - self.grid_delta // 2
                # Add a "│" on the previous line (same column) if the current is a ctrl-related node
                if wire > 0:
                    if GATE_MAPPING[NODE_TYPES.CTRL_TOP_WIRE] in c:
                        if wire > self.circuit_grid_model.get_gate_wire_for_control_node(wire, col):
                            lines[line_offset] = lines[line_offset][:left_padding] + GATE_MAPPING[NODE_TYPES.TRACE] + lines[line_offset][left_padding:]
                    elif GATE_MAPPING[NODE_TYPES.NOT_GATE] in c and self.render_gate(wire - 1, col) in allowed_trace_connectors:
                        lines[line_offset] = lines[line_offset][:left_padding] + GATE_MAPPING[NODE_TYPES.TRACE] + lines[line_offset][left_padding:]
                    elif GATE_MAPPING[NODE_TYPES.TRACE] in c:
                        lines[line_offset] = lines[line_offset][:left_padding] + GATE_MAPPING[NODE_TYPES.TRACE] + lines[line_offset][left_padding:]

        lines.append(" " * self.width)

        lines = self.draw_cursor(lines)

        return lines

    def draw_cursor(self, lines):
        line_idx = ceil(self.grid_delta / 2)
        offset = (self.circuit_grid.selected_wire * self.grid_delta) + line_idx
        left_padding = (self.circuit_grid.selected_column * (self.width // MAX_COLUMNS)) + (self.width // MAX_COLUMNS) // 2

        gate = self.render_gate(self.circuit_grid.selected_wire, self.circuit_grid.selected_column)
        cursor = [
            "╭─ ─╮",
            f"{gate.center(5)}",
            "╰─ ─╯"
        ]
        cursor.reverse()

        for idx, line in enumerate(cursor):
            cursor_line_size = len(cursor[idx]) // 2
            cursor_height_size = len(cursor) // 2
            cursor_offset_y = offset - (idx - cursor_height_size)

            lines[cursor_offset_y] = lines[cursor_offset_y][:left_padding - cursor_line_size] + \
                cursor[idx] + lines[cursor_offset_y][left_padding + cursor_line_size + 1:]

        return lines

    def render_rotated_gate(self, rads, node_type):
        c = GATE_MAPPING[node_type]
        c += " "
        deg = int(degrees(rads % (2 * np.pi)))
        if deg < 90:
            c += "◯"
        elif 90 <= deg < 180:
            c += "◔"
        elif 180 <= deg < 270:
            c += "◑"
        elif 270 <= deg < 360:
            c += "◕"

        return c

    def render_gate(self, wire, col):
        c = "▅"
        node = self.circuit_grid_model.get_node(wire, col)
        computed_type = self.circuit_grid_model.get_node_gate_part(wire, col)

        if not node:
            return c
        if computed_type == NODE_TYPES.H:
            c = GATE_MAPPING[node.node_type]
        elif computed_type == NODE_TYPES.X:
            if node.ctrl_a >= 0 or node.ctrl_b >= 0:
                # This is a control-X gate or Toffoli gate
                # TODO: Handle Toffoli gates more completely
                if wire > max(node.ctrl_a, node.ctrl_b):
                    if node.radians != 0:
                        c = self.render_rotated_gate(node.radians, NODE_TYPES.NOT_GATE)
                    else:
                        c = GATE_MAPPING[NODE_TYPES.NOT_GATE]
                else:
                    if node.radians != 0:
                        c = self.render_rotated_gate(node.radians, NODE_TYPES.NOT_GATE)
                    else:
                        c = GATE_MAPPING[NODE_TYPES.NOT_GATE]
            elif node.radians != 0:
                c = self.render_rotated_gate(node.radians, node.node_type)
            else:
                c = GATE_MAPPING[NODE_TYPES.X]
        elif computed_type == NODE_TYPES.Y:
            if node.radians != 0:
                c = self.render_rotated_gate(node.radians, node.node_type)
            else:
                c = GATE_MAPPING[node.node_type]
        elif computed_type == NODE_TYPES.Z:
            if node.radians != 0:
                c = self.render_rotated_gate(node.radians, node.node_type)
            else:
                c = GATE_MAPPING[node.node_type]
        elif computed_type == NODE_TYPES.CTRL:
            if wire > \
                    self.circuit_grid_model.get_gate_wire_for_control_node(wire, col):
                c = GATE_MAPPING[NODE_TYPES.CTRL_BOTTOM_WIRE]
            else:
                c = GATE_MAPPING[NODE_TYPES.CTRL_TOP_WIRE]
        else:
            try:
                c = GATE_MAPPING[node.node_type]
            except KeyError:
                pass

        return c

    def predict(self):
        circuit = self.circuit_grid_model.compute_circuit()

        backend_sv_sim = BasicAer.get_backend('statevector_simulator')
        job_sim = execute(circuit, backend_sv_sim, shots=NUM_SHOTS)
        result_sim = job_sim.result()
        quantum_state = result_sim.get_statevector(circuit, decimals=3)

        return quantum_state

    def measure(self, reset=False):
        circuit = self.circuit_grid_model.compute_circuit()

        backend_sv_sim = BasicAer.get_backend('qasm_simulator')
        cr = ClassicalRegister(self.qbit_qty)
        measure_circuit = deepcopy(circuit)  # make a copy of circuit
        measure_circuit.add_register(cr)  # add classical registers for measurement readout
        measure_circuit.measure(measure_circuit.qregs[0], measure_circuit.cregs[0])
        job_sim = execute(measure_circuit, backend_sv_sim, shots=NUM_SHOTS)
        result_sim = job_sim.result()
        counts = result_sim.get_counts(circuit)

        if reset:
            self.circuit_grid_model.reset_circuit()

        return counts
