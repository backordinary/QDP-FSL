# https://github.com/radumarg/uranium-quantum/blob/2cd84047f3c1b238939db883857ee5ad6cf425bf/uranium_quantum/circuit_exporter/qiskit-exporter.py
import importlib
import numpy as np

BaseExporter = importlib.import_module("uranium_quantum.circuit_exporter.base-exporter")
class Exporter(BaseExporter.BaseExporter):
    def imports_and_or_headers_section(self):
        return f"\
import numpy as np\n\
from qiskit import QuantumRegister\n\
from qiskit.circuit import ClassicalRegister\n\
from qiskit import QuantumCircuit\n\
\n\
from qiskit.circuit.library.standard_gates import XGate, YGate, ZGate, HGate\n\
from qiskit.circuit.library import RXGate, RYGate, RZGate\n\
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate\n\
from qiskit.circuit.library import RZXGate\n\
from qiskit.circuit.library import SXGate, SXdgGate\n\
from qiskit.circuit.library import SGate, SdgGate, TGate, TdgGate\n\
from qiskit.circuit.library import UGate, U1Gate\n\
from qiskit.circuit.library import SwapGate, iSwapGate\n\
from qiskit.circuit.library import QFT\n\
from uranium_quantum.circuit_exporter.qiskit_custom_gates import *\n\
\n\n"
    def start_circuit_code(self, circuit_name):
        if self._bits:
            return f"\
cr_{circuit_name} = ClassicalRegister({self._bits})\n\
qr_{circuit_name} = QuantumRegister({self._qubits})\n\
qc_{circuit_name} = QuantumCircuit(qr_{circuit_name}, cr_{circuit_name})\n\n\n"
        else:
            return f"\
qr_{circuit_name} = QuantumRegister({self._qubits})\n\
qc_{circuit_name} = QuantumCircuit(qr_{circuit_name})\n\n\n"


    def end_circuit_code(self):
        return f""

    @staticmethod
    def get_control_states(controls):
        controlstates = ""
        for control in controls:
            if str(control['state']) in ["0", "+", "+i"]:
                controlstates += "0"
            else:
                controlstates += "1"
        # in qiskit higher qubit indices are more significant
        # while we order controls by target index in increasing order
        controlstates = controlstates[::-1]
        return controlstates

    @staticmethod
    def get_circuit_power(power):
      sign = 1
      if power[0] == '-':
          sign = -1
          power = power[1:]
      if '^' in power:
        return sign * (2 ** int(power[2:]))
      else:
        return sign * int(power)

    @staticmethod
    def get_plain_gate(name, targets, label, theta_radians=None, phi_radians=None, lambda_radians=None, root=None):
        params = ""
        if theta_radians != None:
          params += f"{theta_radians}"
        if phi_radians != None:
          if params:
            params += f", {phi_radians}"
          else:
            params += f"{phi_radians}"
        if lambda_radians != None:
          if params:
            params += f", {lambda_radians}"
          else:
            params += f"{lambda_radians}"
        if root != None:
          if params:
            params += f", {root}"
          else:
            params += f"{root}"

        if name == "QFT":
            params = len(targets)

        if params:
            if label:
                return f"{name}({params}, label='{label}')"
            else:
                return f"{name}({params})"
        else:
            if label:
                return f"{name}(label='{label}')"
            else:
                return f"{name}()"

    @staticmethod
    def get_controlled_gate(name, controls, targets, label, theta_radians=None, phi_radians=None, lambda_radians=None, root=None):
        controlstates = Exporter.get_control_states(controls)
        params = ""
        if theta_radians != None:
            params += f"{theta_radians}"
        if phi_radians != None:
            if params:
                params += f", {phi_radians}"
            else:
                params += f"{phi_radians}"
        if lambda_radians != None:
            if params:
                params += f", {lambda_radians}"
            else:
                params += f"{lambda_radians}"
        if root != None:
            if params:
                params += f", {root}"
            else:
                params += f"{root}"

        if name == "QFT":
            params = len(targets)

        if label:
            return f"{name}({params}).control(num_ctrl_qubits={len(controls)}, ctrl_state='{controlstates}', label='{label}')"
        else:
            return f"{name}({params}).control(num_ctrl_qubits={len(controls)}, ctrl_state='{controlstates}')"


    @staticmethod
    def rotate_state_to_x_basis(circuit_name, target):
        return f"qc_{circuit_name}.h(qr_{circuit_name}[{target}])\n"

    @staticmethod
    def rotate_state_to_y_basis(circuit_name, target):
        return f"qc_{circuit_name}.unitary(gate_rotation_to_y_basis(), [{target}])\n"

    @staticmethod
    def undo_rotate_state_to_x_basis(circuit_name, target):
        return f"qc_{circuit_name}.h(qr_{circuit_name}[{target}])\n"

    @staticmethod
    def undo_rotate_state_to_y_basis(circuit_name, target):
        return f"qc_{circuit_name}.unitary(gate_undo_rotation_to_y_basis(), [{target}])\n"

    @staticmethod
    def controlled_gate_code(name, circuit_name, controls, targets, label=None, theta_radians=None, phi_radians=None, root=None, lambda_radians=None, inverse=False, power=1):
        assert isinstance(circuit_name, str)
        assert isinstance(controls, list)
        assert isinstance(targets, list)
        if inverse:
            controlled_gate = Exporter.get_controlled_gate(name, controls, targets, label, theta_radians, phi_radians, lambda_radians, root) + ".inverse()"
        else:
            controlled_gate = Exporter.get_controlled_gate(name, controls, targets, label, theta_radians, phi_radians, lambda_radians, root)

        qubits = ""
        for control in controls:
            if qubits:
                qubits += ", "
            qubits += f"qr_{circuit_name}[{control['target']}]"
        for target in targets:
            if qubits:
                qubits += ", "
            qubits += f"qr_{circuit_name}[{target}]"

        code = ""
        for control in controls:
            if '+i' in control['state'] or '-i' in control['state']:
                code += Exporter.rotate_state_to_y_basis(circuit_name, control['target'])
            elif '+' in control['state'] or '-' in control['state']:
                code += Exporter.rotate_state_to_x_basis(circuit_name, control['target'])

        for _ in range(abs(power)):
            code += f"qc_{circuit_name}.append({controlled_gate}, [{qubits}])\n"

        for control in controls:
            if '+i' in control['state'] or '-i' in control['state']:
                code += Exporter.undo_rotate_state_to_y_basis(circuit_name, control['target'])
            elif '+' in control['state'] or '-' in control['state']:
                code += Exporter.undo_rotate_state_to_x_basis(circuit_name, control['target'])

        return code


    @staticmethod
    def plain_gate_code(name, circuit_name, targets, label=None, theta_radians=None, phi_radians=None, root=None, lambda_radians=None, inverse=False, power=1):
        assert isinstance(circuit_name, str)
        assert isinstance(targets, list)
        if inverse:
          plain_gate = Exporter.get_plain_gate(name, targets, label, theta_radians, phi_radians, lambda_radians, root) + ".inverse()"
        else:
          plain_gate = Exporter.get_plain_gate(name, targets, label, theta_radians, phi_radians, lambda_radians, root)
        qubits = ""
        for target in targets:
            if qubits:
                qubits += ", "
            qubits += f"qr_{circuit_name}[{target}]"
        out = ""
        for _ in range(abs(power)):
            out += f"qc_{circuit_name}.append({plain_gate}, [{qubits}])\n"
        return out

    @staticmethod
    def _gate_u3(
        circuit_name, controls, targets, theta_radians, phi_radians, lambda_radians, add_comments=True
    ):
        out = "# u3 gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('UGate', circuit_name, controls, targets, theta_radians=theta_radians, phi_radians=phi_radians, lambda_radians=lambda_radians)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.u({theta_radians}, {phi_radians}, {lambda_radians}, qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_u2(circuit_name, controls, targets, phi_radians, lambda_radians, add_comments=True):
        out = "# u2 gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('UGate', circuit_name, controls, targets, theta_radians=(np.pi/2), phi_radians=phi_radians, lambda_radians=lambda_radians)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.u(np.pi/2, {phi_radians}, {lambda_radians}, qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_u1(circuit_name, controls, targets, lambda_radians, add_comments=True):
        out = "# u1 gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('U1Gate', circuit_name, controls, targets, lambda_radians=lambda_radians)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.p({lambda_radians}, qr_{circuit_name}[{targets[0]}])\n"
        return out


    @staticmethod
    def _gate_identity(circuit_name, targets, add_comments=True):
        out = "# identity gate\n" if add_comments else ""
        out += f"qc_{circuit_name}.id(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_hadamard(circuit_name, controls, targets, add_comments=True):
        out = "# hadamard gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('HGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.h(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_hadamard_xy(circuit_name, controls, targets, add_comments=True):
        out = "# hadamard-xy gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('hadamard_xy', circuit_name, controls, targets, label='hadamard-xy')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(hadamard_xy(), [{targets[0]}], label='hadamard-xy')\n"
        return out

    @staticmethod
    def _gate_hadamard_yz(circuit_name, controls, targets, add_comments=True):
        out = "# hadamard-yz gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('hadamard_yz', circuit_name, controls, targets, label='hadamard-yz')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(hadamard_yz(), [{targets[0]}], label='hadamard-yz')\n"
        return out

    @staticmethod
    def _gate_hadamard_zx(circuit_name, controls, targets, add_comments=True):
        out = "# hadamard-zx gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('HGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.h(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_pauli_x(circuit_name, controls, targets, add_comments=True):
        out = "# pauli-x gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('XGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.x(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_pauli_y(circuit_name, controls, targets, add_comments=True):
        out = "# pauli-y gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('YGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.y(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_pauli_z(circuit_name, controls, targets, add_comments=True):
        out = "# pauli-z gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('ZGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.z(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_pauli_x_root(circuit_name, controls, targets, root, add_comments=True):
        out = "# pauli-x-root gate\n" if add_comments else ""
        root = f"(2**{root[4:]})" if '^' in root else root[2:]
        if controls:
            code = Exporter.controlled_gate_code('pauli_x_root', circuit_name, controls, targets, root=root, label='pauli-x-root')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(pauli_x_root({root}), [{targets[0]}], label='pauli-x-root')\n"
        return out

    @staticmethod
    def _gate_pauli_y_root(circuit_name, controls, targets, root, add_comments=True):
        out = "# pauli-y-root gate\n" if add_comments else ""
        root = f"(2**{root[4:]})" if '^' in root else root[2:]
        if controls:
            code = Exporter.controlled_gate_code('pauli_y_root', circuit_name, controls, targets, root=root, label='pauli-y-root')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(pauli_y_root({root}), [{targets[0]}], label='pauli-y-root')\n"
        return out

    @staticmethod
    def _gate_pauli_z_root(circuit_name, controls, targets, root, add_comments=True):
        out = "# pauli-z-root gate\n" if add_comments else ""
        root = f"(2**{root[4:]})" if '^' in root else root[2:]
        if controls:
            code = Exporter.controlled_gate_code('pauli_z_root', circuit_name, controls, targets, root=root, label='pauli-z-root')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(pauli_z_root({root}), [{targets[0]}], label='pauli-z-root')\n"
        return out

    @staticmethod
    def _gate_pauli_x_root_dagger(circuit_name, controls, targets, root, add_comments=True):
        out = "# pauli-x-root-dagger gate\n" if add_comments else ""
        root = f"(2**{root[4:]})" if '^' in root else root[2:]
        if controls:
            code = Exporter.controlled_gate_code('pauli_x_root_dagger', circuit_name, controls, targets, root=root, label='pauli-x-root-dagger')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(pauli_x_root_dagger({root}), [{targets[0]}], label='pauli-x-root-dagger')\n"
        return out

    @staticmethod
    def _gate_pauli_y_root_dagger(circuit_name, controls, targets, root, add_comments=True):
        out = "# pauli-y-root-dagger gate\n" if add_comments else ""
        root = f"(2**{root[4:]})" if '^' in root else root[2:]
        if controls:
            code = Exporter.controlled_gate_code('pauli_y_root_dagger', circuit_name, controls, targets, root=root, label='pauli-y-root-dagger')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(pauli_y_root_dagger({root}), [{targets[0]}], label='pauli-y-root-dagger')\n"
        return out

    @staticmethod
    def _gate_pauli_z_root_dagger(circuit_name, controls, targets, root, add_comments=True):
        out = "# pauli-z-root-dagger gate\n" if add_comments else ""
        root = f"(2**{root[4:]})" if '^' in root else root[2:]
        if controls:
            code = Exporter.controlled_gate_code('pauli_z_root_dagger', circuit_name, controls, targets, root=root, label='pauli-z-root-dagger')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(pauli_z_root_dagger({root}), [{targets[0]}], label='pauli-z-root-dagger')\n"
        return out

    @staticmethod
    def _gate_t(circuit_name, controls, targets, add_comments=True):
        out = "# t gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('TGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.t(qr_{circuit_name}[{targets[0]}])\n"
        return out


    @staticmethod
    def _gate_t_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# t-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('TdgGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.tdg(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_rx_theta(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# rx-theta gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('RXGate', circuit_name, controls, targets, theta_radians=theta_radians)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.rx({theta_radians}, qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_ry_theta(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# ry-theta gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('RYGate', circuit_name, controls, targets, theta_radians=theta_radians)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.ry({theta_radians}, qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_rz_theta(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# rz-theta gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('RZGate', circuit_name, controls, targets, theta_radians=theta_radians)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.rz({theta_radians}, qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_v(circuit_name, controls, targets, add_comments=True):
        out = "# v gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('SXGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('SXGate', circuit_name, targets)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_v_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# v-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('SXdgGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('SXdgGate', circuit_name, targets)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_h(circuit_name, controls, targets, add_comments=True):
        out = "# h gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('h', circuit_name, controls, targets, label='h')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(h(), [{targets[0]}], label='h')\n"
        return out

    @staticmethod
    def _gate_h_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# h-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('h_dagger', circuit_name, controls, targets, label='h-dagger')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(h_dagger(), [{targets[0]}], label='h-dagger')\n"
        return out

    @staticmethod
    def _gate_c(circuit_name, controls, targets, add_comments=True):
        out = "# c gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('c', circuit_name, controls, targets, label='c')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(c(), [{targets[0]}], label='c')\n"
        return out

    @staticmethod
    def _gate_c_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# c-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('c_dagger', circuit_name, controls, targets, label='c-dagger')
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.unitary(c_dagger(), [{targets[0]}], label='c-dagger')\n"
        return out


    @staticmethod
    def _gate_p(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# p gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('U1Gate', circuit_name, controls, targets, theta_radians=theta_radians)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.p({theta_radians}, qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_s(circuit_name, controls, targets, add_comments=True):
        out = "# s gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('SGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.s(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_s_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# s-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('SdgGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.sdg(qr_{circuit_name}[{targets[0]}])\n"
        return out

    @staticmethod
    def _gate_swap(circuit_name, controls, targets, add_comments=True):
        out = "# swap gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('SwapGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.swap(qr_{circuit_name}[{targets[0]}], qr_{circuit_name}[{targets[1]}])\n"
        return out

    @staticmethod
    def _gate_swap_root(circuit_name, controls, targets, root, add_comments=True):
        out = "# swap root gate\n" if add_comments else ""
        root = f"(2**{root[4:]})" if '^' in root else root[2:]
        if controls:
            code = Exporter.controlled_gate_code('swap_root', circuit_name, controls, targets, root=root, label="swap-root")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('swap_root', circuit_name, targets, root=root, label="swap-root")
            out += f"{code}\n"
        return out


    @staticmethod
    def _gate_swap_root_dagger(circuit_name, controls, targets, root, add_comments=True):
        out = "# swap root dagger gate\n" if add_comments else ""
        root = f"(2**{root[4:]})" if '^' in root else root[2:]
        if controls:
            code = Exporter.controlled_gate_code('swap_root_dagger', circuit_name, controls, targets, root=root, label="swap-root-dagger")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('swap_root_dagger', circuit_name, targets, root=root, label="swap-root-dagger")
            out += f"{code}\n"
        return out

  
    @staticmethod
    def _gate_iswap(circuit_name, controls, targets, add_comments=True):
        out = "# iswap gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('iSwapGate', circuit_name, controls, targets)
            out += f"{code}"
        else:
            out += f"qc_{circuit_name}.iswap(qr_{circuit_name}[{targets[0]}], qr_{circuit_name}[{targets[1]}])\n"
        return out

    @staticmethod
    def _gate_fswap(circuit_name, controls, targets, add_comments=True):
        out = "# fswap gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('fswap', circuit_name, controls, targets, label="fswap")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('fswap', circuit_name, targets, label="fswap")
            out += f"{code}\n"
        return out


    @staticmethod
    def _gate_swap_theta(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# swap theta gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('swap_theta', circuit_name, controls, targets, theta_radians=theta_radians, label="swap-theta")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('swap_theta', circuit_name, targets, theta_radians=theta_radians, label="swap-theta")
            out += f"{code}\n"
        return out


    @staticmethod
    def _gate_sqrt_swap(circuit_name, controls, targets, add_comments=True):
        out = "# sqrt-swap gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('sqrt_swap', circuit_name, controls, targets, label="sqrt-swap")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('sqrt_swap', circuit_name, targets, label="sqrt-swap")
            out += f"{code}\n"
        return out


    @staticmethod
    def _gate_sqrt_swap_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# sqrt-swap-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('sqrt_swap_dagger', circuit_name, controls, targets, label="sqrt-swap-dagger")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('sqrt_swap_dagger', circuit_name, targets, label="sqrt-swap-dagger")
            out += f"{code}\n"
        return out


    @staticmethod
    def _gate_xx(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# xx gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('RXXGate', circuit_name, controls, targets, theta_radians=theta_radians)
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('RXXGate', circuit_name, targets, theta_radians=theta_radians)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_yy(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# yy gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('RYYGate', circuit_name, controls, targets, theta_radians=theta_radians)
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('RYYGate', circuit_name, targets, theta_radians=theta_radians)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_zz(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# zz gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('RZZGate', circuit_name, controls, targets, theta_radians=theta_radians)
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('RZZGate', circuit_name, targets, theta_radians=theta_radians)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_xy(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# xy gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('xy', circuit_name, controls, targets, theta_radians=theta_radians, label="xy")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('xy', circuit_name, targets, theta_radians=theta_radians, label="xy")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_givens(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# givens gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('givens', circuit_name, controls, targets, theta_radians=theta_radians, label="givens")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('givens', circuit_name, targets, theta_radians=theta_radians, label="givens")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_a(circuit_name, controls, targets, theta_radians, phi_radians, add_comments=True):
        out = "# a gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('a', circuit_name, controls, targets, theta_radians=theta_radians, phi_radians=phi_radians, label="a")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('a', circuit_name, targets, theta_radians=theta_radians, phi_radians=phi_radians, label="a")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_cross_resonance(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# crosss-resonance gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('RZXGate', circuit_name, controls, targets, theta_radians=theta_radians, label='cross-resonance')
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('RZXGate', circuit_name, targets, theta_radians=theta_radians)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_cross_resonance_dagger(circuit_name, controls, targets, theta_radians, add_comments=True):
        out = "# crosss-resonance-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('RZXGate', circuit_name, controls, targets, theta_radians=-theta_radians, label='cross-resonance-dg')
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('RZXGate', circuit_name, targets, theta_radians=-theta_radians)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_molmer_sorensen(circuit_name, controls, targets, add_comments=True):
        out = "# molmer-sorensen gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('molmer_sorensen', circuit_name, controls, targets, label="molmer-sorensen")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('molmer_sorensen', circuit_name, targets, label="molmer-sorensen")
            out += f"{code}\n"
        return out


    @staticmethod
    def _gate_molmer_sorensen_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# molmer-sorensen-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('molmer_sorensen_dagger', circuit_name, controls, targets, label="molmer-sorensen-dagger")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('molmer_sorensen_dagger', circuit_name, targets, label="molmer-sorensen-dagger")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_berkeley(circuit_name, controls, targets, add_comments=True):
        out = "# berkeley gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('berkeley', circuit_name, controls, targets, label="berkeley")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('berkeley', circuit_name, targets, label="berkeley")
            out += f"{code}\n"
        return out


    @staticmethod
    def _gate_berkeley_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# berkeley-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('berkeley_dagger', circuit_name, controls, targets, label="berkeley-dagger")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('berkeley_dagger', circuit_name, targets, label="berkeley-dagger")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_ecp(circuit_name, controls, targets, add_comments=True):
        out = "# ecp gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('ecp', circuit_name, controls, targets, label="ecp")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('ecp', circuit_name, targets, label="ecp")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_ecp_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# ecp-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('ecp_dagger', circuit_name, controls, targets, label="ecp-dagger")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('ecp_dagger', circuit_name, targets, label="ecp-dagger")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_magic(circuit_name, controls, targets, add_comments=True):
        out = "# magic gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('magic', circuit_name, controls, targets, label="magic")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('magic', circuit_name, targets, label="magic")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_magic_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# magic-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('magic_dagger', circuit_name, controls, targets, label="magic-dagger")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('magic_dagger', circuit_name, targets, label="magic-dagger")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_w(circuit_name, controls, targets, add_comments=True):
        out = "# w gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('w', circuit_name, controls, targets, label="w")
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('w', circuit_name, targets, label="w")
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_circuit(
        circuit_name, controls, targets, circuit_id, circuit_gate_name, circuit_power, add_comments
    ):
        out = "# circuit gate\n" if add_comments else ""
        circuit_power = Exporter.get_circuit_power(circuit_power)
        take_inverse = circuit_power < 0
        if controls:
            code = Exporter.controlled_gate_code(f'qc_{circuit_gate_name}.to_gate', circuit_name, controls, targets, label=circuit_gate_name, inverse=take_inverse, power=abs(circuit_power))
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code(f'qc_{circuit_gate_name}.to_gate', circuit_name, targets, label=circuit_gate_name, inverse=take_inverse, power=abs(circuit_power))
            out += f"{code}"
        return out

    @staticmethod
    def _gate_qft(circuit_name, controls, targets, add_comments=True):
        out = "# qft gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('QFT', circuit_name, controls, targets)
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('QFT', circuit_name, targets)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_qft_dagger(circuit_name, controls, targets, add_comments=True):
        out = "# qft-dagger gate\n" if add_comments else ""
        if controls:
            code = Exporter.controlled_gate_code('QFT', circuit_name, controls, targets, inverse=True)
            out += f"{code}"
        else:
            code = Exporter.plain_gate_code('QFT', circuit_name, targets, inverse=True)
            out += f"{code}\n"
        return out

    @staticmethod
    def _gate_measure_x(circuit_name, targets, classic_bit, add_comments=True):
        raise BaseExporter.ExportException("The measure-x gate is not implemented.")

    @staticmethod
    def _gate_measure_y(circuit_name, targets, classic_bit, add_comments=True):
        raise BaseExporter.ExportException("The measure-y gate is not implemented.")

    @staticmethod
    def _gate_measure_z(circuit_name, targets, classic_bit, add_comments=True):
        out = "# measure-z gate\n" if add_comments else ""
        out += f"qc_{circuit_name}.measure({targets[0]}, {classic_bit})\n"
        return out
