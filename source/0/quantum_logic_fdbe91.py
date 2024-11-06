# https://github.com/iQuHACK/2021_MI-P-T-ex-AU/blob/b79c5718c40179cd89b3f0abce2b5dbad956645a/quantum_logic.py
import numpy as np
import qiskit
import re
from typing import List, Tuple
from qiskit_ionq_provider import IonQProvider

from util import draw_ids_on_ships


class QuantumCircuits:
    def __init__(self, qc_type='qasm_simulator'):
        if qc_type == 'qasm_simulator':
            self.backend = qiskit.BasicAer.get_backend('qasm_simulator')
        elif qc_type == "ionq_simulator":
            with open("topsecret.txt", "r") as f:
                token = f.readline().strip()
            provider = IonQProvider(token = token)
            self.backend = provider.get_backend("ionq_simulator")

    # write other types of backends

    def my_bin(self, el: int, n: int) -> str:
        return "0" * (n - len(bin(el)[2:])) + bin(el)[2:]

    def create_psi(self, ships, qset):
        self.nqubits = qset.nqubits
        self.intersection_ids = qset.intersection_ids
        self.possible_psi = [self.my_bin(i, self.nqubits) for i in np.arange(2 ** self.nqubits)]

        self.ships_nq = {}
        self.ship_start_pos = {}
        self.ships_n_re = {}
        pos = 0
        for ship in qset.ship_ids:
            self.ships_nq[ship] = ships[ship].nqubits
            self.ships_n_re[ship] = len(ships[ship].coordinates)
            self.ship_start_pos[ship] = pos
            pos += self.ships_nq[ship]

        for ship in qset.ship_ids:
            for i in range(self.ships_n_re[ship], 2 ** self.ships_nq[ship]):
                pattern = re.compile(
                    "[0,1]{%i}" % self.ship_start_pos[ship] +
                    self.my_bin(i, self.ships_nq[ship]) +
                    "[0,1]{%i}" % (qset.nqubits - self.ship_start_pos[ship] - self.ships_nq[ship]))
                rm_elements = []
                for psi in self.possible_psi:
                    if re.match(pattern, psi):
                        rm_elements.append(psi)
                for el in rm_elements:
                    self.possible_psi.remove(el)

        for intersection in qset.intersection_ids:
            # print(intersection)
            if self.ships_nq[intersection[0][0]] * self.ships_nq[intersection[1][0]] != 0:
                if self.ship_start_pos[intersection[0][0]] < self.ship_start_pos[intersection[1][0]]:
                    id_0, q_id_0 = intersection[0]
                    id_1, q_id_1 = intersection[1]
                else:
                    id_1, q_id_1 = intersection[0]
                    id_0, q_id_0 = intersection[1]
                pattern = re.compile(
                    "[0,1]{%i}" % self.ship_start_pos[id_0] +
                    self.my_bin(q_id_0, self.ships_nq[id_0]) +
                    "[0,1]{%i}" % (self.ship_start_pos[id_1] - self.ships_nq[id_0] - self.ship_start_pos[id_0]) +
                    self.my_bin(q_id_1, self.ships_nq[id_1]) +
                    "[0,1]{%i}" % (qset.nqubits - self.ship_start_pos[id_1] - self.ships_nq[id_1]))
                rm_elements = []
                # print(pattern)
                for psi in self.possible_psi:
                    if re.match(pattern, psi):
                        rm_elements.append(psi)

                for el in rm_elements:
                    self.possible_psi.remove(el)
            elif self.ships_nq[intersection[0][0]] == 0 or self.ships_nq[intersection[1][0]] == 0:
                if self.ships_nq[intersection[0][0]] != 0:
                    id_0, q_id_0 = intersection[0]
                else:
                    id_0, q_id_0 = intersection[1]
                pattern = re.compile(
                    "[0,1]{%i}" % self.ship_start_pos[id_0] +
                    self.my_bin(q_id_0, self.ships_nq[id_0]) +
                    "[0,1]{%i}" % (qset.nqubits - self.ship_start_pos[id_0] - self.ships_nq[id_0]))
                # print(pattern)
                rm_elements = []
                for psi in self.possible_psi:
                    if re.match(pattern, psi):
                        rm_elements.append(psi)
                for el in rm_elements:
                    self.possible_psi.remove(el)
            else:
                raise UserWarning("Two determenistic ship have an intersection")

        if len(self.possible_psi) == 0:
            raise UserWarning("This state is imposible even in quantum world =(")

    def change_sign(self, var: str):
        for i, v_i in enumerate(var[::-1]):
            if v_i == "0":
                self.qc.x(self.q_reg[i])

    def recursion_qubit(self, path: str, possible_psi: List[str]):
        last = [i[-1] for i in possible_psi[:]]
        if "1" in last and "0" in last:
            if path == "":
                self.qc.h(self.q_reg[0])
            else:
                self.change_sign(path)
                self.qc.mct(self.q_reg[:len(path)], self.aux_reg[0])
                self.qc.ch(self.aux_reg[0], self.q_reg[len(path)])
                self.qc.mct(self.q_reg[:len(path)], self.aux_reg[0])
                self.change_sign(path)
        elif "1" in last:
            if path == "":
                self.qc.x(self.q_reg[0])
            else:
                self.change_sign(path)
                self.qc.mct(self.q_reg[:len(path)], self.q_reg[len(path)])
                self.change_sign(path)

        for val in np.unique(last):
            if len(possible_psi[0]) > 1:
                new_psi = [s[:-1] for s in possible_psi if s[-1] == val]
                self.recursion_qubit(val + path, new_psi)

    def create_qc(self):
        self.q_reg = qiskit.QuantumRegister(self.nqubits)
        self.aux_reg = qiskit.QuantumRegister(1, "aux")
        self.c_reg = qiskit.ClassicalRegister(self.nqubits)
        self.qc = qiskit.QuantumCircuit(self.q_reg, self.aux_reg, self.c_reg)

        self.recursion_qubit("", self.possible_psi)
        self.qc.measure(self.q_reg, self.c_reg)
        self.qc = qiskit.compiler.transpile(self.qc, backend=self.backend)

    def run_qc(self, shots=1):
        job = qiskit.execute(self.qc, self.backend, shots=shots)  # , seed_simulator = 3
        result = job.result()
        self.count = result.get_counts()
        self.measured = list(self.count.keys())[0]

    def ship_measure(self, ship_id):
        assert ship_id in self.ships_nq
        if self.ships_nq[ship_id] == 0:
            return 0
        else:
            sum_res = 0
            val = self.measured[self.ship_start_pos[ship_id]: self.ship_start_pos[ship_id] + self.ships_nq[ship_id]]
            for i, v in enumerate(val[::-1]):
                sum_res += 2 ** i * int(v)
            return sum_res


class Battleship:
    def __init__(self, shape: int, coordinates: List[Tuple[int, int, int]]):
        self.shape = shape
        self.coordinates = coordinates
        self.nqubits = None
        self.damage = np.ones(shape).astype(int)
        self.health = 2
        self.q_set = None


class QubitSet:
    def __init__(self, idx: int, ship_ids: List[int], intersection_ids=None, nqubits=0):
        self.idx = idx
        self.nqubits = nqubits
        self.ship_ids = ship_ids
        self.intersection_ids: List[Tuple[Tuple[int, int], Tuple[int, int]]] = \
            intersection_ids if intersection_ids is not None else []
        self.current_shoots = []

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.ship_ids == other.ship_ids and \
                   sorted(self.intersection_ids) == sorted(other.intersection_ids) and \
                   self.nqubits == other.nqubits
        else:
            return False


class QuantumGame:
    def __init__(self, ships: List[Battleship], default_shoots_number: int, field_size: int):
        self.ships = ships
        for i, ship in enumerate(self.ships):
            self.ships[i].nqubits = int(np.ceil(np.log2(len(ship.coordinates))))
        self.default_shoots_number = default_shoots_number
        self.field_size = field_size

    def check_one_shoot(self, shoot_num: int, one_shoot_coordinate: Tuple[int, int]):
        for ship in self.ships:
            for variants in ship.coordinates:
                x_ship = variants[0]
                y_ship = variants[1]
                d_ship = variants[2]
                if x_ship <= one_shoot_coordinate[0] <= x_ship + (1 - d_ship) * (ship.shape - 1) and \
                        y_ship <= one_shoot_coordinate[1] <= y_ship + d_ship * (ship.shape - 1):
                    self.qsets[ship.q_set].current_shoots.append(shoot_num)
                    return 1
        return 0

    def shoot_cells(self, coordinates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        print("shoot coordinates:", coordinates)
        self.qsets = self.qubit_sets_from_ships()
        # for qset in self.qsets:
        #    print(qset.ship_ids)
        shoots_res: List[List[int]] = []
        for shoot_num, one_shoot_coordinate in enumerate(coordinates):
            if not self.check_one_shoot(shoot_num, one_shoot_coordinate):
                shoots_res.append([one_shoot_coordinate[0], one_shoot_coordinate[1], -1, -2])
        for one_set in self.qsets:
            if len(one_set.current_shoots) != 0:
                qc = None
                if one_set.nqubits > 0:
                    qc = QuantumCircuits()
                    qc.create_psi(self.ships, one_set)
                    qc.create_qc()
                    qc.run_qc()

                for one_shoot_num in one_set.current_shoots:
                    x_shoot, y_shoot = coordinates[one_shoot_num]
                    shoots_res.append([x_shoot, y_shoot, -1, -1])
                    for ship_id in one_set.ship_ids:
                        ship = self.ships[ship_id]
                        if one_set.nqubits > 0:
                            x_ship, y_ship, d_ship = ship.coordinates[qc.ship_measure(ship_id)]
                        else:
                            x_ship, y_ship, d_ship = ship.coordinates[0]
                        if x_ship <= x_shoot <= x_ship + (1 - d_ship) * (ship.shape - 1) and \
                                y_ship <= y_shoot <= y_ship + d_ship * (ship.shape - 1):

                            self.ships[ship_id].damage[y_shoot - y_ship + x_shoot - x_ship] = 0
                            # probably some of us want to know which part of the ship was shot
                            ship.health = 1
                            if sum(self.ships[ship_id].damage) == 0:
                                ship.health = 0
                            # check is ship dead
                            shoots_res[-1][2] = ship_id
                            shoots_res[-1][3] = ship.health

                for ship_id in one_set.ship_ids:
                    ship = self.ships[ship_id]
                    #                for ship_id, ship in enumerate(self.ships):
                    if ship.health != 2:
                        x_ship, y_ship, d_ship = ship.coordinates[qc.ship_measure(ship_id) if one_set.nqubits > 0 else 0]
                        ship.coordinates = [(x_ship, y_ship, d_ship)]
                        print(ship.coordinates)

                one_set.current_shoots = []

        shoots_res = [tuple(res) for res in shoots_res]
        # print(shoots_res)
        return shoots_res

    def get_shoots_number(self):
        return self.default_shoots_number

    def qubit_sets_from_ships(self) -> List[QubitSet]:
        qubitsets = {}
        nset_current = 0
        field = -100 * np.ones(shape=(self.field_size + 2, self.field_size + 2)).astype(int)
        for i, qship in enumerate(self.ships):
            new_set_idx = nset_current
            for j, shipcoords in enumerate(qship.coordinates):
                x, y, hv = shipcoords
                fromx = x - 1 if x != 0 else 0
                fromy = y - 1 if y != 0 else 0
                tox = x + 2 if hv == 1 else x + 1 + qship.shape
                toy = y + 2 if hv == 0 else y + 1 + qship.shape

                tox = min(tox, self.field_size)
                toy = min(toy, self.field_size)

                for x_ in range(fromx, tox):
                    for y_ in range(fromy, toy):
                        fvalue = field[x_][y_]
                        if fvalue >= 0 and i not in qubitsets[fvalue].ship_ids:
                            qset = QubitSet(idx=nset_current, ship_ids=qubitsets[fvalue].ship_ids + [i])
                            qubitsets[nset_current] = qset
                            qubitsets.pop(fvalue)
                            field = draw_ids_on_ships(field, [self.ships[k] for k in qset.ship_ids], nset_current)
                            nset_current += 1

            if new_set_idx == nset_current:
                qset = QubitSet(idx=nset_current, ship_ids=[i])
                qubitsets[nset_current] = qset
                field = draw_ids_on_ships(field, [self.ships[i]], nset_current)
                nset_current += 1
            # print(field)

        return self._construct_qubitset_list(qubitsets)

    def _construct_qubitset_list(self, qubitsets):
        qubitsets_list = []
        for v in qubitsets.values():
            qubitsets_list.append(v)
        for i, qset in enumerate(qubitsets_list):
            qubitsets_list[i].idx = i
            qubitsets_list[i].nqubits = sum([self.ships[j].nqubits for j in qset.ship_ids])
            for j in qset.ship_ids:
                self.ships[j].q_set = i
            for j, idx1 in enumerate(qset.ship_ids):
                for idx2 in qset.ship_ids[j + 1:]:
                    qubitsets_list[i].intersection_ids += self._get_ship_intersections(idx1, idx2)
        return qubitsets_list

    def _get_ship_intersections(self, idx1, idx2):
        qship1, qship2 = self.ships[idx1], self.ships[idx2]
        result = []
        for i, (x1, y1, hv1) in enumerate(qship1.coordinates):
            for j, (x2, y2, hv2) in enumerate(qship2.coordinates):
                fromx = x1 - 1 if x1 != 0 else 0
                fromy = y1 - 1 if y1 != 0 else 0
                tox = x1 + 2 if hv1 == 1 else x1 + 1 + qship1.shape
                toy = y1 + 2 if hv1 == 0 else y1 + 1 + qship1.shape
                x2end, y2end = x2 + (1 - hv2) * (qship2.shape - 1), y2 + hv2 * (qship2.shape - 1)
                if (fromx <= x2 < tox and fromy <= y2 < toy) or (fromx <= x2end < tox and fromy <= y2end < toy):
                    result.append(((idx1, i), (idx2, j)))
        return result
