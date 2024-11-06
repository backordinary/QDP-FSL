# https://github.com/TheGupta2012/QickStart/blob/65501cc7d5e98506b90e58e43d8d4a2aee551739/src/grader/graders/problem_5/grader.py
import os
import signal
from contextlib import contextmanager
from qiskit import execute, QuantumCircuit, Aer
from qiskit.circuit import qpy_serialization

from .answer import get_qram, get_qram_rotations
from numpy import exp, floor, ceil, log2

from ..google_sheets import append_values

# for the time out management


class TimeOutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeOutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
script_dir = os.path.dirname(__file__)
REL_INPUT_PATH = "./tests/inputs/"
REL_OUTPUT_PATH = "./tests/outputs/"
INPUT_PATH = os.path.join(script_dir, REL_INPUT_PATH)
OUTPUT_PATH = os.path.join(script_dir, REL_OUTPUT_PATH)

CORRECT_STMT = "Congratulations, your answer is correct!"
WRONG_STMT = "Uh-oh, that's not quite correct :("
SERVER_STMT = "The server couldn't process your request correctly at the moment, please try again in a while"
TIME_STMT = "Uh-oh, time limit exceeded for the problem :("

QBRAID_API = None
SV_BACKEND = Aer.get_backend("statevector_simulator")


def test_loader_1(file_path):
    # make it faster ...
    with open(file_path, "r") as file:

        for row_id, row in enumerate(file):
            if row_id == 0:
                n, m = row.split()
                n, m = int(n), int(m)
            else:
                array = [int(x) for x in row.split()]

    return n, m, array


def test_loader_2(file_path):
    n, rotations = None, []
    with open(file_path, "r") as file:
        for row_id, row in enumerate(file):
            if row_id == 0:
                n = int(row)
            else:
                r_type, value = row.split()
                value = float(value)
                rotations.append((r_type, value))

    return n, rotations


# may not be needed
def output_vector_loader(file_path):
    # make a statevector from a dict
    with open(file_path, "rb") as file:
        circuit = qpy_serialization.load(file)[0]
    statevector = execute(circuit, SV_BACKEND).result().get_statevector()
    return statevector


class grader1:
    ip_task_path = INPUT_PATH + "task-1-"
    op_task_path = OUTPUT_PATH + "task-1-"

    total_tests = 10
    time_limit = 10

    return_json = {
        "team-id": None,
        "problem": {"5.1": {"points": "75", "done": False, "wrong": False}},
    }

    @classmethod
    def get_team_id(cls):
        value = os.getenv("TEAMID", "Please enter your TEAMID")
        return value

    @staticmethod
    def _valid_circuit(n, m, circuit):
        if not isinstance(circuit, QuantumCircuit):
            return False

        size_circuit = len(circuit.qubits)
        index_size = int(ceil(log2(n)))
        value_size = int(floor(log2(m))) + 1

        return size_circuit == index_size + value_size

    @staticmethod
    def run(qram_4q):

        correct = 0
        # load your tests here
        for test in range(grader1.total_tests):
            ip_test_path = grader1.ip_task_path + str(test) + ".txt"
            n, m, test_array = test_loader_1(ip_test_path)
            try:
                user_circuit = qram_4q(m, test_array)

                if not grader1._valid_circuit(n, m, user_circuit):
                    break
                # build the qcirc
                user_statevector = (
                    execute(user_circuit, SV_BACKEND).result().get_statevector()
                )

                expected_circuit = get_qram(n, m, test_array)

                expected_statevector = (
                    execute(expected_circuit, SV_BACKEND).result().get_statevector()
                )

                # if doubt solved, fine

                # op_test_path = grader1.op_task_path + str(test) + ".qpy"
                # expected_statevector = output_vector_loader(op_test_path)
            except:
                break

            if user_statevector == expected_statevector:
                correct += 1
            else:
                break

        return correct == grader1.total_tests

    @classmethod
    def evaluate(cls, qram_4q):
        # update team
        if "TEAMID" in os.environ:
            cls.return_json["team-id"] = cls.get_team_id()
        else:
            cls.return_json["team-id"] = "NO TEAMID"
            print("Please add your TEAMID as an env variable")
            return

        tle = False

        try:
            with time_limit(grader1.time_limit):
                success = grader1.run(qram_4q)
        except:
            success = False
            tle = True

        # update json
        cls.return_json["problem"]["5.1"]["done"] = success
        cls.return_json["problem"]["5.1"]["wrong"] = not success

        # post the json to server

        append_values(
            [
                cls.return_json["team-id"],
                "5.1",
                cls.return_json["problem"]["5.1"]["points"],
                cls.return_json["problem"]["5.1"]["done"],
                cls.return_json["problem"]["5.1"]["wrong"],
            ]
        )

        # if request.ok:
        if success:
            print(CORRECT_STMT)
        else:
            if tle:
                print(TIME_STMT)
            else:
                print(WRONG_STMT)
        # else:
        #     print(SERVER_STMT)
        # wrong answer


class grader2:
    ip_task_path = INPUT_PATH + "task-2-"
    op_task_path = OUTPUT_PATH + "task-2-"
    total_tests = 10
    time_limit = 20
    return_json = {
        "team-id": None,
        "problem": {"5.2": {"points": "125", "done": False, "wrong": False}},
    }

    @classmethod
    def get_team_id(cls):
        value = os.getenv("TEAMID", "Please enter your TEAMID")
        return value

    @staticmethod
    def _valid_circuit(n, m, circuit):
        if not isinstance(circuit, QuantumCircuit):
            return False

        size_circuit = len(circuit.qubits)
        index_size = int(ceil(log2(n)))
        value_size = int(floor(log2(m))) + 1

        return size_circuit == index_size + value_size

    @staticmethod
    def run(qram_general):

        correct = 0
        # load your tests here
        for test in range(grader2.total_tests):
            ip_test_path = grader2.ip_task_path + str(test) + ".txt"
            n, m, test_array = test_loader_1(ip_test_path)
            try:
                user_circuit = qram_general(n, m, test_array)

                if not grader2._valid_circuit(n, m, user_circuit):
                    break
                # build the qcirc
                user_statevector = (
                    execute(user_circuit, SV_BACKEND).result().get_statevector()
                )

                expected_circuit = get_qram(n, m, test_array)

                expected_statevector = (
                    execute(expected_circuit, SV_BACKEND).result().get_statevector()
                )

                """Diff output format for statevectors!!"""
                # op_test_path = grader1.op_task_path + str(test) + ".json"
                # expected_statevector = output_vector_loader(op_test_path)
            except:
                break
            if user_statevector == expected_statevector:
                correct += 1
            else:
                break

        return correct == grader2.total_tests

    @classmethod
    def evaluate(cls, qram_general):

        # update team
        if "TEAMID" in os.environ:
            cls.return_json["team-id"] = cls.get_team_id()
        else:
            cls.return_json["team-id"] = "NO TEAMID"
            print("Please add your TEAMID as an env variable")
            return

        tle = False
        success = False
        try:
            with time_limit(grader2.time_limit):
                success = grader2.run(qram_general)
        except:
            success = False
            tle = True

        # update json
        cls.return_json["problem"]["5.2"]["done"] = success
        cls.return_json["problem"]["5.2"]["wrong"] = not success

        # post the json to server : to do

        append_values(
            [
                cls.return_json["team-id"],
                "5.2",
                cls.return_json["problem"]["5.2"]["points"],
                cls.return_json["problem"]["5.2"]["done"],
                cls.return_json["problem"]["5.2"]["wrong"],
            ]
        )

        # if request.ok:
        if success:
            print(CORRECT_STMT)
        else:
            if tle:
                print(TIME_STMT)
            else:
                print(WRONG_STMT)
        # else:
        #     print(SERVER_STMT)
        # wrong answer


class grader3:

    ip_task_path = INPUT_PATH + "task-3-"
    op_task_path = OUTPUT_PATH + "task-3-"
    total_tests = 10
    time_limit = 20
    return_json = {
        "team-id": None,
        "problem": {"5.3": {"points": "200", "done": False, "wrong": False}},
    }

    @classmethod
    def get_team_id(cls):
        value = os.getenv("TEAMID", "Please enter your TEAMID")
        return value

    @staticmethod
    def _valid_circuit(n, circuit):
        if not isinstance(circuit, QuantumCircuit):
            return False

        size_circuit = len(circuit.qubits)
        index_size = ceil(log2(n))
        value_size = 1

        return size_circuit == index_size + value_size

    @staticmethod
    def run(qram_general):

        correct = 0
        # load your tests here
        for test in range(grader3.total_tests):
            ip_test_path = grader3.ip_task_path + str(test) + ".txt"
            n, rotations = test_loader_2(ip_test_path)
            try:
                user_circuit = qram_general(n, rotations)

                if not grader3._valid_circuit(n, user_circuit):
                    break
                # build the qcirc
                user_statevector = (
                    execute(user_circuit, SV_BACKEND).result().get_statevector()
                )

                expected_circuit = get_qram_rotations(n, rotations)

                expected_statevector = (
                    execute(expected_circuit, SV_BACKEND).result().get_statevector()
                )

                """Diff output format for statevectors!!"""
                # op_test_path = grader1.op_task_path + str(test) + ".json"

                # expected_statevector = output_vector_loader(op_test_path)
            except:
                break

            if user_statevector == expected_statevector:
                correct += 1
            else:
                break

        return correct == grader3.total_tests

    @classmethod
    def evaluate(cls, qram_general):
        # update team
        if "TEAMID" in os.environ:
            cls.return_json["team-id"] = cls.get_team_id()
        else:
            cls.return_json["team-id"] = "NO TEAMID"
            print("Please add your TEAMID as an env variable")
            return

        tle = False

        try:
            with time_limit(grader3.time_limit):
                success = grader3.run(qram_general)
        except:
            success = False
            tle = True

        # update json
        cls.return_json["problem"]["5.3"]["done"] = success
        cls.return_json["problem"]["5.3"]["wrong"] = not success

        # post the json to server : to do

        append_values(
            [
                cls.return_json["team-id"],
                "5.3",
                cls.return_json["problem"]["5.3"]["points"],
                cls.return_json["problem"]["5.3"]["done"],
                cls.return_json["problem"]["5.3"]["wrong"],
            ]
        )

        # if request.ok:
        if success:
            print(CORRECT_STMT)
        else:
            if tle:
                print(TIME_STMT)
            else:
                print(WRONG_STMT)
        # else:
        #     print(SERVER_STMT)
        # wrong answer
