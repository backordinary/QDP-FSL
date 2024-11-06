# https://github.com/ownsupernoob2/Blimp-Academy-Flask/blob/9e43c2927ea9c40c924ee60d49f4e513fd374adc/generate_quiz_data.py
import pathlib

from models.circuit import Circuit, LearnCircuit
from models.quiz_data import QuizQuestion, MultipleChoices, CircuitDraw

from matplotlib import pyplot
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit.visualization import plot_bloch_vector, plot_bloch_multivector
from generate_bloch_sphere import generate_bloch_sphere_from_circuit

total_questions = 5


def generate_quiz_data():
    # generate bloch sphere for q1
    q1_bloch_spheres = []
    for answer, coords in {
        "A": [1, 0, 0],
        "B": [0, 0, -1],
        "C": [0, 0, 1],
        "D": [1, 0, 1]
    }.items():
        fig = plot_bloch_vector(coords)
        fpath = f"static/images/q1_{answer}.png"

        fpath = pathlib.Path(str(pathlib.Path(__file__).parent) + "/static/images/q1_{answer}.png")
        fig.savefig(fpath, format="png", transparent=True)
        pyplot.close(fig)
        q1_bloch_spheres.append(fpath)

    q1 = QuizQuestion(
        question_id="1",
        progression=1 / total_questions,
        name="What is the state of the qubit after this circuit?",
        name_media=Circuit(
            gates=["X"],
        ),
        media_type="circuit",
        is_multiple_choice=True,
        multiple_choice=MultipleChoices(
            correct="B",
            answers_media=q1_bloch_spheres,
            answers_type="path"
        ),
        next_page="/quiz/2"
    )

    q2_bloch_sphere = plot_bloch_vector([-1, 0, 0])
    fname = pathlib.Path(f"static/images/q2_media.png")
    fname = pathlib.Path(str(pathlib.Path(__file__).parent) + "/static/images/q2_media.png")
    q2_bloch_sphere.savefig(fname, format="png", transparent=True)
    pyplot.close(q2_bloch_sphere)

    q2_answers = [
        Circuit(
            gates=["X"],
        ),
        Circuit(
            gates=["H"],
        ),
        Circuit(
            gates=["X", "H"],
        ),
        Circuit(
            gates=["H", "I"]
        )
    ]

    q2 = QuizQuestion(
        question_id="2",
        progression=2 / total_questions,
        name="Which Circuit will produce the following state?",
        name_media=fname,
        media_type="path",
        is_multiple_choice=True,
        multiple_choice=MultipleChoices(
            correct="C",
            answers_media=q2_answers,
            answers_type="circuit"
        ),
        next_page="/quiz/3"
    )

    q3_bloch_sphere = plot_bloch_vector([0, 0, 1])
    fname = pathlib.Path(f"static/images/q3_media.png")
    fname = pathlib.Path(str(pathlib.Path(__file__).parent) + "/static/images/q3_media.png")
    q3_bloch_sphere.savefig(fname, format="png", transparent=True)
    pyplot.close(q3_bloch_sphere)

    q3_answers = [
        "0%", "25%", "50%", "100%"
    ]

    q3 = QuizQuestion(
        question_id="3",
        progression=3 / total_questions,
        name="What is the probability of measuring a 1 in this state?",
        name_media=fname,
        media_type="path",
        is_multiple_choice=True,
        multiple_choice=MultipleChoices(
            correct="A",
            answers_media=q3_answers,
            answers_type="str"
        ),
        next_page="/quiz/4"
    )

    q4_bloch_sphere = plot_bloch_vector([-1, 0, 0])
    fname = pathlib.Path(f"static/images/q4_media.png")
    fname = pathlib.Path(str(pathlib.Path(__file__).parent) + "/static/images/q4_media.png")
    q4_bloch_sphere.savefig(fname, format="png", transparent=True)
    pyplot.close(q4_bloch_sphere)

    q4 = QuizQuestion(
        question_id="4",
        progression=4 / total_questions,
        name="Create a circuit that makes this state",
        name_media=fname,
        media_type="path",
        is_circuit_draw=True,
        circuit_draw=CircuitDraw(
            correct=Circuit(
                gates=["X", "H"]
            )
        ),
        next_page="/quiz/5"
    )

    empty_circuit = QuantumCircuit(1)
    sv = Statevector.from_instruction(empty_circuit)
    q5_bloch_sphere = plot_bloch_multivector(sv)
    fname = pathlib.Path(f"static/images/q5_media.png")
    fname = pathlib.Path(str(pathlib.Path(__file__).parent) + "/static/images/q5_media.png")
    q5_bloch_sphere.savefig(fname, format="png", transparent=True)
    pyplot.close(q5_bloch_sphere)
    q5 = QuizQuestion(
        question_id="5",
        progression=5 / total_questions,
        name="Create a circuit that will measure 1 with probability 100%",
        is_circuit_draw=True,
        circuit_draw=CircuitDraw(
            start=LearnCircuit(
                learn_circuit_id="q5"
            ),
            correct=Circuit(
                gates=["X"]
            )
        ),
        interactive_sphere=True,
        bloch_sphere=fname,
        next_page="/quiz_result"
    )

    global questions
    questions = [q1, q2, q3, q4, q5]

    questions = {q.question_id: q for q in questions}
    return questions
