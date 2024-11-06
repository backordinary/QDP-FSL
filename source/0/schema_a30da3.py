# https://github.com/Zhantai-Nuradinovich/TestGraphQLFlaskApp/blob/5001f96e28fbb73219ac4d92f843e19c2548bc5c/flask-graphql/tests/schema.py
from graphql.type.definition import (GraphQLArgument, GraphQLField,
                                     GraphQLNonNull, GraphQLObjectType,
                                     GraphQLList)
from graphql.type.scalars import GraphQLString, GraphQLBoolean, GraphQLInt
from graphql.type.schema import GraphQLSchema

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.monitor import job_monitor

from qiskit.tools.visualization import plot_bloch_multivector

# import basic plot tools
from qiskit.visualization import plot_histogram


def resolve_raises(*_):
    raise Exception("Throws!")


movies = [{"id": 1, "title": "Hello", "director": "me", "composer": "com", "release_date": "2020"},
          {"id": 2, "title": "Hell", "director": "m52e", "composer": "cm", "release_date": "202"},
          {"id": 3, "title": "Hel", "director": "e", "composer": "co", "release_date": "220"},
          {"id": 4, "title": "He", "director": "m", "composer": "om", "release_date": "2"}]

MovieType = GraphQLObjectType(
    name="movies",
    description="this represents a movie",
    fields={
        "id": GraphQLInt,
        "title": GraphQLString,
        "director": GraphQLString,
        "composer": GraphQLString,
        "release_date": GraphQLString
    }
)

GroverType = GraphQLObjectType(
    name="grover",
    description="this represents counts",
    fields={
        "name": GraphQLString,
        "value": GraphQLString
    }
)

QueryRootType = GraphQLObjectType(
    name="QueryRoot",
    fields={
        "thrower": GraphQLField(GraphQLNonNull(GraphQLString), resolve=resolve_raises),
        "request": GraphQLField(
            GraphQLNonNull(GraphQLString),
            resolve=lambda obj, info: info.context["request"].args.get("q"),
        ),
        "movies": GraphQLField(
            description="This is a list of books",
            type_=GraphQLList(MovieType),
            resolve=lambda x, y: movies
        ),
        "groverZeroZero": GraphQLField(
            description="Grover search alg 00",
            type_=GroverType,
            resolve=lambda x, y: GroverTwo()
        ),
        "context": GraphQLField(
            GraphQLObjectType(
                name="context",
                fields={
                    "session": GraphQLField(GraphQLString),
                    "request": GraphQLField(
                        GraphQLNonNull(GraphQLString),
                        resolve=lambda obj, info: info.context["request"],
                    ),
                },
            ),
            resolve=lambda obj, info: info.context,
        ),
        "test": GraphQLField(
            type_=GraphQLString,
            args={"who": GraphQLArgument(GraphQLString)},
            resolve=lambda obj, info, who="World": "Hello %s" % who,
        ),
        "characters": GraphQLField(
            type_=GraphQLString,
            args={"who": GraphQLArgument(GraphQLString)},
            resolve=lambda obj, info, who="World": "Hello %s" % who,
        ),
    },
)

MutationRootType = GraphQLObjectType(
    name="MutationRoot",
    description="Root mutations",
    fields={
        "writeTest": GraphQLField(type_=QueryRootType, resolve=lambda *_: QueryRootType),
        "addMovie": GraphQLField(
            type_=MovieType,
            description="Add a movie",
            args={
                "id": GraphQLArgument(GraphQLInt),
                "title": GraphQLArgument(GraphQLString),
                "director": GraphQLArgument(GraphQLString),
                "composer": GraphQLArgument(GraphQLString),
                "release_date": GraphQLArgument(GraphQLString)
            },
            resolve=lambda obj, info, id, title, director, composer, release_date: newMovie(id, title, director,
                                                                                            composer, release_date)
        )
    },
)


def newMovie(id, title, director, composer, release_date):
    movie = {"id": id, "title": title, "director": director, "composer": composer, "release_date": release_date}
    movies.append(movie)
    return movie


def GroverTwo():
    n = 2
    grover_circuit = QuantumCircuit(n)

    for qubit in range(n):
        grover_circuit.h(qubit)

    # Оракул для |00>

   # for qubit in range(n):
    #    grover_circuit.x(qubit)

#        grover_circuit.cz(0, 1)

 #   for qubit in range(n):
  #      grover_circuit.x(qubit)


    # Оракул для |01>
    # grover_circuit.x(1)
    # grover_circuit.cz(0, 1)
    # grover_circuit.x(1)

    # Оракул для |10>
    # grover_circuit.x(0)
    # grover_circuit.cz(0, 1)
    # grover_circuit.x(0)

    # Оракул для |11>
    grover_circuit.cz(0, 1)

    for qubit in range(n):
        grover_circuit.h(qubit)

    # Рефлексия

    for qubit in range(n):
        grover_circuit.z(qubit)
    grover_circuit.cz(0, 1)

    for qubit in range(n):
        grover_circuit.h(qubit)

    backend_sim = Aer.get_backend('statevector_simulator')
    job_sim = execute(grover_circuit, backend_sim)
    statevec = job_sim.result().get_statevector()
    plot_bloch_multivector(statevec)

    grover_circuit.measure_all()

    backend = Aer.get_backend('qasm_simulator')
    shots = 1024
    results = execute(grover_circuit, backend=backend, shots=shots).result()
    answer = results.get_counts()
    return {"name": "00", "value": answer}

    # qc = IBMQ.load_account()
    # provider = IBMQ.get_provider('ibm-q')
    # device = provider.get_backend('ibmqx2')
    # print(device)

    # Запуск на реальном устройстве
    # job = execute(grover_circuit, backend=device, shots=1024, max_credits=10)
    # job_monitor(job, interval=2)

    # results = job.result()
    # answer = results.get_counts(grover_circuit)
    # return plot_histogram(answer)


Schema = GraphQLSchema(QueryRootType, MutationRootType)
