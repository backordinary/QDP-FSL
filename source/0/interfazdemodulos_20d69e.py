# https://github.com/Albertojuanse/ibmq_chemical_core/blob/1ea9eb6232dafadd0640e6570819e94629605335/ibmq_chemical_core/interfazdemodulos.py
"""Esta clase servirá de enlace entre Quiskit, drivers... y el cliente"""

# Conexiones por arquitectura

# Dependencias de Qiskit
from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister

# Dependencias de aqua
from qiskit_aqua_chemistry.drivers import ConfigurationManager
from qiskit_aqua_chemistry import FermionicOperator
from qiskit_aqua import (get_algorithm_instance, get_optimizer_instance,
                          get_variational_form_instance, get_initial_state_instance)
# Dependencias de PySCF

# Otras dependencias
from dependencies.eventos import Evento
import dependencies.integrals
import numpy

# Funciones necesarias para ejecutar ibmq_vqe


def __procesar_molecula_pyscf(configuracionmolecula, barrido=None):
    """Esta función construye una molécula usando el driver PySCF con los datos de entrada proporcionados"""
    # Se calcula la distancia entre los átomos y se reconfigura la molécula
    definicion_atomo = configuracionmolecula["configuracion"]["properties"]["atom"].split(";")
    definicion_atomo_1 = definicion_atomo[0].split()
    definicion_atomo_2 = definicion_atomo[1].split()
    atomo1 = definicion_atomo_1[0]
    atomo1x = definicion_atomo_1[1]
    atomo1y = definicion_atomo_1[2]
    atomo1z = definicion_atomo_1[3]
    atomo2 = definicion_atomo_2[0]
    atomo2x = definicion_atomo_2[1]
    atomo2y = definicion_atomo_2[2]
    atomo2z = definicion_atomo_2[3]

    # Si es modo dibujo se impone la distancia
    if barrido:
        nueva_definicion = atomo1 + " 0 0 0;" + atomo2 + " 0 0 " + str(barrido)
        configuracionmolecula["configuracion"]["properties"]["atom"] = nueva_definicion
    else:
        distancia = float(numpy.sqrt(float(numpy.power(float(atomo2x) - float(atomo1x), 2)) +
                                     float(numpy.power(float(atomo2y) - float(atomo1y), 2)) +
                                     float(numpy.power(float(atomo2z) - float(atomo1z), 2))
                                     ))
        nueva_definicion = atomo1 + " 0 0 0;" + atomo2 + " 0 0 " + str(distancia)
        configuracionmolecula["configuracion"]["properties"]["atom"] = nueva_definicion
    molecula = dependencies.integrals.compute_integrals(configuracionmolecula["configuracion"]["properties"])
    return molecula


def procesar_molecula(configuracionmolecula, distancia=None):
    """Esta función analiza que tipo de procesado necesita una molécula para pedir su construcción"""
    gestorconfiguracion = ConfigurationManager()
    if configuracionmolecula["driver"] == "PYSCF":
        molecula = __procesar_molecula_pyscf(configuracionmolecula, distancia)
    else:
        driver = gestorconfiguracion.get_driver_instance(configuracionmolecula["driver"])
        molecula = driver.run(configuracionmolecula["configuracion"])
    return molecula


def leer_propiedades_molecula(molecula, supervisorderesultados=None):
    """Esta función recupera las propiedades necesarias para los cálculos del objeto molécula"""
    propiedadesmolecula = {}
    propiedadesmolecula["h1"] = molecula._one_body_integrals
    propiedadesmolecula["h2"] = molecula._two_body_integrals
    propiedadesmolecula["energia_de_repulsion_nuclear"] = molecula._nuclear_repulsion_energy
    propiedadesmolecula["numero_de_particulas"] = molecula._num_alpha + molecula._num_beta
    propiedadesmolecula["numero_de_orbitales"] = molecula._num_orbitals
    propiedadesmolecula["numero_de_orbitales_de_spin"] = molecula._num_orbitals * 2
    propiedadesmolecula["energia_HF"] = molecula._hf_energy
    # Solo se remiten los resultados si se precisan dado el modo de ejecución
    if supervisorderesultados:
        eventoderesultados = Evento("evento de resultados")
        eventoderesultados.registro(supervisorderesultados)
        eventoderesultados.anunciarse(
            {"mensaje": "Energía con el método Hartree-Fock: {}".format(propiedadesmolecula["energia_HF"] -
                                                                        propiedadesmolecula["energia_de_repulsion_nuclear"])
             })
        eventoderesultados.anunciarse(
            {"mensaje": "# de electrones: {}".format(propiedadesmolecula["numero_de_particulas"])
             })
        eventoderesultados.anunciarse(
            {"mensaje": "# de orbitales de spin: {}".format(propiedadesmolecula["numero_de_orbitales_de_spin"])
             })
        eventoderesultados.dar_de_baja_todos()
    return propiedadesmolecula


def __obtener_operador_fermionico(h1, h2):
    """Esta función obtiene el operador fermiónico en función de las integrales de la molécula"""
    return FermionicOperator(h1=h1, h2=h2)


def __reductor_de_orbitales(propiedadesmolecula, operadorfermionico):
    """Esta función elimina de la configuración electrónica de la molécula los orbitales que no se van a computar"""
    energy_shift = 0.0
    # please be aware that the idx here with respective to original idx
    freeze_list = [0]
    remove_list = [-3, -2]  # negative number denotes the reverse order

    # prepare full idx of freeze_list and remove_list
    # convert all negative idx to positive
    remove_list = [x % propiedadesmolecula["numero_de_orbitales"] for x in remove_list]
    freeze_list = [x % propiedadesmolecula["numero_de_orbitales"] for x in freeze_list]
    # update the idx in remove_list of the idx after frozen, since the idx of orbitals are changed after freezing
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + propiedadesmolecula["numero_de_orbitales"] - len(freeze_list) for x in remove_list]
    freeze_list += [x + propiedadesmolecula["numero_de_orbitales"] for x in freeze_list]

    # prepare fermionic hamiltonian with orbital freezing and eliminating, and then map to qubit hamiltonian
    # and if PARITY mapping is selected, reduction qubits

    if len(freeze_list) > 0:
        operadorfermionico, energy_shift = operadorfermionico.fermion_mode_freezing(freeze_list)
        propiedadesmolecula["numero_de_orbitales_de_spin"] -= len(freeze_list)
        propiedadesmolecula["numero_de_particulas"] -= len(freeze_list)
    if len(remove_list) > 0:
        operadorfermionico = operadorfermionico.fermion_mode_elimination(remove_list)
        propiedadesmolecula["numero_de_orbitales_de_spin"] -= len(remove_list)

    return operadorfermionico, energy_shift


def __necesaria_reduccion(tipo_de_mapeo):
    """Esta función devuelve True si es preciso reducir el número de orbitales electrónicos"""
    return True if tipo_de_mapeo == "parity" else False


def __obtener_operador_qbit(propiedadesmolecula, configuracionaqua, operadorfermionico):
    """Esta función permite mapear el hamiltoniano del problema a la evolución que habrá en el ordenador"""
    operadorqubit = operadorfermionico.mapping(map_type=configuracionaqua["general"]["tipo_de_mapeo"], threshold=0.00000001)
    operadorqubit = operadorqubit.two_qubit_reduced_operator(
        propiedadesmolecula["numero_de_particulas"]) if __necesaria_reduccion(configuracionaqua["general"]["tipo_de_mapeo"]) else operadorqubit
    operadorqubit.chop(10**-10)
    return operadorqubit


def obtener_operadores_hamiltonianos(propiedadesmolecula, configuracionaqua):
    """Esta función construye el hamiltoniano de la evolución con el operador fermiónico"""
    operadorfermionico = __obtener_operador_fermionico(propiedadesmolecula["h1"], propiedadesmolecula["h2"])
    operadorfermionico, energy_shift = __reductor_de_orbitales(propiedadesmolecula, operadorfermionico)
    operadorqubit = __obtener_operador_qbit(propiedadesmolecula, configuracionaqua, operadorfermionico)
    return {"operadorfermionico": operadorfermionico, "energy_shift": energy_shift, "operadorqubit": operadorqubit}


def calcular_energia_clasico(propiedadesmolecula, operadorqubit, energy_shift, supervisorderesultados=None):
    """Esta función usa obtiene la energía de enlace menor mediante el cálculo del menor autovalor del sistema"""
    exact_eigensolver = get_algorithm_instance('ExactEigensolver')
    exact_eigensolver.init_args(operadorqubit, k=1)
    ret = exact_eigensolver.run()

    if supervisorderesultados:
        eventoderesultados = Evento("evento de resultados")
        eventoderesultados.registro(supervisorderesultados)
        eventoderesultados.anunciarse({"mensaje": "The computed energy is: {:.12f}".format(ret["eigvals"][0].real)})
        eventoderesultados.anunciarse({"mensaje": "The total ground state energy is: {:.12f}".format(ret["eigvals"][0].real +
                                                    energy_shift + propiedadesmolecula["energia_de_repulsion_nuclear"])})
        eventoderesultados.dar_de_baja_todos()


def configurar_COBYLA(configuracionaqua):
    """Esta función obtiene una instancia del optimizador COBYLA configurada"""
    cobyla = get_optimizer_instance('COBYLA')
    cobyla.set_options(maxiter=configuracionaqua["COBYLA"]["max_eval"])
    return cobyla


def configurar_hartreefock(operadorqubit, configuracionaqua, propiedadesmolecula):
    """Esta función obtiene una instancia configurada del método de aproximación de la función de onda Hartree-Fock"""
    HF = get_initial_state_instance('HartreeFock')
    HF.init_args(operadorqubit.num_qubits, propiedadesmolecula["numero_de_orbitales_de_spin"],
                 configuracionaqua["general"]["tipo_de_mapeo"], __necesaria_reduccion(configuracionaqua["general"]["tipo_de_mapeo"]),
                 propiedadesmolecula["numero_de_particulas"])
    return HF


def configurar_UCCSD(operadorqubit, configuracionaqua, propiedadesmolecula, HF):
    """Esta función obtiene una instancia configurada del método numérico UCCSD"""
    UCCSD = get_variational_form_instance('UCCSD')
    UCCSD.init_args(operadorqubit.num_qubits, depth=configuracionaqua["UCCSD"]["profundidad"],
                    num_orbitals=propiedadesmolecula["numero_de_orbitales"],
                    num_particles=propiedadesmolecula["numero_de_particulas"],
                    active_occupied=configuracionaqua["UCCSD"]["orbitales_activos_ocupados"],
                    active_unoccupied=configuracionaqua["UCCSD"]["orbitales_activos_no_ocupados"],
                    initial_state=HF , qubit_mapping=configuracionaqua["general"]["tipo_de_mapeo"],
                    two_qubit_reduction=__necesaria_reduccion(configuracionaqua["general"]["tipo_de_mapeo"]),
                    num_time_slices=configuracionaqua["UCCSD"]["numero_de_slices"])
    return UCCSD


def configurar_VQE(operadorqubit, UCCSD, cobyla, backend):
    """Esta función obtiene una instancia configurada del algoritmo VQE"""
    VQE = get_algorithm_instance('VQE')
    VQE.setup_quantum_backend(backend)
    VQE.init_args(operadorqubit, 'matrix', UCCSD, cobyla)
    return VQE

# Otras funciones


def circuito_numeros_aleatorios(cifras):
    """Esta función genera un circuito con el que obtener números aleatorios de n cifras"""
    registroscuanticos = QuantumRegister(cifras)
    registrosclasicos = ClassicalRegister(cifras)
    circuito = QuantumCircuit(registroscuanticos, registrosclasicos)
    for i in range(cifras):
        circuito.h(registroscuanticos[i])
    circuito.measure(registroscuanticos, registrosclasicos)
    return circuito
