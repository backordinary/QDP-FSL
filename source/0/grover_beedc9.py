# https://github.com/lorenzosaraiva/INF2102/blob/42bb8642036ca2240e50865c20577af5246aa20d/grover.py
# Código criado por Lorenzo Saraiva. 
###############  Módulo do Grover Search Algorithm  ############### 

import numpy as np
import math

from initialize import Initialize
from utils import reverse, to_bin
from encode import Encoding

from qiskit import Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import IBMQ

# Token do account da IBM
TOKEN = "e492922085ef19c9a6385f62e574331f4d14fed52b4cf22c1602b07926149ccdec378bc33fe141b02236cd93b127edc0bfd607ac533cb864b3ab80368aabfa87"

def generic_oracle(qc, search_qubits, output_qubit, target, n):
    """Função que constrói um oráculo dinamicamente para a iteração do Grover
    
    Parâmetros:
        qc (QuantumCircuit): Circuito quântico que será modificado
        search_qubits (QuantumRegister): Qubits que representam a parte da busca 
            nos quais o Oráculo do GSA será aplicado
        output_qubit (QuantumRegister): Qubit que foi preparado para ser o alvo
            do phase kickback
        target(int): Valor que para qual o Oráculo irá fazer o phase kickback
        n(int): Número de qubits em um lado da base de dados entrelaçada
    """

    # Converte o target para binário
    binary = reverse(to_bin(target, n))

    # Onde tem zeros, faz um gate Pauli-X
    for i in range(n):
        if binary[i] == '0':
            qc.x(search_qubits[i])

    # Utiliza um gate Multi-controlled Toffolli para fazer o phase kickback e marcar a solução correta
    qc.mct(search_qubits, output_qubit)


    # Desfaz as mudanças
    for i in range(n):
        if binary[i] == '0':
            qc.x(search_qubits[i])


def diffuser(qc, n, qubits, uncompute, prepare, v):
    """Função que constrói um difusor para a iteração do Grover
    
    Parâmetros:
        qc (QuantumCircuit): Circuito quântico que será modificado
        n(int): Número de qubits em um lado da base de dados entrelaçada
        qubits (QuantumRegister): Qubits nos quais será aplicado o difusor
        uncompute (QuantumCircuit): Circuito para a transformação |s> -> |00..0> 
        prepare(QuantumCircuit): Circuito para a transformação |00..0> -> |s>
        v(int): Número da cláusula atômica que está sendo buscada
    """

    size = n * 2
    start = v * size
    
    # Aplica transformação |s> -> |00..0> 
    qc.append(uncompute, qubits[start: start + size])

    # Aplica transformação |00..0> -> |11..1> 
    qc.x(qubits[start: start + size])

    # Aplica um multi-controlled Z 
    qc.h(qubits[start + size - 1])
    qc.mct(list(range(start, start + size - 1)), start + size - 1)  # multi-controlled-toffoli
    qc.h(qubits[start + size - 1])

    
    # Aplica transformação |11..1> -> |00..0>
    qc.x(qubits[start: start + size])

    qc.barrier()

    # Aplica transformação |00..0> -> |s>
    qc.append(prepare, qubits[start: start + size])


def grover_search_algorithm(n, entries, initial_state, variables, target, shots, is_local, device):
    """Função principal que execute o Grover Search Algorithm

    Parâmetros:
        n(int): Número de qubits em um lado da base de dados entrelaçada
        entries(int): Número de cláusulas atômicas
        initial_state ([float]): Vetor que representa o estado quântico
        variables (int): Número de claúsulas atômicas a serem buscadas
        target(int): Caso seja uma execução única, o número da claúsula atômica
            que será buscada o par
        shots(int): Número de execuções do circuito
        is_local(bool): Booleano que marca se a execução é local ou no backend da IBM
        device(str): Nome do simulador ou computador quântico onde o programa será
            executado

    Retorno:
        answer_dict(dict): Dicionário com os resultados da execução do circuito

    """



    # Cria conjunto de gates |00..0> -> |initial_state> e |initial_state> -> |00..0> 


    ##### Método presente no módulo encode.py #####
    # enc = Encoding(initial_state)
 
    # prepare = enc.qcircuit

    # uncompute = prepare.inverse()
    ##############################################

    ##### Método presente no módulo initialize.py #####
    init_instruction = Initialize(initial_state)

    uncompute = init_instruction.gates_to_uncompute()

    prepare = uncompute.inverse()
    ##############################################
    
    size = n * 2

    # Cria os registradores 
    var_qubits = QuantumRegister((size) * variables, name='v') # Define o registrador quântico que representa as variáveis
    output_qubit = QuantumRegister(1, name='out') # Define um qubit para o phase kickback
    c_bits = ClassicalRegister(n * variables, name='c') # Define um registrador clássico para ser feita a medição

    qc = QuantumCircuit(var_qubits, output_qubit, c_bits)  # Cria o circuito quântico

    
    # Prepara o estado quântico correto em 'variables' grupos de 'size' qubits
    for i in range(variables):   
        qc.append(prepare, var_qubits[i * size: i * size + size]) 
        

    qc.initialize([1, -1]/np.sqrt(2), output_qubit) # Prepara o qubit de phase kickback no estado correto


    num_iterations = math.floor(((math.pi) * math.sqrt(entries))/4) # Calcula o numero ótimo de iterações do GSA
  
    # Loop do GSA

    for v in range(variables): 

        for i in range(num_iterations):

            # Aplica Oracle oráculo nos qubits de busca
            if variables == 1:
                generic_oracle(qc, var_qubits[v * size: v * size + n], output_qubit, target, n)  
            else:
                generic_oracle(qc, var_qubits[v * size: v * size + n], output_qubit, v, n)          
            

            qc.barrier() 

            diffuser(qc, n, var_qubits, uncompute, prepare, v)  # Aplica o difusor
            qc.barrier()
        
        qc.measure(var_qubits[v * size + n:v * size + 2 * n], c_bits[v * n:v * n + n]) # Faz as medições
            

    qc.barrier()

    # Simulação do circuito
    backend = ""
    if is_local:
        backend = Aer.get_backend(device) # Escolhe o simulador 
    else:
        IBMQ.save_account(TOKEN, overwrite=True)
        provider = IBMQ.load_account()
        backend = provider.get_backend(device)

   
    transpiled_qc = transpile(qc, backend) # Faz o transpiling para o simulador escolhido
    #qobj = assemble(transpiled_qc) 
    result = backend.run(transpiled_qc, shots = shots).result() # Executa o circuito 'shots' vezes

    

    answer_dict = dict(sorted(result.get_counts().items(), key=lambda item: item[1])) # Coleta e ordena os resultados da execução

    return answer_dict
