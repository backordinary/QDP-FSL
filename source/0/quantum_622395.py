# https://github.com/MarceloVelludo/Dpq/blob/8a5c4dd22975109e9a1cab89240d053dd1ea3c15/code/quantum.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from qiskit.providers.ibmq import least_busy
from math import pi
from sklearn.metrics import mean_squared_error
from collections import Counter
from qiskit import IBMQ,transpile
from qiskit.quantum_info.operators import Operator
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers import JobStatus, JobError
from utils import make_dir
from check_stage import check_quantum
from graphics import quantum_report
from fpdf import FPDF

def init_pdf(pdf, title):
    
    #pdf.add_page()
    #Font
    #Line break
    pdf.ln(5)
    pdf.set_font("Times", 'B', 13)
    #Title
    pdf.write(5,title)
    #pdf.cell(60,140, title,0,1, 'L')
    #Line break
    pdf.ln(1)
    pdf.set_font("Times", '', 12)
    
    return pdf
#Classe que trabalha com a parte quântica
class Qt:
    
    def __init__(self, name, model,shots=20000, pdf = FPDF()):
        IBMQ.save_account('e6e296eb7eff807ce8dd197fe1a59dde9e017c44fbc49fd2cb7707ea1936510f1b2eb5fe405d4fa9dacb818c2cae8ba99a6f4a6c3ea1340db9efc794b2356c50')
        IBMQ.load_account()
        self.name = name
        self.shots = shots
        self.provider = IBMQ.get_provider(hub='ibm-q')
        self.path = make_dir("%s/quantum"%self.name)
        self.model = model
        self.pdf = init_pdf(pdf, "Modelo Quântico: %s"%name)
        
    def set_textimg_pdf(self, nome , descricao = '', path =''):
        self.pdf.write(5,nome)
        self.pdf.write(5,descricao)
        if path != '':
            self.pdf.cell(5,5, '',0,1, 'L')
            self.pdf.image(path,x=5,w=120,h=80)
            self.pdf.cell(5,5, '',0,1, 'L')

        return

        
    #Seleciona dentre os backends pequenos o que estiver mais desocupado.
    def get_quantum_backend(self):
        small_devices = self.provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                          and not x.configuration().simulator)
        backend = least_busy(small_devices)
        return backend

    #Monta circuito de acordo com hamiltoniana.
    def compose_circ(self,j1=None, j2=None, b1=None, b2=None, j12=None, t=None):
        #Trotter-suzuki parameters:
        # n-Numero de divisões
        n = 1
        # delt_t- divisão do tempo por 
        delt_j1 = (j1*t)/n
        delt_j2 = (j2*t)/n
        delt_b1 = (b1*t)/n
        delt_b2 = (b2*t)/n
        delt_j12 = (j12*t)/(2*n)

        #pi
        pi =3.1415

        #Inicializa circuito.
        circ_h = QuantumCircuit(2,2)

        #Inicializa os dois qubits no estado inicial.
        circ_h.h(0)
        circ_h.h(1)

        #Parametros para os gates
        params_ = {'h1': delt_j1, 'h2': delt_j2, 'h3': delt_j12, 'h4': delt_b1, 'h5': delt_b2}
        params = {'h1': Parameter('h1'), 'h2': Parameter('h2'), 'h3': Parameter('h3'), 'h4': Parameter('h4'), 'h5': Parameter('h5')}


        #Parte h1
        def h1_circ(parametro = params['h1']):
            circ_h1 = QuantumCircuit(2,2)
            circ_h1.barrier([0,1])
            #circ_h1.h(1)
            circ_h1.cnot(1,0)
            circ_h1.rz(parametro,1)
            circ_h1.cnot(1,0)
            #circ_h1.h(1)
            #circ_h1.i(1)
            circ_h1.barrier([0,1])
            return circ_h1
        #Parte h1
        def h1_circ_2(parametro = params['h1']):
            circ_h1 = QuantumCircuit(2,2)
            circ_h1.barrier([0,1])
            #circ_h1.h(1)
            circ_h1.rz(parametro,1)
            #circ_h1.h(1)
            #circ_h1.i(1)
            circ_h1.barrier([0,1])
            return circ_h1


        #Parte h2
        def h2_circ(parametro = params['h2']):
            circ_h2 = QuantumCircuit(2,2)
            circ_h2.barrier([0,1])
            circ_h2.cnot(0,1)
            circ_h2.rz(parametro,0)
            circ_h2.cnot(0,1)
            circ_h2.barrier([0,1])
            return circ_h2

        #Parte h2
        def h2_circ_2(parametro = params['h2']):
            circ_h2 = QuantumCircuit(2,2)
            circ_h2.barrier([0,1])
            circ_h2.rz(parametro,0)
            circ_h2.barrier([0,1])
            return circ_h2

        #Parte h3-1
        def h3_circ1_2(parametro = params['h3']):        
            circ_h3 = QuantumCircuit(2,2)
            circ_h3.barrier([0,1])
            circ_h3.rzz(parametro, 1, 0)
            circ_h3.barrier([0,1])
            return circ_h3

        #Parte h3-2
        def h3_circ2(parametro = params['h3']):
            return h1_circ_2(-parametro)
        #Parte h3-3
        def h3_circ3(parametro = params['h3']):        
            return h2_circ_2(-parametro)


        #Parte h4
        def h4_circ(parametros=params['h4']):
            circ_h4 = QuantumCircuit(2,2)
            circ_h4.barrier([0,1])
            circ_h4.h(1)
            circ_h4.compose(h1_circ(parametros),inplace=True)#rz? estimar theta com j12?
            circ_h4.h(1)
            circ_h4.barrier([0,1])
            return circ_h4

        #parte h5
        def h5_circ(parametros=params['h5']):
            circ_h5 = QuantumCircuit(2,2)
            circ_h5.barrier([0,1])
            circ_h5.h(0)
            circ_h5.compose(h2_circ(parametros),inplace=True)#rz? estimar theta com j12?
            circ_h5.h(0)
            circ_h5.barrier([0,1])
            return circ_h5

        #Constroi circuito para Hamiltoniana completa
        def ht_circ():
            circ_t = QuantumCircuit(2,2)
            circ_t.compose(h1_circ_2(), inplace=True)
            circ_t.compose(h2_circ_2(), inplace=True)
            circ_t.compose(h3_circ1_2(), inplace=True)
            circ_t.compose(h3_circ2(), inplace=True)
            circ_t.compose(h3_circ3(), inplace=True)
            #circ_t.compose(h4_circ(), inplace=True)
            #circ_t.compose(h5_circ(), inplace=True)
            return circ_t


        #Trotter-suzuki 1 ordem.
        def trotSuzi_1(circ):
            #Cria circuito da trotter suzuki de primeira ordem.
            ts = QuantumCircuit(2,2)
            #_ = list(map(ts.compose(circ, inplace=True),range(n)))
            for a in range(n):
                ts.compose(circ, inplace=True)
            circ_h.compose(ts, inplace=True)
            return

        #Trotter-suzuki 2 ordem.
        def trotSuzi_2(circ):
            #Cria circuito da trotter suzuki de primeira ordem.
            ts = QuantumCircuit(2,2)
            ts.compose(circ, inplace=True)
            ts_rev = ts.reverse_ops()
            ts.compose(ts_rev, inplace=True)
            for a in range(n):
                ts.compose(circ, inplace=True)
            circ_h.compose(ts, inplace=True)
            return 

        #Trotter-suzuki 1 ordem para teste.
        def trotSuzi_1t(circ):
            #Cria circuito da trotter suzuki de primeira ordem.
            ts = QuantumCircuit(2,2)
            #_ = list(map(ts.compose(circ, inplace=True),range(n)))
            for a in range(n):
                ts.compose(circ, inplace=True)
            circ_h.compose(ts, inplace=True)
            return

        #Trotter-suzuki 2 ordem para teste.
        def trotSuzi_2t(circ):
            #Cria circuito da trotter suzuki de primeira ordem.
            ts = QuantumCircuit(2,2)
            for a in range(n):
                ts.compose(circ, inplace=True)
            ts_rev = ts.reverse_ops()
            circ_h.compose(ts, inplace=True)
            circ_h.compose(ts_rev, inplace=True)
            return 

        circ_t = ht_circ()
        circ_h.compose(circ_t,inplace=True)

        #trotSuzi_1(circ_t.assign_parameters([delt_j1,delt_j2,delt_j12,delt_b1,delt_b2]))
        #trotSuzi_2(circ_t.assign_parameters([delt_j1/2,delt_j2/2,delt_j12/2,delt_b1/2,delt_b2/2]))
        circ_h.assign_parameters([delt_j1,delt_j2,delt_j12], inplace=True)
        #circ_h.measure_all(add_bits=False)
        #circ_h.save_expectation_value(Z, [0])
        return circ_h

    #Constroi circuito para medida ZZ
    def measure_ZZ(self):
        circ_zz = QuantumCircuit(2,2)
        circ_zz.measure_all(add_bits=False)
        return circ_zz

    #Constroi circuito para medida YY
    def measure_YY(self):
        circ_yy = QuantumCircuit(2,2)
        circ_yy.rx(pi/2,0) #transforma |y+> em |0> e |y-> em |1> para efetuar medida em X
        circ_yy.rx(pi/2,1)
        circ_yy.measure_all(add_bits=False)
        return circ_yy

    #Constroi circuito para medida XX
    def measure_XX(self):
        circ_xx = QuantumCircuit(2,2)
        circ_xx.h(0)
        circ_xx.h(1)
        circ_xx.measure_all(add_bits=False)
        return circ_xx

    #Constroi os circuitos para execução
    def build_quantum_circ(self,params_l, arrayT):
        print("params:",params_l)
        params_l = [item for sublist in params_l for item in sublist]
        print("params:",params_l)
        circs_l = []
        count = 0
        count2 = 0
        for params in params_l:
            count +=1
            for t in arrayT:
                count2 +=1
                circ_base = self.compose_circ(j1=params[0], j2=params[1], b1=params[2], b2=params[3], j12=params[4], t=t)
                circs_l.append(circ_base.compose(self.measure_XX(), inplace=False))
                circs_l.append(circ_base.compose(self.measure_YY(), inplace=False))
                circs_l.append(circ_base.compose(self.measure_ZZ(), inplace=False))
        return circs_l

    #Coloca em execução no computador quantico a lista de circuitos e aguarda pelo seu resultado.
    def exec_quantum(self,circs):
        #Captura a instancia quantica que está mais desocupada
        backend = self.get_quantum_backend()
        # Use Job Manager to break the circuits into multiple jobs.
        job_manager = IBMQJobManager()
        transpilado = transpile(circs, backend=backend)
        job_set_DPQ = job_manager.run(transpilado, backend=backend, name='DPQ', shots = self.shots, memory=True)

        #Captura os resultados do circuito
        try:
            job_results = job_set_DPQ.results()  # It will block until the job finishes.
            print("The job finished with result {}".format(job_results))
        except JobError as ex:
            print("Something wrong happened!: {}".format(ex))

        job_result = job_results.combine_results()
        return job_result, len(circs)

    #Função que captura amostras até n
    def get_samples(self,l,n):
        return l[:n]

    def qt_to_pandas(self, result, len_result, lenT, colunas, lista_elementos):
        shots = self.shots
        passo_shots = 1000
        linhas = np.array(lista_elementos).reshape((3, len(lista_elementos[0][0])))
        index = ["min", "max", "median"]
        df_l = []
        for m_n in range(passo_shots, self.shots+passo_shots, passo_shots):
            expec_l = [] 
            for r_i in range(len_result):
                #Lista de medidas para cada circuito
                memory = result.get_memory(r_i)
                #Para cada memoria do circuito, selecionamos n samples, com n crescendo a cada iteração.
                samples = self.get_samples(memory, m_n)
                #Conta as medidas
                counts = Counter(samples)
                #Captura as expectativas.
                # expectation value
                expec_l.append((counts['00']+counts['01']-counts['10']-counts['11'])/shots)
                expec_l.append((counts['00']+counts['10']-counts['01']-counts['11'])/shots)
            expec_a = np.array(expec_l).reshape((3,lenT*6))
            expec_a = np.hstack((linhas,expec_a))
            df_l.append(pd.DataFrame(expec_a, index=index, columns = colunas))

        return df_l

    def apply_quantum_data(self, df_l):
        name_rows = ["min", "max", "median"]
        mse_all = []
        for name in name_rows:
            mse = []
            for df in df_l:
                X = df.loc[name][5:]
                y = df.loc[name][4]
                y_pred = self.model.model.predict(np.array(X).reshape(1, 120))
                mse.append(mean_squared_error([y], [y_pred]))
            mse_all.append(mse)
            set_textimg_pdf(name, "Média do erro ao qudrado(MSE):%s"%mse)
            set_textimg_pdf(name, "Mse conforme o numero de shots utilizados:", path =quantum_report(mse, name, self.name, shots=self.shots))
        mse_total = np.zeros(len(mse_all[0]))
        for mse_ in mse_all:
            mse_total = np.array(mse_)+ mse_total
        mse_total = (mse_total/3)
        set_textimg_pdf("Todos", "Média do erro ao qudrado(MSE) para Minimos, maximos e mediana:%s"%mse)
        set_textimg_pdf("Mse conforme o numero de shots utilizados:", path=quantum_report(mse_total, "Minimos, Maximos e medianas, Somados", self.name, shots=self.shots))
        return 

    def elements_to_qt(self, min_tup, max_tup, median_tup):
        elementos_list = [min_tup[1].elementos_iter, max_tup[1].elementos_iter, median_tup[1].elementos_iter]
        #Se os elementos ja foram executados uma vez, captura o resultado.
        if check_quantum(self.name):
            results, circs_len = pickle.load(open(self.path + "/rslts-circs-quanticos", 'rb'))  
        else:
            #Constroi circuitos quânticos com os elementos
            circs = self.build_quantum_circ(elementos_list, min_tup[1].arrayT)

            #Executa circuitos quânticos
            results, circs_len = self.exec_quantum(circs)
            #Salva resultados para evitar re-processamento
            pickle.dump((results,circs_len), open(self.path + "/rslts-circs-quanticos", 'wb'))
        
        #Transforma os resultados quânticos em pandas DataFrames
        #Parametros: Resultados dos circuitos executados,
        # (Numero de elementos de pontos de dados)*(Numero de tempos utilizados)*(Como queremos medir nas três bases, multiplica-se por 3),
        #Numero de tempos utilizado.
        df_l = self.qt_to_pandas(results, circs_len ,len(min_tup[1].arrayT), min_tup[1].getNames(), elementos_list) #Todos os arrayT(array de tempos são iguais)
        
        #Aplica modelo e extrai as métricas
        self.apply_quantum_data(df_l)
        
        return 
        
    def output(self):
        self.pdf.output("../experimentos/%s/RelatórioFinal.pdf"%self.name, "F")
        return