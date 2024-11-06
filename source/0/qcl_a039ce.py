# https://github.com/Andris-Huang/particle_reco/blob/7de146a39b2b38f67b1e0bee872fa5c5cd161a1c/src/models/QCL.py
import importlib
from qiskit import QuantumCircuit, assemble, transpile, Aer, IBMQ
from qiskit.visualization import *
#from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit.utils import QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA, COBYLA
#from qiskit.aqua.components.feature_maps import SecondOrderExpansion, FirstOrderExpansion
#from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, EfficientSU2
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
#from qiskit.aqua.input import ClassificationInput
import numpy as np
import matplotlib.pyplot as plt
#import os

from qiskit.tools.visualization import plot_histogram

utils = importlib.import_module("utils")
base_file = importlib.import_module(f"src.models.base_QML")
Base = base_file.Base

class Model(Base):
    """
    Model Class for QCL
    """
    def __init__(self, config, save_fig=True, overwrite=False, debug=False, hide_display=False):
        super().__init__(config, save_fig, overwrite, debug, hide_display)
        self.base_params = {
            "num_evt": None,
            "num_job": 7,
            "encoder_depth": 3,
            "vqc_depth": 4,
            "num_iter": 4,
            "shots": 1024,
            "optimizer": COBYLA,
            "feature_map": ZZFeatureMap,
            "var_form": EfficientSU2, # ansatz
            "features": None, # Placeholder
            "backend": "aer_simulator"
        }
        self.config = self._process_config(config)
        self.training_input, self.test_input = self.process_data()
        self.model = self.construct_model()
        

    def construct_model(self):
        """
        Construct a VQC circuit model
        """
        # Get Hyperparameters
        jobn = self.config['num_job']
        backend_name = self.config['backend']
        niter = self.config['num_iter']
        feature_dim = len(self.config['features'])
        shots = self.config['shots']
        uin_depth = self.config['encoder_depth']
        uvar_depth = self.config['vqc_depth']
        Optimizer = self.config['optimizer']
        featureMap = self.config['feature_map']
        varForm = self.config['var_form']
        random_seed = 10598+1010*uin_depth+101*uvar_depth+jobn
        training_input = self.training_input
        test_input = self.test_input
        

        # Construct Circuit
        simulator = Aer.get_backend(backend_name)
        optimizer = Optimizer(maxiter=niter, disp=True)
        feature_map = featureMap(feature_dimension=feature_dim, reps=uin_depth)
        var_form = varForm(num_qubits=feature_dim, reps=uvar_depth)
        vqc = VQC(optimizer, feature_map, var_form, training_input, test_input)
        self.quantum_instance = QuantumInstance(simulator, shots=shots, seed_simulator=random_seed, seed_transpiler=random_seed, skip_qobj_validation=True)
        
        return vqc

    
    def train(self):
        """
        Train the QML model by running VQC
        """
        vqc = self.model
        self.result = vqc.run(self.quantum_instance)
        counts = vqc.get_optimal_vector()
        if not self.hide_display:
            print(">>> Counts (w/o noise) =",counts)
        plot = plot_histogram(counts, title='Bit counts (w/o noise)')
        if self.save_fig:
            file_name = f"{self.output_dir}/Bit_counts.png"
            file_path = utils.rename_file(file_name, overwrite=self.overwrite)
            #os.mkdir(file_path)
            plot.savefig(file_path)
        plot.clf()

    def predict(self, input_data):
        return self.model.predict(input_data)

    def evaluate(self):
        datapoints, class_to_label = split_dataset_to_data_and_labels(self.test_input)
        predicted_probs, predicted_labels = self.predict(datapoints[0])
        predicted_classes = map_label_to_class_name(predicted_labels, self.model.label_to_class)

        n_sig = np.sum(datapoints[1]==1)
        n_bg = np.sum(datapoints[1]==0)
        n_sig_match = np.sum(datapoints[1]+predicted_labels==2)
        n_bg_match = np.sum(datapoints[1]+predicted_labels==0)
        
        if not self.hide_display:
            print(">>> Testing success ratio: ", self.result['testing_accuracy'],"(w/o noise)")
            print(">>> Signal eff =",n_sig_match/n_sig, ", Background eff =",(n_bg-n_bg_match)/n_bg, " (w/o noise)")


        from sklearn.metrics import roc_curve, auc, roc_auc_score
        prob_test_signal = predicted_probs[:,1]
        fpr, tpr, thresholds = roc_curve(datapoints[1], prob_test_signal, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='Testing w/o noise (area = %0.3f)' % roc_auc)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

        datapoints_tr, class_to_label_tr = split_dataset_to_data_and_labels(self.training_input)
        predicted_probs_tr, predicted_labels_tr = self.predict(datapoints_tr[0])
        prob_train_signal = predicted_probs_tr[:,1]
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(datapoints_tr[1], prob_train_signal, drop_intermediate=False)
        roc_auc_tr = auc(fpr_tr, tpr_tr)
        plt.plot(fpr_tr, tpr_tr, color='darkblue', lw=2, label='Training w/o noise (area = %0.3f)' % roc_auc_tr)
        plt.legend(loc="lower right")
        if self.save_fig:
            file_name = f"{self.output_dir}/ROC.png"
            plt.savefig(utils.rename_file(file_name, overwrite=self.overwrite))
        plt.clf()
        if not self.hide_display:
            print(f'>>> AUC(training) w/o noise = {roc_auc_tr:.3f}')
            print(f'>>> AUC(testing) w/o noise = {roc_auc:.3f}')
        
        return roc_auc_tr, roc_auc