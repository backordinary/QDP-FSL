# https://github.com/axelschacher/quantum_error_mitigation/blob/c67fc397d568a6d90b88ac76b26f1c73be78a975/quantum_error_mitigation/model/NN_Base_Class/Neural_Network_Base_Class.py
import os
import math
import time
import torch
import qiskit
import locale
import datetime
import requests
import functools
# import matplotlib
import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchtyping import TensorType
from typing import List, Tuple, Optional, Union, Any
from quantum_error_mitigation.data.data_persistence.Data_Persistence import save_object, load_object, save_model, load_model
from quantum_error_mitigation.model.Basic_Dataset.Basic_Dataset import Basic_Dataset
from quantum_error_mitigation.data.information_handler import Information_Handler
from quantum_error_mitigation.visualization import Visualization


class Neural_Network_Base_Class(nn.Module):
    def __init__(self):
        """
        Constructor of class Neural_Network_Base_Class.
        """
        super().__init__()
        # check availability of GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
            print("using GPU")
        else:
            self.device = torch.device("cpu")
            print("\nusing CPU")
        # matplotlib.use('macosx') # TkAgg or agg for linux, macosx for mac

    def load_training_data(self, path: str = "", full_path: Optional[str] = None):
        """
        Loads the generated training data.

        Args:
            path: The folder in ../../data/training_data/ which contains the data to load. Use "", "simulator", or "quantum_computer" or any oyher folder that contains the dataset.

        Returns:
            training_inputs: The input for the NN as prepared in Data_Generator.
            training_solutions: The solution for the NN for each input as prepared in Data_Generator.
        """
        if full_path is not None:
            data_path = full_path
        else:
            data_path = os.path.relpath("../../data/training_data/" + path + "/")
        self.data_path = data_path
        training_inputs = load_object(
            os.path.join(data_path, "training_inputs.pkl"))
        training_solutions = load_object(
            os.path.join(data_path, "training_solutions.pkl"))
        if np.count_nonzero(training_solutions[:, 1:, :]) == 0:  # check if there is only one goal state in the training_data --> method is calibration_bits
            self.data_generator_method = "calibration_bits"
        else:
            self.data_generator_method = "rotation_angles"
        return training_inputs, training_solutions

    def prepare_training_data(self, training_inputs: npt.NDArray, training_solutions: npt.NDArray, inputs_for_NN: str, split_indices: Optional[list] = None, output_size: Optional[str] = "full"):
        """
        Transforms raw data from data generator to inputs and outputs used for the Neural Network.

        Args:
            training_inputs: Array of shape [num_samples, n_different_measurements, n_features (bitstring(n_qubits)+probability+integer_value_of_bitstring)] containing the pre-computed states, probability and integer value of the corresponding register for each sample.
            training_solutions: Array of shape [num_samples, number_of_considered_states, n_qubits+2] containing the same features (their theoretical probability) as training_inputs for the number_of_considered_states for each samples.
            inputs_for_NN: How the neurons of the NN are assigned with. Possible values are 'probability_for_each_state' (the neurons correspond to the integer value of the bitstring and their value is their probablity) or 'k_most_frequent_states' (the k most frequent states are coded in an input of size [k*n_qubits]).
            split_indices: list of lists containing the indices of qubits that are clustered together for the clustered version of the FCN with 'probability_for_each_state' inputs.
            output_size: str: whether the full probability distribution is the desired NN input or the subregister(s).

        Returns:
            datasets_train: the prepared dataset used for training.
            datasets_val: the prepared dataset used for validation.
        """
        assert output_size == "full" or output_size == "subregisters"
        self.n_qubits = training_inputs.shape[2] - 2
        self.split_indices = split_indices
        n_samples = training_inputs.shape[0]
        if inputs_for_NN == 'probability_for_each_state':
            # inputs are here the probabilities for each measured outcome. It is a vector and the first entry is p(0).
            X, y = self._make_probability_for_each_state_features(training_inputs, training_solutions, self.n_qubits, n_samples, output_size, split_indices)
        if inputs_for_NN == 'k_most_frequent_states':
            # inputs are coded as the probability of 'measuring a 1' for this qubit. There exist k groups, one group codes one measured state. If we measure a '1' for a specific state, the corresponding neuron is set to the probability, if we measure a '0', the input is the negative probability.
            X, y = self._make_k_most_frequent_state_features(training_inputs, training_solutions, self.n_qubits, n_samples, output_size, self.k)
        self._prepare_unmitigated_dataset_with_no_splits(training_inputs, training_solutions, self.n_qubits, n_samples)
        X_train, X_val, y_train, y_val = self._train_val_split(X, y)
        datasets_train, datasets_val = self._make_Basic_Datasets(X_train, X_val, y_train, y_val)
        return datasets_train, datasets_val

    def train_model(model, datasets_train: list[Basic_Dataset], datasets_val: Optional[list[Basic_Dataset]] = None, evaluation: bool = True, evaluation_path: Optional[str] = None) -> None: # self is here called model to make it clearer
        """
        Main method to train a model.
        Uses the models parameters to get the required hyperparameters.
        Updates the model itself (its weights).

        Args:
            datasets_train: list[Basic_Dataset]: training dataset for each split.
            datasets_val: list[Basic_Dataset]: validation dataset for each split.
            evaluation: bool: Whether the model is evaluated regularly or not.
            evaluation_path: str: Path where the evaluation files (Plots) are stored.
        """
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)
        evaluation_epochs = []
        training_time = 0.
        for epoch in range(model.epochs):
            model.current_epoch = epoch
            epoch_loss, time_for_epoch = model.train_one_epoch(
                datasets_train, optimizer)  # Optimizer arguments are updated because the optimizer is call-by-reference
            training_time += time_for_epoch

            if evaluation:
                assert datasets_val is not None
                if (epoch == 0):
                    # print(f"This model has {model._count_parameters()} trainable parameters.")
                    # print(f"It has {model.num_input_nodes} input nodes and {model.num_output_nodes} output nodes.")
                    # print(f"The dataset contains {len(dataset_train)} training and {len(dataset_val)} evaluation samples, each of size {dataset_train.__getitem__(idx=0)['X'].shape}.")
                    # print(f"\nEpoch {epoch+1}, train loss = {epoch_loss:.6f}, current time = {datetime.datetime.now()}, elapsed time per Epoch: {(time_for_epoch):.5f} s.")
                    training_losses = []
                    validation_losses = []
                    training_losses, validation_losses = model.evaluate_model(datasets_val, datasets_train, training_losses, validation_losses, epoch, evaluation_path)
                    evaluation_epochs.append(epoch)
                    # save_model(model)

                if (((epoch + 1) % model.evaluation_frequency_epochs) == 0):
                    print(f"\nEpoch {epoch+1}, train loss = {epoch_loss:.8f}, current time = {datetime.datetime.now()}, elapsed time per Epoch: {training_time/model.evaluation_frequency_epochs:.5f} s.")
                    training_losses, validation_losses = model.evaluate_model(datasets_val, datasets_train, training_losses, validation_losses, epoch, evaluation_path)
                    evaluation_epochs.append(epoch)
                    training_time = 0.
                    save_model(model)
        save_object(evaluation_epochs, 'evaluation_epochs.pkl')

    def train_one_epoch(model, datasets_train: list[Basic_Dataset], optimizer: optim) -> None:
        """
        Trains the model one epoch using the training datasets and the given optimizer.

        Args:
            datasets_train: list[Basic_Dataset]: the training datasets.
            optimizer: optim: the optimizer to compute the parameter updates.
        """
        model.train()
        start_time = time.time()
        batches, _ = model._sample_the_dataset(datasets_train)
        epoch_loss = 0.
        for batch_idx in range(len(batches[0])):
            model.zero_grad()
            X_input, y_true = model._load_NN_IO(batches, batch_idx)
            y_hat = model.forward(X_input)
            # batch_loss = F.mse_loss(y_hat, y_true.float().to(model.device))
            batch_loss = model.m_rel_se_loss(y_hat, y_true.float().to(model.device))
            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        end_time = time.time()
        time_for_epoch = end_time - start_time
        return epoch_loss, time_for_epoch

    def m_rel_se_loss(model, y_hat, y_true):
        """
        Computes mean relative squared error loss.
        """
        n_samples = y_hat.shape[0]
        len_per_sample = y_hat.shape[1]
        error = torch.norm((y_true - y_hat) / y_true, 2) ** 2
        return error / (n_samples*len_per_sample)

    def convert_khp_features_to_probability_vector(model, features: TensorType) -> TensorType:
        """
        Decodes the output coded with the KHP NN into a probability distribution.
        Keep in mind that the in- and output features contain variing states, not only variing probabilities.

        Args:
            features: The output of the KHP NN containing the mitigated information. It is of shape [model.batch_size, model.k*n_qubits]
        """
        y = torch.zeros((features.shape[0], 2**model.n_qubits))
        for sample in range(y.shape[0]):
            for i in range(model.k):
                start_idx = i*model.n_qubits
                end_idx = (i+1)*model.n_qubits
                feature_vector_for_one_state = features[sample, start_idx:end_idx]
                register = torch.zeros(model.n_qubits)
                prob = 0.
                for j, val in enumerate(feature_vector_for_one_state):
                    if val > 0:
                        register[j] = 1
                    else:
                        register[j] = 0
                    prob += torch.abs(val)  # The output probability is the average of all absolute values.
                int_value = Information_Handler.quantum_register_to_integer(register, big_endian=True)
                y[sample, int_value] = prob/model.n_qubits
        y = F.normalize(y, p=1.0, dim=1)
        return y

    def evaluate_model(model, datasets_val: list[Basic_Dataset], datasets_train: list[Basic_Dataset], training_losses: List, validation_losses: List, epoch: int, evaluation_path) -> Tuple[List, List]:  # self is here called model to make it clearer
        """
        Evaluates the model on the different datasets by computing different performance measures.
        Results are stored to the knowledge database and visualized.
        """
        model.eval()
        with torch.no_grad():
            losses = model.compute_performance_measures(datasets_train, datasets_val)
            training_loss = losses['mse_train']
            validation_loss = losses['mse_val']
            training_losses.append([epoch, training_loss])
            validation_losses.append([epoch, validation_loss])
            model.add_losses_to_knowledge_database(losses, evaluation_path)
            Visualization.plot_performance_measures(evaluation_path)
            model.plot_learning_progress(training_losses, validation_losses, evaluation_path)
        return training_losses, validation_losses

    def add_losses_to_knowledge_database(model, losses: dict, evaluation_path: str):
        """
        Stores all losses for the model into the knowledge database.
        """
        for metric, loss in losses.items():
            Information_Handler.add_sample_to_knowledge_database({'model_name': model.model_name, 'metric': metric, 'n_qubits': model.n_qubits, 'n_epochs': model.current_epoch+1, 'loss': loss}, evaluation_path)

    def learning_rate_range_test(model, datasets_train: list[Basic_Dataset], datasets_val: list[Basic_Dataset]) -> None:
        """
        Tests the training progress for differen learning rates.
        Used to find an optimal learning rate for each model.
        """
        num_of_tested_lrs = 50
        train_losses = np.zeros((num_of_tested_lrs, 2))
        validation_losses = np.zeros((num_of_tested_lrs, 2))
        results = [train_losses, validation_losses]
        models_output_size = model.output_size
        models_y_measured = model.y_measured
        data_path = model.data_path
        n_qubits = model.n_qubits
        input_type = model.input_type
        for i, learning_rate in enumerate(np.logspace(-7, 0, num=num_of_tested_lrs, base=10.0)):
            model = model.new_instance()
            model.epochs = 100
            model.learning_rate = learning_rate
            model.output_size = models_output_size
            model.y_measured = models_y_measured
            model.data_path = data_path
            model.n_qubits = n_qubits
            model.input_type = input_type
            print(f"test learning rate {learning_rate}")
            model.train_model(datasets_train, evaluation=False)

            losses = model.compute_performance_measures(datasets_train, datasets_val)
            training_loss = losses['mse_train']
            validation_loss = losses['mse_val']
            results[0][i, 0] = learning_rate
            results[0][i, 1] = training_loss
            results[1][i, 0] = learning_rate
            results[1][i, 1] = validation_loss
        # plot results
        fig, ax = plt.subplots()
        ax.plot(results[0][:, 0], results[0]
                [:, 1], 'k--', label='train loss')
        ax.plot(results[1][:, 0], results[1][:, 1],
                'k', label='validation loss')
        ax.set_xscale('log')  # alternatively 'linear'
        ax.set_yscale('log')
        ax.legend(loc='upper left')
        plt.xlabel('learning rate [-]')
        plt.ylabel(f'loss after {model.epochs} epochs [-]')
        plt.tight_layout()
        plt.savefig(os.path.join('Plots/LR_Test/lr_test.pdf'))
        plt.close()

    def evaluate_on_another_dataset(model, datasets_train: List[Basic_Dataset], datasets_val: List[Basic_Dataset], evaluation_path: Optional[str] = None):
        """
        Evaluates a model on a different dataset that was not passed to it as training or validation set.
        """
        model_files = os.listdir("./Trained_model")
        model_files.remove(".DS_Store")
        model_files = sorted(model_files, key=functools.cmp_to_key(locale.strcoll))
        evaluation_epochs = load_object("evaluation_epochs.pkl")
        for i, file in enumerate(model_files):
            model2 = load_model(model, name=file)
            epoch = evaluation_epochs[i]
            model2.model_name = model.model_name + "_other_dataset"
            model2.current_epoch = epoch
            if (epoch == 0):
                training_losses = []
                validation_losses = []
                training_losses, validation_losses = model2.evaluate_model(datasets_val, datasets_train, training_losses, validation_losses, epoch, evaluation_path)
                Visualization.plot_double_dataset_evaluation_data_generation_method(model.model_name, model2.model_name, evaluation_path)
            else:
                training_losses, validation_losses = model2.evaluate_model(datasets_val, datasets_train, training_losses, validation_losses, epoch, evaluation_path)
                Visualization.plot_double_dataset_evaluation_data_generation_method(model.model_name, model2.model_name, evaluation_path)


    def plot_learning_progress(self, training_losses: List, validation_losses: List, evaluation_path: str) -> None:
        """
        Plots training and validation mse to easily track the learning progress.
        """
        fig, ax = plt.subplots()
        training_losses_array = np.array(training_losses)
        validation_losses_array = np.array(validation_losses)
        ax.plot(training_losses_array[:, 0], training_losses_array
                [:, 1], 'k--', label='training loss')
        ax.plot(validation_losses_array[:, 0], validation_losses_array[:, 1],
                'k', label='validation loss')
        ax.legend(loc='upper right')
        plt.xlabel('trained epochs [-]')
        plt.ylabel('loss [-]')
        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_path, 'training_progress.pdf'))
        plt.close()

    def compute_performance_measures(model, datasets_train: list[Basic_Dataset], datasets_val: list[Basic_Dataset]) -> dict:
        """
        Compute unmitigated, training and validation performance measures for the model.
        Measures are computed for unmitigated, linear inversion, tpnm and neural network error mitigation models.
        Performance measures are MSE, Kullback-Leibler (KL) divergence and Infidelity
        """
        model.eval()
        global unmitigated_loss
        global linear_inversion_loss
        global tpnm_loss
        try:
            unmitigated_loss = None if unmitigated_loss is None else unmitigated_loss
            linear_inversion_loss = None if linear_inversion_loss is None else linear_inversion_loss
            tpnm_loss = None if tpnm_loss is None else tpnm_loss
        except NameError:
            unmitigated_loss = None
            linear_inversion_loss = None
            tpnm_loss = None

        with torch.no_grad():
            if unmitigated_loss is None:
                mse_unmitigated, kl_div_unmitigated, if_unmitigated = model.compute_performance_measures_on_dataset(datasets_train, mitigation_method="unmitigated", dataset_name='train')
                unmitigated_loss = [mse_unmitigated, kl_div_unmitigated, if_unmitigated]
            else:
                mse_unmitigated, kl_div_unmitigated, if_unmitigated = unmitigated_loss

            if linear_inversion_loss is None:
                mse_li, kl_div_li, if_li = model.compute_performance_measures_on_dataset(datasets_val, mitigation_method="linear_inversion", dataset_name='val')
                linear_inversion_loss = [mse_li, kl_div_li, if_li]
            else:
                mse_li, kl_div_li, if_li = linear_inversion_loss

            if tpnm_loss is None:
                mse_tpnm, kl_div_tpnm, if_tpnm = model.compute_performance_measures_on_dataset(datasets_val, mitigation_method="tpnm", dataset_name='val')
                tpnm_loss = [mse_tpnm, kl_div_tpnm, if_tpnm]
            else:
                mse_tpnm, kl_div_tpnm, if_tpnm = tpnm_loss
            mse_train, kl_div_train, if_train = model.compute_performance_measures_on_dataset(datasets_train, mitigation_method="neural_network")
            mse_val, kl_div_val, if_val = model.compute_performance_measures_on_dataset(datasets_val, mitigation_method="neural_network")
            losses_dict = {'mse_unmitigated':mse_unmitigated, 'kl_div_unmitigated':kl_div_unmitigated, 'if_unmitigated':if_unmitigated, 'mse_train':mse_train, 'kl_div_train':kl_div_train, 'if_train':if_train, 'mse_val':mse_val, 'kl_div_val':kl_div_val, 'if_val':if_val, 'mse_li':mse_li, 'kl_div_li':kl_div_li, 'if_li':if_li, 'mse_tpnm':mse_tpnm, 'kl_div_tpnm':kl_div_tpnm, 'if_tpnm':if_tpnm}
        return losses_dict

    def compute_performance_measures_on_dataset(model, datasets: list[Basic_Dataset], mitigation_method: str, dataset_name: Optional[str] = None):
        """
        Computes (averaged) mean-squared-error, Kullback-Leibler-Divergence and infidelity-loss on the given dataset.
        """
        batches, batch_sampler_indices = model._sample_the_dataset(datasets)
        mse_loss_on_dataset = 0.
        kld_loss_on_dataset = 0.
        infidelity_loss_on_dataset = 0.
        total_samples = 0
        for batch_idx in range(len(batches[0])):
            if mitigation_method == "unmitigated":
                X_input, y_true = model._load_NN_IO(batches, batch_idx, y_true_type = "Tensor", is_evaluation=True)
                y_hat = model.get_unmitigated_data(batch_sampler_indices, batch_idx, dataset_name=dataset_name)
            elif mitigation_method == "linear_inversion":
                # The measured (unmitigated) data is mitigated with the QuAntiL REM service
                X_input_for_rem_service = model.get_unmitigated_data(batch_sampler_indices, batch_idx, dataset_name=dataset_name)
                y_hat = model.mitigate_with_service(X_input_for_rem_service, cm_gen_method = "standard")
                _, y_true = model._load_NN_IO(batches, batch_idx, y_true_type = "Tensor", is_evaluation=True)
            elif mitigation_method == "tpnm":
                # The measured (unmitigated) data is mitigated with the QuAntiL REM service
                X_input_for_rem_service = model.get_unmitigated_data(batch_sampler_indices, batch_idx, dataset_name=dataset_name)
                y_hat = model.mitigate_with_service(X_input_for_rem_service, cm_gen_method = "tpnm")
                _, y_true = model._load_NN_IO(batches, batch_idx, y_true_type = "Tensor", is_evaluation=True)
            elif mitigation_method == "neural_network":
                X_input, y_true = model._load_NN_IO(batches, batch_idx, y_true_type = "Tensor", is_evaluation=True)
                y_hat = model.forward(X_input)
                if model.input_type == "subregisters":
                    y_hat = model.convert_khp_features_to_probability_vector(y_hat)
            else:
                raise ValueError(f"Mitigation method {mitigation_method} not supported. Please choose between 'unmitigated', 'linear_inversion', 'tpnm' or 'neural_network'")
            total_samples += batches[0][batch_idx]['X'].shape[0]
            batch_mse_loss = F.mse_loss(y_hat, y_true.float().to(model.device), reduction='sum')
            y_hat_eps = y_hat+1e-5
            batch_kld_loss = F.kl_div(torch.log(y_hat_eps), y_true.float().to(model.device), reduction='sum')
            batch_if_loss = model.infidelity_loss(y_hat, y_true.float().to(model.device), reduction='sum')
            mse_loss_on_dataset += batch_mse_loss.item()
            kld_loss_on_dataset += batch_kld_loss.item()
            assert batch_kld_loss.item() < np.inf
            infidelity_loss_on_dataset += batch_if_loss
        mse_loss_on_dataset = mse_loss_on_dataset / total_samples
        kld_loss_on_dataset = kld_loss_on_dataset / total_samples
        infidelity_loss_on_dataset = infidelity_loss_on_dataset / total_samples
        return mse_loss_on_dataset, kld_loss_on_dataset, infidelity_loss_on_dataset

    def nn_features_to_qiskit_counts(model, X_for_one_sample: TensorType) -> Any:
        """
        Converts inputs for one sample into qiskit counts.
        """
        assert model.input_type == "full" or model.input_type == "subregisters"
        if model.input_type == "subregisters":
            X_for_one_sample = model.convert_khp_features_to_probability_vector(X_for_one_sample.reshape(1, -1)).reshape(-1)
        counts = {}
        for register_value, prob in enumerate(X_for_one_sample):
            bitstring = Information_Handler.integer_value_to_classical_register(register_value, n_bits=model.n_qubits, big_endian=False)
            bitstring = ''.join(str(int) for int in bitstring)
            counts[bitstring] = int(np.rint(model.shots*prob.detach().cpu().numpy()))
        counts = qiskit.result.Counts(counts)
        return counts

    def qiskit_counts_to_NN_input(model, counts: Any, split_indices: Optional[list] = None) -> TensorType:
        """
        Converts qiskit counts into the fearures needed for the NNs.
        """
        training_inputs = model._qiskit_counts_to_training_array(counts)
        training_solutions = training_inputs # no solutions given, so we don't need them
        n_samples = 1
        if model.inputs_for_NN == "probability_for_each_state":
            model.input_size = "full"
            output_size = "full"
            X, _ = model._make_probability_for_each_state_features(training_inputs, training_solutions, model.n_qubits, n_samples, output_size, split_indices)
        elif model.inputs_for_NN == "k_most_frequent_states":
            output_size = "subregisters"
            X, _ = model._make_k_most_frequent_state_features(training_inputs, training_solutions, model.n_qubits, n_samples, output_size, model.k)
        X = np.array(X)
        X = torch.FloatTensor(X)
        return X

    def mitigate_counts(model, counts: Any) -> Any:
        """
        Runs the NN mitigation on a given qiskit Counts object.
        """
        X_input = model.qiskit_counts_to_NN_input(counts)
        mitigated_result = model.forward(X_input)
        mitigated_result_counts = model.nn_features_to_qiskit_counts(mitigated_result[0, :])
        return mitigated_result_counts

    def mitigate_with_service(model, X_input: TensorType, cm_gen_method: str) -> TensorType:
        """
        Mitigates measurements using the quantum error mitigation service.
        The service is provided via GitHub (https://github.com/UST-QuAntiL/error-mitigation-service) and has to run to be accessible.
        """
        mitigation_service = "http://127.0.0.1:5071/"
        y_hat = torch.zeros((X_input.shape[0], X_input.shape[1]))
        job_properties = load_object(os.path.join(model.data_path, "job_properties.pkl"))
        model.shots = job_properties["shots"]
        for sample_idx in range(X_input.shape[0]):
            counts = model.nn_features_to_qiskit_counts(X_input[sample_idx, :])
            request = model.generate_mitigation_service_request(counts, cm_gen_method)
            response = requests.post(mitigation_service+"rem/", json=request)
            mitigated_counts = response.json()
            for mitigated_output, mitigated_count in mitigated_counts.items():
                int_value = Information_Handler.quantum_register_to_integer(mitigated_output, big_endian=False)
                prob = mitigated_count / model.shots
                y_hat[sample_idx, int_value] = prob
        # set unphysical probabilities to zero and normalize to total probability 1
        y_hat[y_hat<0] = 0.
        y_hat = F.normalize(y_hat, p=1.0, dim=1)
        return y_hat

    def get_unmitigated_data(model, batch_sampler_indices: list[int], batch_idx, dataset_name: str):
        """
        Returns the measured (unmitigated) probability distribution, according to the index of the samples in the batch.
        Works for training dataset as well as validation dataset.
        """
        assert dataset_name == 'train' or dataset_name == 'val'
        batch_indices = batch_sampler_indices[batch_idx]
        if dataset_name == 'val':
            batch_indices = [idx+model.n_train for idx in batch_indices]
        y_hat = torch.tensor(model.y_measured[batch_indices, :])
        return y_hat

    def infidelity_loss(self, y_hat: list[TensorType], y_true: list[TensorType], reduction: str = "sum"):
        """
        Computes averaged infidelity.
        Predicted and true values have to be a probability distribution to give meaningful results.
        See Leymann2020, p.16 for a definition of fidelity.
        """
        if not isinstance(y_hat, list):
            y_hat = [y_hat]
        if not isinstance(y_true, list):
            y_true = [y_true]
        assert len(y_hat) == len(y_true)
        assert y_hat[0].shape == y_true[0].shape
        total_infidelity = 0.
        total_samples = 0
        for split_idx in range(len(y_hat)):
            y_hat_split = y_hat[split_idx]
            y_true_split = y_true[split_idx]
            for sample_idx in range(y_hat_split.shape[0]):
                if_sum_split = 0.
                for register_value in range(y_hat_split.shape[1]):
                    if_sum_split += math.sqrt(y_true_split[sample_idx, register_value] * y_hat_split[sample_idx, register_value])
                infidelity = 1 - (if_sum_split ** 2)
                total_infidelity += infidelity
                if split_idx == 0:
                    total_samples += 1
        total_infidelity = total_infidelity / len(y_hat)  # average
        # reduce loss to one value by computing the arithmetic mean or sum everything up
        if reduction == "sum":
            infidelity = total_infidelity
        elif reduction == "mean":
            infidelity = total_infidelity / total_samples
        else:
            raise ValueError(f"Reduction method {reduction} not supported.")
        return infidelity


    def load_model_weights_from_file(model, *args):
        """
        Loads a (pretrained) model with weights from a file.
        If no argument is passed, the model named 'model_state_dict_trained.pth' is loaded, else other models can be chosen.
        """
        return load_model(model, *args)

    @classmethod  # classmethods are inherited from subclasses
    def new_instance(cls):
        """
        Creates a new instance of the class, i.e. a new model with default parameters.
        """
        return cls()

    def _train_val_split(model, X: list[npt.NDArray], y: list[npt.NDArray]) -> Tuple[list[npt.NDArray], list[npt.NDArray], list[npt.NDArray], list[npt.NDArray]]:
        """
        Accepts unsplitted (the full reigster) or splitted data and splits them into training and validation set.
        If the lists X,y contain only 1 element (the whole dataset), the output also contains only 1 element per list.

        Args:
            X: list[npt.NDArray]: The inputs.
            y: list[npt.NDArray]: The outputs.

        Returns:
            X_train: [list[npt.NDArray]: The training inputs.
            X_val: [list[npt.NDArray]: The validation inputs.
            y_train: [list[npt.NDArray]: The training solutions.
            y_val: [list[npt.NDArray]: The validation solutions.
        """
        X_train = []
        X_val = []
        y_train = []
        y_val = []
        num_register_splits = len(X)
        for subregister_idx in range(num_register_splits):
            X_train_sub = X[subregister_idx][0:model.n_train, :]
            X_val_sub = X[subregister_idx][model.n_train:, :]
            X_train.append(X_train_sub)
            X_val.append(X_val_sub)
            if model.output_size == "subregisters":
                y_train_sub = y[subregister_idx][0:model.n_train, :]
                y_val_sub = y[subregister_idx][model.n_train:, :]
            elif model.output_size == "full":
                y_train_sub = y[0][0:model.n_train, :]
                y_val_sub = y[0][model.n_train:, :]
            y_train.append(y_train_sub)
            y_val.append(y_val_sub)
        return X_train, X_val, y_train, y_val

    def _make_Basic_Datasets(model, X_train: list[npt.NDArray], X_val: list[npt.NDArray], y_train: list[npt.NDArray], y_val: list[npt.NDArray]):
        """
        Converts the feature lists into Basic_Datasets.

        Returns:
            dataset_train: the dataset used for training.
            dataset_val: the dataset used for validation.
        """
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        datasets_train = []
        datasets_val = []
        num_register_splits = len(X_train)
        for subreg_idx in range(num_register_splits):
            dataset_train = Basic_Dataset(X_train[subreg_idx], y_train[subreg_idx])
            dataset_val = Basic_Dataset(X_val[subreg_idx], y_val[subreg_idx])
            datasets_train.append(dataset_train)
            datasets_val.append(dataset_val)
        return datasets_train, datasets_val

    def _make_k_most_frequent_state_features(model, training_inputs:npt.NDArray, training_solutions:npt.NDArray, n_qubits: int, n_samples: int, output_size: str, k: int = 100) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Stores the measured and expected probabilities for each sample (executed circuit) in X and y, respecively.
        The in- and outputs have the same structure.
        Each input consists of k groups of size n_qubits as a vector.
        Each group codes one sample.
        Measured '0'is represented by a -p, and measured '1' is represented by the probability p of measuring the corresponding state.
        Example: We measure 10110 with p=0.75. Then the group is coded as [0.75, -0.75, 0.75, 0.75, -0.75].
        The group with the highest probability is coming first until the k-highest group is reached.
        """
        model.output_size = output_size
        model.input_type = "subregisters"
        X_list = []
        X = np.zeros((n_samples, k*n_qubits))
        for sample in range(n_samples):
            for i in range(k):
                neuron_group = np.zeros(n_qubits)
                prob = training_inputs[sample, i, -2]
                for j in range(n_qubits):
                    if training_inputs[sample, i, j] == 1:
                        neuron_group[j] = prob
                    else:
                        neuron_group[j] = -prob
                X[sample, n_qubits*i: n_qubits*(i+1)] = neuron_group
        X_list.append(X)

        y_list = []
        y = np.zeros((n_samples, k*n_qubits))
        for sample in range(n_samples):
            for i in range(k):
                neuron_group = np.zeros(n_qubits)
                prob = training_solutions[sample, i, -2]
                for j in range(n_qubits):
                    if training_solutions[sample, i, j] == 1:
                        neuron_group[j] = prob
                    else:
                        neuron_group[j] = -prob
                y[sample, n_qubits*i: n_qubits*(i+1)] = neuron_group
        y_list.append(y)
        return X_list, y_list

    def _make_probability_for_each_state_features(model, training_inputs: npt.NDArray, training_solutions: npt.NDArray, n_qubits: int, n_samples: int, output_size: str, split_indices: Optional[list] = None) -> Tuple[list[npt.NDArray], list[npt.NDArray]]:
        """
        Stores the measured and expected probabilities for each sample (executed circuit) in X and y, respecively.
        Each index stands for the register with that's index value.
        To split, we can choose indices to seperate the register into subregisters.
        Then, each index stands for the subregister with that's index value.
        To get the total register then, we can insert each bit at the position where it belongs to.
        """
        assert output_size == "full" or output_size == "subregisters"
        model.output_size = output_size
        model.input_type = "full"
        if split_indices is None:
            split_indices = [np.arange(n_qubits)]
        X = []
        for subindices in split_indices: # loop over all clusters
            X_sub = np.zeros((training_inputs.shape[0], 2 ** len(subindices)))
            for sample_idx in range(training_inputs.shape[0]):  # iterate over all samples
                for measured_outcome in range(training_inputs.shape[1]):
                    prob, int_value = model._get_sample(training_inputs, sample_idx, measured_outcome, subindices)  # int_value is based on the sub_register
                    X_sub[sample_idx, int_value] += prob
            X.append(X_sub)
        y = []
        if model.output_size == "subregisters":
            for subindices in split_indices: # loop over all "extra NNs"
                y_sub = np.zeros((training_inputs.shape[0], 2 ** len(subindices)))
                for sample_idx in range(training_inputs.shape[0]):  # iterate over all samples
                    y_sub = model._add_prob_for_value_to_y_sub(y_sub, training_solutions, sample_idx, subindices)
                y.append(y_sub)
        elif model.output_size == "full":
            y_array = np.zeros((training_inputs.shape[0], 2 ** n_qubits))
            for sample_idx in range(training_inputs.shape[0]):
                y_array = model._add_prob_for_value_to_y_sub(y_array, training_solutions, sample_idx, np.arange(n_qubits))
            y.append(y_array)
        return X, y

    def _prepare_unmitigated_dataset_with_no_splits(model, training_inputs: npt.NDArray, training_solutions, n_qubits, n_samples):
        """
        Stores the measured data into the model.
        Used for computing the unmitigated losses.
        """
        # compute "y_hat" for unmitigated loss
        y_measured = np.zeros((training_inputs.shape[0], 2 ** n_qubits))
        for sample_idx in range(training_inputs.shape[0]):
            for measured_outcome in range(training_inputs.shape[1]):
                prob, int_value = model._get_sample(training_inputs, sample_idx, measured_outcome, np.arange(n_qubits))
                y_measured[sample_idx, int_value] += prob  # += so that later coming zero values don't overwrite information
        model.y_measured = y_measured

    def _get_sample(model, training_array: npt.NDArray, sample_idx: int, state_idx: int, subindices: Union[list, npt.ArrayLike]):
        """
        Returns the probability and value of a sample given by index, state index and cluster subindices.
        """
        prob = training_array[sample_idx, state_idx, -2]
        value = 0
        for subregister_idx, qubit_idx in enumerate(subindices):
            value += Information_Handler.get_value_of_qubit(bit=training_array[sample_idx, state_idx, qubit_idx], index=subregister_idx, big_endian=True)
        int_value = int(value)
        return prob, int_value

    def _add_prob_for_value_to_y_sub(model, y_sub: npt.NDArray, training_solutions: npt.NDArray, sample_idx: int, subindices: list) -> npt.NDArray:
        """
        Adds the probabiltiy to y_sub at the corresponding postition for the sample.
        """
        for goal_state in range(training_solutions.shape[1]):
            prob, int_value = model._get_sample(training_solutions, sample_idx, goal_state, subindices)
            y_sub[sample_idx, int_value] += prob
        return y_sub

    def _count_parameters(model) -> int:
        """
        Counts all trainable parameters.
        Issue: Counts shared weights more than once.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _sample_the_dataset(model, datasets: list[Basic_Dataset]) -> Tuple[list, list]:
        """
        Creates a list of batches and the corresponding sample indices.
        Ensures that each dataset in datasets contains the same indices at the same position.
        """
        random_sampler = data.RandomSampler(data_source=datasets[0])
        batch_sampler = data.BatchSampler(sampler=random_sampler, batch_size=model.batch_size, drop_last=False)
        batch_sampler_indices = list(batch_sampler)  # side effect: every call of batch_sampler (also as an argument) returns new indices
        data_loaders = model._get_data_loaders(datasets, batch_sampler_indices)
        batches = model._get_batches_from_data_loaders(data_loaders)
        return batches, batch_sampler_indices

    def _get_data_loaders(model, datasets: list[Basic_Dataset], batch_sampler_indices: list[int]) -> list[data.DataLoader]:
        """
        Returns the DataLoaders with the samples from batch_sampler.
        Needed to access the DataLoaders with the same batch for each dataset (containing to a split).
        """
        data_loaders = []
        for split_idx in range(len(datasets)):
            data_loader = data.DataLoader(
                dataset=datasets[split_idx], batch_sampler=batch_sampler_indices)
            data_loaders.append(data_loader)
        return data_loaders

    def _get_batches_from_data_loaders(model, data_loaders: list[data.DataLoader]) -> list:
        """
        Returns all batches as a list of lists for multiple data_loaders, all created from the given data loaders.

        Returns:
            batches: (nested) list of length number_of_data_loaders each containing all batches in the right order.
        """
        batches = []
        for i in range(len(data_loaders)):
            data_loader = data_loaders[i]
            dl_batches = []
            for batch in data_loader:
                dl_batches.append(batch)
            batches.append(dl_batches)
        return batches

    def _load_NN_IO(model, batches: list, batch_idx: int, y_true_type: str = "Tensor", is_evaluation: bool = False) -> Tuple[list[TensorType], TensorType]:
        """
        Computes the in- and output for the neural network.

        Returns:
            X_input: list[TensorType] that contains the sub-inputs for each NN
            y_true: Tensor that contains the full reference-solution.
        """
        assert y_true_type == "Tensor" or y_true_type == "list"
        assert model.input_type == "full" or model.input_type == "subregisters"
        y_true = []
        X_input = []
        for split_idx in range(len(batches)):
            sub_batch = batches[split_idx][batch_idx]
            X_input.append(sub_batch['X'].float().to(model.device))
            y_true.append(batches[split_idx][batch_idx]['y'])
        if y_true_type == "Tensor":
            if model.output_size == "full":  # all y's are the same, so we're only interested in one of them
                y_true = y_true[0]
            elif model.output_size == "subregisters":
                y_true = torch.cat(y_true, dim=1)
        if is_evaluation and model.input_type == "subregisters":
            y_true = model.convert_khp_features_to_probability_vector(y_true)
        return X_input, y_true

    def _qiskit_counts_to_training_array(model, counts: Any) -> npt.NDArray:
        """
        Converts qiskit counts into the state we use to process our data.
        """
        sorted_counts = sorted(counts, key=counts.get, reverse=True)
        n_qubits = len(sorted_counts[0])
        model.n_qubits = n_qubits
        model.shots = counts.shots()
        training_array = np.zeros((1, len(sorted_counts), n_qubits+2))
        for j, measured_outcome in enumerate(sorted_counts):
            big_endian_measured_outcome = measured_outcome[::-1]
            measured_outcome_integer = Information_Handler.quantum_register_to_integer(big_endian_measured_outcome, big_endian=True)
            n_qubits = len(big_endian_measured_outcome)
            for k, bit in enumerate(big_endian_measured_outcome):
                training_array[0, j, k] = bit
            training_array[0, j, n_qubits] = counts[measured_outcome]/model.shots
            training_array[0, j, n_qubits+1] = measured_outcome_integer
        return training_array
