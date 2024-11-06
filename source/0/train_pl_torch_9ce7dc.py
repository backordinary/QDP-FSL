# https://github.com/QQQYang/Shuffle-QUDIO/blob/bba4b77de7a948c4ac66099723e21159cbd4a0e0/train_pl_torch.py
"""
https://pytorch.org/tutorials/intermediate/dist_tuto.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import pennylane as qml
from pennylane import numpy as np
import qiskit
import qiskit.providers.aer.noise as noise
import os
import time
import argparse
from argparse import Namespace

from tqdm import tqdm
import json

from circuit.vqe_pl import VQE_HE, gen_h, VQE_H2
from logger import setup_logger

backends = ['ibmq_belem', 'ibmq_bogota', 'ibmq_quito', 'ibmq_lima']

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--W", type=int, default=4)
    parser.add_argument("--p", type=float, default=0)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--mol", type=str, default='h2')
    parser.add_argument("--aggre", type=str, default='average')
    parser.add_argument("--random", type=int, default=0, help='0: share the same random seed; 1: not')
    opt = parser.parse_args()
    return opt

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def weighted_weights(model, loss):
    loss_gather = [torch.ones_like(loss) for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(loss_gather, loss, async_op=False)
    loss_gather = torch.stack(loss_gather, dim=0)
    loss_gather = F.softmax(-loss_gather, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    for param in model.parameters():
        tensors_gather = [torch.ones_like(param.data) for _ in range(torch.distributed.get_world_size())]
        dist.all_gather(tensors_gather, param.data, async_op=False)
        tensors_gather = torch.stack(tensors_gather, dim=0)
        #print('loss:{}, param:{}'.format(loss_gather.size(), tensors_gather.size()))
        param.data = torch.sum(loss_gather*tensors_gather, 0)

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

""" Weight averaging. """
def average_weights(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def median_weights(model):
    size = dist.get_world_size()
    for param in model.parameters():
        param_gather = [torch.ones_like(param.data) for _ in range(size)]
        dist.all_gather(param_gather, param.data)
        param_median = torch.median(torch.stack(param_gather, dim=0), 0)[0]
        param.data = param_median

def random_weights(model):
    size = dist.get_world_size()
    for param in model.parameters():
         param_gather = [torch.ones_like(param.data) for _ in range(size)]
         dist.all_gather(param_gather, param.data)
         index = np.random.randint(0, size)
         param_random = torch.stack(param_gather, dim=0)[index]
         param.data = param_random

# def gen_h(data_dir, mol, mol_dist):
#     symbols, coordinates = qml.qchem.read_structure(os.path.join(data_dir, mol+'_'+str(mol_dist)+'.xyz'))
#     h, qubits = qml.qchem.molecular_hamiltonian(
#         symbols=symbols,
#         name=mol+str(mol_dist),
#         coordinates=coordinates,
#         charge=0,
#         mult=1,
#         basis='sto-3g',
#         mapping='jordan_wigner'
#     )
#     return h, qubits

def step(model, feat, label, optimizer, K):
    local_batch_size = len(feat) // K
    for i in range(0, len(feat), local_batch_size):
        loss = 0
        for j in range(local_batch_size):
            loss += (model(feat[i+j]) - label[i+j])**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, feat_test, label_test):
    n_correct = 0
    feat_test = torch.from_numpy(feat_test.astype(np.complex64))
    for i in range(len(feat_test)):
        predict = model(feat_test[i])
        if (predict.item()-0.5) * label_test[i] > 0:
            n_correct += 1
    return n_correct / len(feat_test)

def train(model, local_iter=8, p=0.1, M=100, distance=0.3, seed=0, h={}, world_size=1, mol='LiH', random=0, aggre='average', var_reduce=False):
    model.train()

    optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': 0.4, 'weight_decay': 0}], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.5, last_epoch=-1)
    #grad_table = {}
    #for j in range(len(h)):
    #    grad_table[h[j][0]] = torch.zeros_like(model.param)

    loss_list = []
    rank = dist.get_rank()
    if rank == 0:
        logger = setup_logger('logs/', name='PT-K{}-W{}-p{}-M{}'.format(world_size, local_iter, p, M))
    if random < 0:
        share_name = 'baseline_group'
    elif random == 0:
        share_name = 'shuffle_pc_group'
    else:
        share_name = 'random_seed'
    log_dir = 'logs/vqe/ideal/'+mol+'/'+share_name+'_'+aggre+'/'+str(seed)
    if rank==0 and (not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    
    time_cost = []
    for i in range(200):
        if random >= 0:
            np.random.shuffle(h)
        data_len = len(h) // world_size + (len(h) % world_size != 0)
        #data_len = int(M)
        h_new = h[rank*data_len:(rank+1)*data_len]
        h_part = {}
        for j in range(len(h_new)):
            h_part[h_new[j][0]] = h_new[j][1]
        model.hamiltonian = gen_h(h_part)

        st = time.time()

        loss = model()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % local_iter == 0:
            #torch.save(model.state_dict(), os.path.join(log_dir, 'model_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'_'+str(rank)+'_'+str(distance)+'_'+str(i)+'.pth'))
            if aggre == 'average':
                average_weights(model)
            elif aggre == 'median':
                median_weights(model)
            elif aggre == 'random':
                random_weights(model)
            elif aggre == 'weight':
                weighted_weights(model, loss)
            else:
                average_weights(model)
        scheduler.step()
        loss_list.append(loss.item())
        time_cost.append(time.time() - st)

        if dist.get_rank() == 0:
            #print('Epoch: {}, loss = {}'.format(i, loss))
            logger.info('Epoch: {}, loss = {}'.format(i, loss))
        # torch.save(model.cpu().state_dict(), 'logs/model.pth')
        if torch.cuda.device_count() > 0:
            model.cuda()
    #log_dir = 'logs/vqe/ideal/'+mol+'/shuffle/'+str(seed)
    #if not os.path.exists(log_dir):
    #    os.makedirs(log_dir)

    h_part = {}
    for j in range(len(h)):
        h_part[h[j][0]] = h[j][1]
    model.hamiltonian = gen_h(h_part)
    loss = model()
    np.save(os.path.join(log_dir, 'energy_partial_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'_'+str(dist.get_rank())+'_'+str(distance)), loss.item())
    #average_weights(model)
    if aggre == 'median':
        median_weights(model)
    elif aggre == 'random':
        random_weights(model)
    elif aggre == 'weight':
        weighted_weights(model, loss)
    else:
        average_weights(model)
    #log_dir = 'logs/vqe/ideal/'+mol+'/shuffle/'+str(seed)
    #if not os.path.exists(log_dir):
    #    os.makedirs(log_dir)
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
        h_part = {}
        for j in range(len(h)):
            h_part[h[j][0]] = h[j][1]
        model.hamiltonian = gen_h(h_part)
        loss = model()
        np.save(os.path.join(log_dir, 'energy_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'_'+str(dist.get_rank())+'_'+str(distance)), loss.item())
    np.save(os.path.join(log_dir, 'loss_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'_'+str(dist.get_rank())+'_'+str(distance)), loss_list)
    np.save(os.path.join(log_dir, 'time'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)), np.mean(time_cost))

def parallel_train(rank, world_size, K, p, M, distance=0.3, seed=0, mol='LiH', random=0, aggre='average'):
    h = ''
    with open('data/'+mol+str(distance)+'.json', 'r') as f:
        h = json.load(f)
    h = list(h.items())
    # temporal for test
    if random <= 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
    #np.random.shuffle(h)
    data_len = len(h) // world_size + (len(h) % world_size != 0)
    h_new = h[rank*data_len:(rank+1)*data_len]
    h_part = {}
    for i in range(len(h_new)):
        h_part[h_new[i][0]] = h_new[i][1]
    n_qubits = len(h_new[0][0])

    # 1. define quantum device
    if p < 0:
        assert world_size<=len(backends), '# available QC is limited: {}, while # workers is {}'.format(len(backends), world_size)
        dev = qml.device('qiskit.ibmq', wires=4, backend=backends[rank], provider=provider)
    elif p > 0 and p < 1:
        #qiskit.IBMQ.load_account()
        #provider = qiskit.IBMQ.providers(group='open')[0]
        #backend = provider.get_backend('ibmq_quito')
        #noise_model = noise.NoiseModel.from_backend(backend)
        if M > 0:
            #dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model, shots=M)
            dev = qml.device('default.mixed', wires=n_qubits, shots=M)
        else:
            #dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)
            dev = qml.device('default.mixed', wires=n_qubits)
    elif p >= 1:
        qiskit.IBMQ.load_account()
        provider = qiskit.IBMQ.providers(group='open')[0]
        backend = provider.get_backend('ibmq_quito')
        noise_model = noise.NoiseModel.from_backend(backend)
        if M > 0:
            dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model, shots=M)
        else:
            dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)
    else:
        if M > 0:
            dev = qml.device("default.qubit", wires=n_qubits, shots=M)
        else:
            dev = qml.device("default.qubit", wires=n_qubits)

    # define model
    n_layers = 4
    #torch.manual_seed(seed)
    if mol == 'h2':
        model = VQE_H2(dev, n_qubits, h_part, p=p)
    else:
        model = VQE_HE(dev, n_qubits, h_part, p=p)
    average_weights(model)

    # parallel wrapper
    if torch.cuda.device_count() > 0:
        model = nn.DataParallel(model)
        model.cuda()

    train(model, local_iter=K, p=p, M=M, distance=distance, seed=seed, h=h, world_size=world_size, mol=mol, random=random, aggre=aggre)

def init_process(rank, size, K, port, p, M, distance, seed, mol, random, aggre, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.set_num_threads(1)
    fn(rank, size, K, p, M, distance, seed=seed, mol=mol, random=random, aggre=aggre)

if __name__ == '__main__':
    opt = get_opt()
    mp.set_start_method("spawn")
    cost = []
    for distance in [0.5]:#, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]:
        for world_size in tqdm([opt.W], desc='world_size'):
            #world_size = 2#mp.cpu_count()

            processes = []
            st = time.time()
            for rank in range(world_size):
                p = mp.Process(target=init_process, args=(rank, world_size, opt.K, opt.port+29500, opt.p, opt.M, distance, opt.seed, opt.mol, opt.random, opt.aggre, parallel_train))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            cost.append(time.time() - st)
    if opt.random < 0:
        share_name = 'baseline_group'
    elif opt.random == 0:
        share_name = 'shuffle_pc_group'
    else:
        share_name = 'random_seed'
    # np.save('logs/vqe/ideal/'+opt.mol+'/'+share_name+'_'+opt.aggre+'/'+str(opt.seed)+'/time'+str(opt.W)+'_'+str(opt.K)+'_'+str(opt.p)+'_'+str(opt.M), cost)

