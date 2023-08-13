import os

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.loader as loader
import torch_geometric.transforms as T
from filelock import FileLock

import horovod
import horovod.torch as hvd

hvd.init()

data_dir = './data'

with FileLock(os.path.expanduser("~/.horovod_lock")):
    dataset = Planetoid(root=data_dir, name='Cora')
    
graph = dataset[0]
clustered_datasets = []

cluster = loader.ClusterData(graph, num_parts=hvd.size())
clusterloader = loader.ClusterLoader(cluster)

for graph_data in clusterloader:
    clustered_datasets.append(graph_data)

def graph_split(graph):
    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    graph = split(graph)
    return graph

if hvd.rank() == 0:
    print(clustered_datasets[0])
    clustered_datasets[0] = graph_split(clustered_datasets[0])
    print('00 train set:' ,clustered_datasets[0].train_mask.sum().item())
    print('00 val set:', clustered_datasets[0].val_mask.sum().item())
    print('00 test set:', clustered_datasets[0].test_mask.sum().item())
if hvd.rank() == 1:
    print(clustered_datasets[1])
    clustered_datasets[1] = graph_split(clustered_datasets[1])
    print('01 train set:' ,clustered_datasets[1].train_mask.sum().item())
    print('01 val set:', clustered_datasets[1].val_mask.sum().item())
    print('01 test set:', clustered_datasets[1].test_mask.sum().item())
if hvd.rank() == 2:
    print(clustered_datasets[2])
    clustered_datasets[2] = graph_split(clustered_datasets[2])
    print('02 train set:' ,clustered_datasets[2].train_mask.sum().item())
    print('02 val set:', clustered_datasets[2].val_mask.sum().item())
    print('02 test set:', clustered_datasets[2].test_mask.sum().item())
if hvd.rank() == 3:
    print(clustered_datasets[3])
    clustered_datasets[3] = graph_split(clustered_datasets[3])
    print('02 train set:' ,clustered_datasets[3].train_mask.sum().item())
    print('02 val set:', clustered_datasets[3].val_mask.sum().item())
    print('02 test set:', clustered_datasets[3].test_mask.sum().item())

    
