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

proc_num = hvd.rank()
print(clustered_datasets[proc_num])
clustered_datasets[proc_num] = graph_split(clustered_datasets[proc_num])
print('train set:' ,clustered_datasets[proc_num].train_mask.sum().item())
print('val set:', clustered_datasets[proc_num].val_mask.sum().item())
print('test set:', clustered_datasets[proc_num].test_mask.sum().item())
