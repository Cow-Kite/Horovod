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
split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
graph = split(graph)

cluster = loader.ClusterData(graph, num_parts=hvd.size())
clusterloader = loader.ClusterLoader(cluster)

num = 0
for graph_data in clusterloader:
    if(hvd.rank()==num):
        print(graph_data)
    num += 1

    
