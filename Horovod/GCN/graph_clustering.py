import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.loader as loader
import torch_geometric.transforms as T

import horovod
import horovod.torch as hvd

hvd.init()
print(hvd.size())

dataset = Planetoid(root='./data', name='Cora')
graph = dataset[0]
split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
graph = split(graph)

cluster = loader.ClusterData(graph, num_parts=hvd.size())
clusterloader = loader.ClusterLoader(cluster)


for i in clusterloader:
    print(i)
    
