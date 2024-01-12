# 모델 동기화가 수행되는지 확인하기 위함

import os
import time

import torch
from torch_geometric.datasets import Amazon
import torch_geometric.loader as loader
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from filelock import FileLock

import horovod
import horovod.torch as hvd

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = self.conv2(x, edge_index)
        return output
    
def train_node_classifier(model, graph, optimizer, n_epochs=180):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out, graph.y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)

        # if epoch % 10 == 0:
        #     print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}')

        if hvd.rank() == 0 and epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}')

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def eval_node_classifier(model, graph):
    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred == graph.y).sum()
    test_accuracy = int(correct) / int(graph.num_nodes)
    print('rank: %d, test_acc: %.2f' %(proc_num, test_accuracy))
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
    return test_accuracy

hvd.init()
torch.set_num_threads(1)

start_time = time.time()

device = "cpu"

data_dir = './amazon'
num_clusters = 5
dataset = Amazon(root=data_dir, name='computers')
graph = dataset[0]

clustered_datasets = [torch.load(f'{data_dir}/cluster_{i}.pt') for i in range(num_clusters)]

gcn = GCN().to(device)  
lr_scaler = hvd.size()
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01 * lr_scaler, weight_decay=5e-4)

hvd.broadcast_parameters(gcn.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer_gcn, root_rank=0)

optimizer_gcn = hvd.DistributedOptimizer(optimizer_gcn,
                                         named_parameters=gcn.named_parameters(),
                                         op=hvd.Average,
                                         gradient_predivide_factor=1.0)

criterion = nn.CrossEntropyLoss()
proc_num = hvd.rank()
test_data = clustered_datasets[4]
train_node_classifier(gcn, clustered_datasets[proc_num], optimizer_gcn)

accuracy = eval_node_classifier(gcn, test_data)

# if proc_num==0:
#     print(f"Parameters in Process {proc_num}:")
#     for param_tensor in gcn.state_dict():
#         print(param_tensor, "\t", gcn.state_dict()[param_tensor].size())
#         print(gcn.state_dict()[param_tensor])


if proc_num == 0:
    print("test accuracy: %.3f" %accuracy)
    print("총 소요 시간: %.3f초" %(time.time() - start_time))