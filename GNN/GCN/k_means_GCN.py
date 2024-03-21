
import os
import time

import torch
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np

import horovod
import horovod.torch as hvd

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(767, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = self.conv2(x, edge_index)
        return output

def graph_split(graph):
    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    graph = split(graph)
    return graph

# train
def train_node_classifier(model, graph, optimizer, n_epochs=150):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        # valid
        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, graph, graph.val_mask)

        if hvd.rank() == 0 and epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def eval_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    test_accuracy = int(correct) / int(mask.sum())
    print('rank: %d, test_acc: %.2f' %(proc_num, test_accuracy))
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
    return test_accuracy

hvd.init()
torch.set_num_threads(1)

device = "cpu"
proc_num = hvd.rank()

start_time = time.time()

data_dir = './K-Means/2part'
num_clusters = 2
clustered_datasets = []

for i in range(num_clusters):
    dataset = torch.load(f'{data_dir}/cluster_{i}.pt')
    unique, counts = np.unique(dataset.y.numpy(), return_counts=True)  # 클래스별 노드 수 계산
    class_counts = dict(zip(unique, counts))
    print(f'Cluster {i} - Class counts: {class_counts}')  # 클래스별 노드 수 출력
    clustered_datasets.append(dataset)

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

clustered_datasets[proc_num] = graph_split(clustered_datasets[proc_num])
train_node_classifier(gcn, clustered_datasets[proc_num], optimizer_gcn)
accuracy = eval_node_classifier(gcn, clustered_datasets[proc_num], clustered_datasets[proc_num].test_mask)

if proc_num == 0:
    print("test accuracy: %.3f" %accuracy)
    print("총 소요 시간: %.3f초" %(time.time() - start_time))cat: q: No such file or directory
