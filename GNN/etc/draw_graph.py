import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import random


data_dir = './amazon'
num_clusters = 5

clusters = [torch.load(f'{data_dir}/cluster_{i}.pt') for i in range(num_clusters)]

def convert_to_networkx(graph, n_sample=None):
    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    return g, y

def plot_graph(g, y):
    plt.figure(figsize=(20, 20))
    nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
    plt.savefig('node.png')
    plt.show()

g, y = convert_to_networkx(clusters[0])
plot_graph(g, y)