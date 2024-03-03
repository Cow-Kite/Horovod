import torch

import torch_geometric.loader as loader
from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T

data_dir = './amazon'
num_clusters = 5
dataset = Amazon(root=data_dir, name='computers')
graph = dataset[0]

cluster = loader.ClusterData(graph, num_parts=num_clusters, recursive=False)
clusterloader = loader.ClusterLoader(cluster)

# 각 배치(파티션)를 파일로 저장
for i, subgraph in enumerate(clusterloader):
    torch.save(subgraph, f'{data_dir}/cluster_{i}.pt')

# 저장된 파티션 확인
print("Saved clusters:", [f'cluster_{i}.pt' for i in range(len(clusterloader))])

loaded_clusters = []

for i in range(num_clusters):
    cluster = torch.load(f'{data_dir}/cluster_{i}.pt')
    loaded_clusters.append(cluster)

for i, cluster in enumerate(loaded_clusters):
    print(f"Cluster {i}:", cluster)

# 각 파티션 로드
clusters = [torch.load(f'{data_dir}/cluster_{i}.pt') for i in range(num_clusters)]
