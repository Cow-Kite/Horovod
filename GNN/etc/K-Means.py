import torch
from torch_geometric.datasets import Amazon
from sklearn.cluster import KMeans
import numpy as np


data_dir = './amazon'
dataset = Amazon(root=data_dir, name='computers')

graph = dataset[0]

kmeans = KMeans(n_clusters=5, random_state=0).fit(graph.x)
labels = kmeans.labels_

# 클래스별 개수 계산 및 출력
unique, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))
print(f'클러스터 별 개수: {cluster_counts}')

# 원래 그래프의 클래스를 클러스터링 결과로 바꿈
graph.y = torch.tensor(labels, dtype=torch.long)

# 변경된 그래프를 .pt 파일로 저장
torch.save(graph, 'clustered_graph.pt')
