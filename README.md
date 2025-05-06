## Horovod 기반 분산 GCN 학습에서 그래프 파티셔닝 특성 분석
### - Paper
__강소연__, 이상훈, 문양세, "Horovod 기반 분산 GCN 학습에서 그래프 파티셔닝 특성 분석", 한국컴퓨터종합학술대회, pp. 161-163, 2024.
### - 기술스택
Horovod, PyTorch, PyTorch Geometric


## Horovod 분산 프레임워크 동작 과정
__(1) Graph Partitioning__<br>
- METIS로 그래프 데이터 파티셔닝 수행 -> 서브그래프 구성<br>

__(2) Subgraph Assignment__<br>
- 서브그래프를 각 프로세스에 할당<br>
  
__(3) Distributed Training__<br>
- 할당받은 서브그래프로 GCN 분산 학습 수행<br>
- 각 서버의 프로세스는 모델을 독립적으로 학습<br>

__(5) Synchronization__<br>
- 각 서버에서 파라미터 계산 후, All-Reduce 연산 수행<br>
- 모든 서버의 모델이 동일한 파라미터로 업데이트 -> 모델의 일관성 보장<br>

![image](https://github.com/user-attachments/assets/2048d289-f61c-4475-b85b-7347e8bc8488)

## 그래프 파티셔닝 특성
__(1) 엣지 컷__<br>
- 그래프 데이터는 노드와 노드 사이를 잇는 엣지로 구성<br>
- 파티셔닝 수행 후, 파티션 간 엣지가 삭제되는 엣지 컷 발생<br> 


__(2) 차수(degree)에 따른 엣지 컷__<br>
- 고 차수(high degree) 그래프 데이터일수록 엣지 컷 많이 발생<br>
- 엣지 컷 -> 정보 손실 -> 학습 성능에 영향<br>





