# Horovod: scale learning rate by lr_scaler.
optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                        momentum=args.momentum)

# Horovod: broadcast parameters & optimizer state.
# 모델의 파라미터를 root worker에서 다른 워커들에게 브로드캐스팅
# 각 워커들은 다른 데이터를 가지고 학습 -> 학습 초기에는 각 워커들이 랜덤한 초기화 파라미터를 가질 수 있음
# 그러나 이 함수를 사용하여 모든 워커에 동일하게 초기화
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

# 모든 워커들이 root worker와 동일한 optimizer 사용
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


# Horovod: wrap optimizer with DistributedOptimizer.
# DistributedOptimizer 생성 
# 이를 통해 옵티마이저의 업데이트(그래디언트 동기화)가 모든 워커에 동시에 적용되도록 함
optimizer = hvd.DistributedOptimizer(optimizer,
                                        named_parameters=model.named_parameters(),
                                        op=hvd.Adasum if args.use_adasum else hvd.Average,
                                        gradient_predivide_factor=args.gradient_predivide_factor)