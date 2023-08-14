# 분산 학습 실행

    horovodrun -np 4 --gloo -H hpc01:2,hpc03:2 python3 gcn_main.py