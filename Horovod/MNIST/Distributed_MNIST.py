import argparse
import os
from packaging import version

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
from torchvision import datasets, transforms

import horovod
import horovod.torch as hvd

import time
start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-mixed-precision', action='store_true', default=False,
                    help='use mixed precision for training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--data-dir',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')

# Arguments when not run through horovodrun
parser.add_argument('--num-proc', type=int)
parser.add_argument('--hosts', help='hosts to run on in notation: hostname:slots[,host2:slots[,...]]')
parser.add_argument('--communication', help='collaborative communication to use: gloo, mpi')

# Model 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def main(args):
    # 분산 환경에서 모델 학습
    def train_epoch(epoch):
        model.train() # 모델을 학습 모드로 설정
        # Horovod: set epoch to sampler for shuffling.
        # 데이터셋을 셔플링 -> 데이터의 순서를 섞는것. 
    
        train_sampler.set_epoch(epoch)
        # 배치 단위로 데이터를 가져옴
        for batch_idx, (data, target) in enumerate(train_loader):
            # gradient를 0으로 초기화
            optimizer.zero_grad()
            # 데이터를 모델에 입력하여 예측값 output을 얻음
            output = model(data)
            # 예측 값 output과 실제 값 target 사이의 loss 계산
            loss = F.nll_loss(output, target)
            loss.backward() # backpropagation
            optimizer.step() # 파라미터 업데이트
            if batch_idx % args.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                           100. * batch_idx / len(train_loader), loss.item()))

    # Metric: accuracy, loss -> 각 워커에서 계산된 값을 모아서 평균을 계산하는 함수
    def metric_average(val, name):
        tensor = torch.tensor(val)
        # allreduce 방식으로 계산
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def test():
        model.eval() # 모델을 평가 모드로
        test_loss = 0.
        test_accuracy = 0.
        # 테스트 데이터셋에서 배치 단위로 data와 target을 가져옴
        for data, target in test_loader:
            # 모델에 데이터 입력 -> 예측 결과를 얻음 
            output = model(data)
            # sum up batch loss
            # batch 마다 loss 누적
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            # 예측된 label을 얻음
            pred = output.data.max(1, keepdim=True)[1]
            # batch 마다 accuracy 누적
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        # 평균 loss and accuracy
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        # Horovod: average metric values across workers.
        # 각 워커에서 계산된 metric(loss, accuracy)를 전체 워커들 간에 평균화
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    # 각 작업별로 CPU thread 1개 사용
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
#    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # dataset load
    data_dir = args.data_dir or './data'
    # FileLock: 분산 학습할 때, FileLock을 사용하여 여러 프로세스가 동시에 dataset을 다운로드하는 것을 방지
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = \
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

    # 분산학습을 위한 데이터 파티셔닝
    # Horovod: use DistributedSampler to partition the training data.
    # train_sampler: 데이터셋을 분산하는 역할, 각 워커에 고르게 분배, 파라미터 동기화
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    
    # train_loader: 데이터셋을 미니 배치로 나누어서 모델에 공급하는 역할
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = \
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    model = Net()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

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

    # 학습과 테스트를 수행하는 반복문
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch)
        # Keep test in full precision since computation is relatively light.
        test()


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # this is running via horovodrun
    main(args)

    if hvd.rank() == 0:
        end_time = time.time()
        print("총 소요 시간: %.3f초" %(end_time - start_time))