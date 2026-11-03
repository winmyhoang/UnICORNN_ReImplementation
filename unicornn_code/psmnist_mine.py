"""
Permuted sequential MNIST with MY pure-PyTorch UnICORNN (unicornn.py),
using the authors' seed, data split, optimiser (Adam) and lr schedule.
Slow (Python loop over 784 steps). Use --max_batches for a quick test.

Put unicornn.py and this file in the same folder, then e.g.:
    python psmnist_mine.py --epochs 1 --max_batches 50
"""
import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from unicornn import UnICORNN

p = argparse.ArgumentParser()
p.add_argument('--nhid', type=int, default=128)
p.add_argument('--nlayers', type=int, default=3)
p.add_argument('--dt', type=float, default=0.482)      # Table 6, 128 units
p.add_argument('--alpha', type=float, default=12.53)   # Table 6, 128 units
p.add_argument('--lr', type=float, default=1.14e-3)    # Table 6, 128 units
p.add_argument('--batch', type=int, default=64)
p.add_argument('--dropout', type=float, default=0.1)
p.add_argument('--epochs', type=int, default=2)
p.add_argument('--reduce_point', type=int, default=0,
               help='divide lr by 10 after this many epochs (0 = never; paper: 650)')
p.add_argument('--max_batches', type=int, default=0,
               help='limit batches per epoch for a quick test (0 = all)')
args = p.parse_args()
print(args)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev)

# same seed / permutation call order as the authors' psMNIST_task.py
torch.manual_seed(5544)
perm = torch.randperm(784)

tf = transforms.ToTensor()
train_full = torchvision.datasets.MNIST('data/', train=True, transform=tf, download=True)
test_set = torchvision.datasets.MNIST('data/', train=False, transform=tf)
train_set, valid_set = torch.utils.data.random_split(train_full, [57000, 3000])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

model = UnICORNN(1, args.nhid, 10, L=args.nlayers, dt=args.dt, alpha=args.alpha,
                 dropout=args.dropout).to(dev)
nn.init.kaiming_normal_(model.readout.weight)   # authors' classifier init
nn.init.zeros_(model.readout.bias)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()


def prep(x):                      # (B,1,28,28) -> (784,B,1), permuted
    x = x.reshape(x.size(0), 1, 784).permute(2, 0, 1)
    return x[perm].to(dev)


def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(prep(x)).argmax(1).cpu() == y).sum().item()
    return 100.0 * correct / len(loader.dataset)


best_valid, test_at_best = 0.0, 0.0
for epoch in range(args.epochs):
    model.train()
    t0 = time.time()
    for i, (x, y) in enumerate(train_loader):
        if args.max_batches and i >= args.max_batches:
            break
        opt.zero_grad()
        loss = loss_fn(model(prep(x)), y.to(dev))
        loss.backward()
        opt.step()
        if i % 20 == 0:
            print(f'  epoch {epoch} batch {i} loss {loss.item():.3f}  ({time.time()-t0:.0f}s)')
    v, t = evaluate(valid_loader), evaluate(test_loader)
    if v > best_valid:
        best_valid, test_at_best = v, t
    print(f'epoch {epoch}: valid {v:.2f}  test {t:.2f}  (test at best valid: {test_at_best:.2f})')
    if args.reduce_point and (epoch + 1) == args.reduce_point:
        for g in opt.param_groups:
            g['lr'] /= 10.0
