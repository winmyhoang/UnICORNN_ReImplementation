"""
Noise-padded MNIST, 10 classes: UnICORNN (my implementation, unicornn.py) vs LSTM vs tanh RNN.

TASK: a 28x28 MNIST digit is fed row by row (28 time steps, 28 numbers each). It is followed
by N steps of Gaussian noise (28 numbers per step, fresh noise for every example). The model
reads the whole sequence and must name the digit (10 classes, chance = 10%). N = 0 is plain
row-wise MNIST (a control: all models should do well); larger N = longer time dependency,
because the digit is only seen at the start.

The digit is the only useful information and it arrives first, so, as in the paper's noise-padded
CIFAR-10 test, the network has to carry it across the noise. Differences from the paper's test:
MNIST instead of CIFAR-10, my own sizes/settings, and fresh noise per example (the authors'
code reuses one fixed noise sequence for all images).

Fairness settings (all models): same parameter count (+-), same Adam learning rate, same gradient
clipping (default 1.0), same data and noise for evaluation. The LSTM additionally gets a large
forget-gate bias (default 5), which my earlier experiments showed it needs. Nothing is tuned for
UnICORNN (dt, alpha are my guesses): if it does badly, try other --dt / --alpha before concluding.

Reported per run: final test accuracy, and test accuracy of the model with the best validation
accuracy (checked every --eval_every iterations).

Put unicornn.py in the same folder. Examples:
    python noisy_mnist_compare.py --lengths 0 100 --iters 300 --eval_every 150     # quick test
    python noisy_mnist_compare.py                                                  # default
    python noisy_mnist_compare.py --lengths 300 --models unicornn --dt 0.2         # one setting
Outputs: <out>_results.csv, <out>_accuracy.png
"""
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from unicornn import UnICORNN

p = argparse.ArgumentParser()
p.add_argument('--lengths', type=int, nargs='+', default=[0, 100, 300], help='noise steps after the image')
p.add_argument('--models', nargs='+', default=['unicornn', 'lstm', 'rnn'])
p.add_argument('--iters', type=int, default=1500)
p.add_argument('--batch', type=int, default=64)
p.add_argument('--seeds', type=int, default=1)
p.add_argument('--nhid', type=int, default=64, help='UnICORNN units per layer')
p.add_argument('--layers', type=int, default=2)
p.add_argument('--dt', type=float, default=0.1)
p.add_argument('--alpha', type=float, default=1.0)
p.add_argument('--lr', type=float, default=3e-3)
p.add_argument('--noise', type=float, default=1.0, help='std of the Gaussian noise steps')
p.add_argument('--clip', type=float, default=1.0, help='grad-norm clipping for all models (0 = off)')
p.add_argument('--forget_bias', type=float, default=5.0, help='LSTM forget-gate bias (negative = PyTorch default)')
p.add_argument('--eval_every', type=int, default=250)
p.add_argument('--val_n', type=int, default=2000)
p.add_argument('--test_n', type=int, default=10000)
p.add_argument('--out', default='noisy_mnist')
args = p.parse_args()
print(args)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev)


# ---------------------------------------------------------------- data
def load(train):
    ds = torchvision.datasets.MNIST('data/', train=train, download=True)
    x = ds.data.float().div(255.0)
    x = (x - 0.1307) / 0.3081
    return x, ds.targets


x_all, y_all = load(True)
x_test, y_test = load(False)
g0 = torch.Generator().manual_seed(0)
perm = torch.randperm(x_all.shape[0], generator=g0)
val_idx, tr_idx = perm[:5000], perm[5000:]
x_tr, y_tr = x_all[tr_idx].to(dev), y_all[tr_idx].to(dev)
x_val, y_val = x_all[val_idx][:args.val_n].to(dev), y_all[val_idx][:args.val_n].to(dev)
x_te, y_te = x_test[:args.test_n].to(dev), y_test[:args.test_n].to(dev)


def make_seq(img, N, gen=None):
    """img: (B,28,28) -> sequence (28+N, B, 28): image rows, then N noise steps."""
    x = img.permute(1, 0, 2)
    if N > 0:
        noise = args.noise * torch.randn(N, img.shape[0], 28, device=dev, generator=gen)
        x = torch.cat([x, noise], 0)
    return x.contiguous()


# models
class RNNClassifier(nn.Module):
    def __init__(self, kind, n_in, hidden, layers, n_out):
        super().__init__()
        if kind == 'lstm':
            self.rnn = nn.LSTM(n_in, hidden, layers)
        else:
            self.rnn = nn.RNN(n_in, hidden, layers, nonlinearity='tanh')
        self.out = nn.Linear(hidden, n_out)
        if kind == 'lstm' and args.forget_bias >= 0:
            for name_, prm in self.rnn.named_parameters():
                if name_.startswith('bias_ih'):
                    prm.data[hidden:2 * hidden] = args.forget_bias
                elif name_.startswith('bias_hh'):
                    prm.data[hidden:2 * hidden] = 0.0

    def forward(self, x):                       # x: (T,B,28)
        o, _ = self.rnn(x)
        return self.out(o[-1])


def n_params(m):
    return sum(q.numel() for q in m.parameters())


def make_unicornn():
    return UnICORNN(28, args.nhid, 10, L=args.layers, dt=args.dt, alpha=args.alpha, dropout=0.0)


def make_model(name):
    if name == 'unicornn':
        return make_unicornn().to(dev)
    target = n_params(make_unicornn())
    best = None
    for h in range(2, 600):
        cnt = n_params(RNNClassifier(name, 28, h, args.layers, 10))
        d = abs(cnt - target)
        if best is None or d < best[0]:
            best = (d, h)
        if cnt > target:
            break
    return RNNClassifier(name, 28, best[1], args.layers, 10).to(dev)


# training / evaluation
def evaluate(model, x, y, N, seed):
    model.eval()
    gen = torch.Generator(device=dev).manual_seed(seed)
    correct = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], 500):
            xb = make_seq(x[i:i + 500], N, gen)
            correct += (model(xb).argmax(1) == y[i:i + 500]).sum().item()
    model.train()
    return 100.0 * correct / x.shape[0]


def run_one(name, N, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(name)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_iter, best_state, curve, diverged = -1.0, 0, None, [], False
    t0 = time.time()
    done = 0
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, x_tr.shape[0], (args.batch,), device=dev)
        xb = make_seq(x_tr[idx], N)
        loss = loss_fn(model(xb), y_tr[idx])
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        done = it
        if it % args.eval_every == 0 or it == args.iters:
            v = evaluate(model, x_val, y_val, N, 1)
            curve.append('%d:%.1f' % (it, v))
            if v > best_val:
                best_val, best_iter = v, it
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    sec = (time.time() - t0) / max(1, done)
    final_test = evaluate(model, x_te, y_te, N, 2)
    best_test = final_test
    if best_state is not None:
        model.load_state_dict(best_state)
        best_test = evaluate(model, x_te, y_te, N, 2)
    return dict(model=name, N=N, seed=seed, params=n_params(model), final_test=final_test,
                best_val=best_val, best_test=best_test, best_iter=best_iter,
                diverged=diverged, sec_per_iter=round(sec, 3), val_curve=' '.join(curve))


if __name__ == '__main__':
    for name in args.models:
        print(f'{name:9s} parameters: {n_params(make_model(name))}')
    rows = []
    for N in args.lengths:
        for name in args.models:
            for seed in range(args.seeds):
                r = run_one(name, N, seed)
                rows.append(r)
                print(f"N={N:4d} {name:9s} seed={seed} params={r['params']:6d} "
                      f"final_test={r['final_test']:.2f} best_val={r['best_val']:.2f}@{r['best_iter']} "
                      f"best_test={r['best_test']:.2f} diverged={r['diverged']} ({r['sec_per_iter']:.2f}s/iter)")
                print('      val curve:', r['val_curve'])
    with open(args.out + '_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    xs = np.arange(len(args.lengths))
    for name in args.models:
        m, s = [], []
        for N in args.lengths:
            a = [r['best_test'] for r in rows if r['model'] == name and r['N'] == N]
            m.append(np.mean(a)); s.append(np.std(a))
        ax.errorbar(xs, m, yerr=s, marker='o', capsize=2, label=name)
    ax.axhline(10, ls='--', color='gray', lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([str(N) for N in args.lengths])
    ax.set_xlabel('noise steps after the image'); ax.set_ylabel('test accuracy (%)')
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(args.out + '_accuracy.png', dpi=150)
    print('saved', args.out + '_results.csv and', args.out + '_accuracy.png')
