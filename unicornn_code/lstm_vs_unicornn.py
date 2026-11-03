"""
Long-memory experiment: UnICORNN (my implementation, unicornn.py) vs LSTM vs
plain tanh-RNN.

TASK ("remember the first input"): a sequence of length N. The first step holds
the label (+2 or -2); every other step is Gaussian noise. The model must output
the label after seeing all N steps. Longer N = longer time dependency.

PART 1 (training): for each sequence length N and each model, train for a fixed
number of iterations and report test accuracy (+ iterations needed to reach 90%
running accuracy).
PART 2 (gradients): at initialisation, measure how strongly the final output
depends on the input k steps back (norm of d output / d input_k versus lag).
NOTE: this is a *proxy* for the paper's gradient quantities (Props. 3.1/3.2),
not the same thing.

All models get a similar number of parameters. Hyperparameters are NOT tuned
(same lr for all; dt, alpha for UnICORNN are the values of the toy demo).
No gradient clipping by default (--clip 0), so exploding gradients are not hidden.

LSTM options added after the default LSTM stayed at chance level:
    --forget_bias B   set the LSTM forget-gate bias (default: PyTorch's initialisation).
                      A large bias (e.g. 5) makes the gate keep its memory at the start.
    --clip 1.0        gradient-norm clipping, for all models.
A run counts as a success if its test accuracy is above 0.9 (chance is 0.5).

Put unicornn.py in the same folder. Examples:
    python lstm_vs_unicornn.py --lengths 100 300 --iters 200      # quick test
    python lstm_vs_unicornn.py                                    # default
    python lstm_vs_unicornn.py --seeds 3                          # mean +- std
    python lstm_vs_unicornn.py --skip_train                       # gradient plot only
    python lstm_vs_unicornn.py --models lstm --lengths 300 --iters 500 --seeds 3 --clip 1.0 --forget_bias 5.0 --skip_grad
Outputs: <out>_results.csv, <out>_accuracy.png, <out>_grad_vs_lag.png
"""
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn

from unicornn import UnICORNN

p = argparse.ArgumentParser()
p.add_argument('--lengths', type=int, nargs='+', default=[100, 300, 1000])
p.add_argument('--models', nargs='+', default=['unicornn', 'lstm', 'rnn'])
p.add_argument('--iters', type=int, default=400)
p.add_argument('--batch', type=int, default=64)
p.add_argument('--seeds', type=int, default=1)
p.add_argument('--nhid', type=int, default=64, help='UnICORNN units per layer')
p.add_argument('--layers', type=int, default=2)
p.add_argument('--dt', type=float, default=0.1)
p.add_argument('--alpha', type=float, default=1.0)
p.add_argument('--lr', type=float, default=3e-3)
p.add_argument('--noise', type=float, default=0.3)
p.add_argument('--clip', type=float, default=0.0, help='grad-norm clipping (0 = off)')
p.add_argument('--forget_bias', type=float, default=None, help='LSTM forget-gate bias (None = PyTorch default)')
p.add_argument('--grad_len', type=int, default=500, help='sequence length for the gradient plot')
p.add_argument('--skip_train', action='store_true')
p.add_argument('--skip_grad', action='store_true')
p.add_argument('--out', default='lstm_vs_unicornn')
args = p.parse_args()
print(args)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev)


#  models
class RNNClassifier(nn.Module):
    def __init__(self, kind, hidden, layers, n_out=2):
        super().__init__()
        if kind == 'lstm':
            self.rnn = nn.LSTM(1, hidden, layers)
        else:
            self.rnn = nn.RNN(1, hidden, layers, nonlinearity='tanh')
        self.out = nn.Linear(hidden, n_out)
        if kind == 'lstm' and args.forget_bias is not None:
            for name_, prm in self.rnn.named_parameters():
                if name_.startswith('bias_ih'):
                    prm.data[hidden:2 * hidden] = args.forget_bias
                elif name_.startswith('bias_hh'):
                    prm.data[hidden:2 * hidden] = 0.0

    def forward(self, x):                       # x: (N,B,1)
        o, _ = self.rnn(x)
        return self.out(o[-1])


def n_params(m):
    return sum(q.numel() for q in m.parameters())


def make_unicornn():
    return UnICORNN(1, args.nhid, 2, L=args.layers, dt=args.dt, alpha=args.alpha, dropout=0.0)


def make_model(name):
    if name == 'unicornn':
        return make_unicornn().to(dev)
    target = n_params(make_unicornn())
    best = None
    for h in range(2, 600):                     # parameter count grows with h
        d = abs(n_params(RNNClassifier(name, h, args.layers)) - target)
        if best is None or d < best[0]:
            best = (d, h)
        if n_params(RNNClassifier(name, h, args.layers)) > target:
            break
    return RNNClassifier(name, best[1], args.layers).to(dev)


#  data
def make_batch(N, B):
    label = torch.randint(0, 2, (B,), device=dev)
    x = args.noise * torch.randn(N, B, 1, device=dev)
    x[0, :, 0] = (2 * label - 1).float() * 2.0
    return x, label


#part 1: training
def run_one(name, N, seed):
    torch.manual_seed(seed)
    model = make_model(name)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    accs, first90, diverged = [], None, False
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x, y = make_batch(N, args.batch)
        out = model(x)
        loss = loss_fn(out, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        accs.append((out.argmax(1) == y).float().mean().item())
        if first90 is None and it >= 20 and np.mean(accs[-20:]) >= 0.9:
            first90 = it
    sec_per_iter = (time.time() - t0) / max(1, len(accs))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(5):
            x, y = make_batch(N, 200)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return dict(model=name, N=N, seed=seed, params=n_params(model),
                test_acc=correct / total, iters_to_90=first90,
                diverged=diverged, sec_per_iter=sec_per_iter)


def part1():
    rows = []
    for N in args.lengths:
        for name in args.models:
            for seed in range(args.seeds):
                r = run_one(name, N, seed)
                rows.append(r)
                print(f"N={N:5d} {name:9s} seed={seed} params={r['params']:6d} "
                      f"test_acc={r['test_acc']:.3f} iters_to_90={r['iters_to_90']} "
                      f"diverged={r['diverged']} ({r['sec_per_iter']:.2f}s/iter)")
    with open(args.out + '_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for name in args.models:
        m, s = [], []
        for N in args.lengths:
            a = [r['test_acc'] for r in rows if r['model'] == name and r['N'] == N]
            m.append(np.mean(a)); s.append(np.std(a))
        ax.errorbar(args.lengths, m, yerr=s, marker='o', capsize=2, label=name)
    ax.axhline(0.5, ls='--', color='gray', lw=0.8)
    ax.set_xscale('log'); ax.set_xlabel('sequence length N'); ax.set_ylabel('test accuracy')
    ax.set_ylim(0.4, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(args.out + '_accuracy.png', dpi=150)
    print('saved', args.out + '_results.csv and', args.out + '_accuracy.png')


# part 2: gradient vs lag
def grad_vs_lag(name, N, seed=0, B=32):
    torch.manual_seed(seed)
    model = make_model(name)
    model.train()   # cuDNN RNN backward needs train mode; no dropout is used
    x = (args.noise * torch.randn(N, B, 1, device=dev)).requires_grad_(True)
    out = model(x)
    (out[:, 0] - out[:, 1]).sum().backward()
    g = x.grad.abs().squeeze(-1).mean(1).detach().cpu().numpy()   # per time step
    return g[::-1].copy()                                          # index = lag


def part2():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    N = args.grad_len
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    print(f'gradient of the output w.r.t. the input `lag` steps back (N={N}, at initialisation)')
    for name in args.models:
        g = grad_vs_lag(name, N)
        lags = [0, N // 10, N // 2, N - 1]
        print(f'  {name:9s}', '  '.join(f'lag {l}: {g[l]:.2e}' for l in lags))
        ax.semilogy(np.arange(N), np.maximum(g, 1e-30), lw=1, label=name)
    ax.set_xlabel('lag (steps between input and output)')
    ax.set_ylabel('mean |d output / d input|')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(args.out + '_grad_vs_lag.png', dpi=150)
    print('saved', args.out + '_grad_vs_lag.png')


if __name__ == '__main__':
    for name in args.models:
        print(f'{name:9s} parameters: {n_params(make_model(name))}')
    if not args.skip_grad:
        part2()
    if not args.skip_train:
        part1()
