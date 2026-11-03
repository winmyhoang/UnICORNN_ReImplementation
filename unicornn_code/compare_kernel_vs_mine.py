"""
Compare the authors' CUDA kernel with my implementation on identical inputs (needs an NVIDIA GPU).

Both compute one UnICORNN recurrence on the same random input x (the already transformed input
V*y + b), the same weights w and c, and the same loss (a linear function of the output, so the
gradient of the loss with respect to the output is a fixed random tensor). The script compares

  * the output sequence,
  * the gradients with respect to x, w and c,

between (a) the authors' kernel, (b) my reversible implementation (UnICORNNFn), and (c) a float64
autograd reference (reference_recurrence in unicornn.py), which serves as the "truth".
It also times forward + backward.

How to read it: both float32 implementations should be close to each other and to the float64
reference. Relative errors of about 1e-7 to 1e-4 are float32 rounding; a relative error close to 1
means the two implementations compute something different (for example a different index convention).

Usage (after patching the authors' network.py with patch_network.py):
    python compare_kernel_vs_mine.py --authors unicornn/psMNIST
Options: --length 784 --batch 16 --hidden 128 --dt 0.482 --alpha 12.53 --seed 0
"""
import argparse
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument('--authors', default='unicornn/psMNIST', help='folder with the (patched) network.py')
p.add_argument('--length', type=int, default=784)
p.add_argument('--batch', type=int, default=16)
p.add_argument('--hidden', type=int, default=128)
p.add_argument('--dt', type=float, default=0.482)
p.add_argument('--alpha', type=float, default=12.53)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--repeats', type=int, default=5, help='repetitions for the timing')
args, _ = p.parse_known_args()   # ignores extra arguments such as Jupyter's '-f kernel.json'

assert torch.cuda.is_available(), 'the authors kernel needs an NVIDIA GPU (use a GPU runtime)'
dev = 'cuda'
sys.path.insert(0, args.authors)
from network import UnICORNN_Compute_GPU            # authors' code (patched)
from unicornn import UnICORNNFn, reference_recurrence  # my code

T, B, m = args.length, args.batch, args.hidden
dt, alpha = args.dt, args.alpha
torch.manual_seed(args.seed)
x0 = torch.randn(T, B, m, device=dev)
w0 = torch.rand(m, device=dev)                         # U(0,1) as in the paper
c0 = 0.2 * torch.rand(m, device=dev) - 0.1             # U(-0.1,0.1) as in the paper
tgt = torch.randn(T, B, m, device=dev)                 # d loss / d output


def authors_fn(x, w, c):
    hy0 = torch.zeros(B, m, device=x.device, dtype=x.dtype)
    hz0 = torch.zeros(B, m, device=x.device, dtype=x.dtype)
    return UnICORNN_Compute_GPU.apply(x, w, hy0, hz0, c, torch.tensor(alpha), torch.tensor(dt))


def mine_fn(x, w, c):
    return UnICORNNFn.apply(x, w, c, dt, alpha)


def ref_fn(x, w, c):
    return reference_recurrence(x, w, c, dt, alpha)


def run(fn, dtype):
    x = x0.to(dtype).clone().requires_grad_(True)
    w = w0.to(dtype).clone().requires_grad_(True)
    c = c0.to(dtype).clone().requires_grad_(True)
    out = fn(x, w, c)
    (out * tgt.to(dtype)).sum().backward()
    return [out.detach().double(), x.grad.double(), w.grad.double(), c.grad.double()]


def rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def mx(a, b):
    return (a - b).abs().max().item()


names = ['output', 'grad x', 'grad w', 'grad c']
ref = run(ref_fn, torch.float64)
res = {'authors kernel': run(authors_fn, torch.float32), 'my implementation': run(mine_fn, torch.float32)}

print(f'T={T} batch={B} hidden={m} dt={dt} alpha={alpha}  (errors: relative norm / max abs)')
for label, r in res.items():
    print(f'\n{label} vs float64 reference')
    for n, a, b in zip(names, r, ref):
        print(f'  {n:7s} rel {rel(a, b):.2e}   max {mx(a, b):.2e}')
print('\nauthors kernel vs my implementation')
for n, a, b in zip(names, res['authors kernel'], res['my implementation']):
    print(f'  {n:7s} rel {rel(a, b):.2e}   max {mx(a, b):.2e}')


def timeit(fn):
    run(fn, torch.float32)                              # warm-up (the kernel is compiled on first use)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.repeats):
        run(fn, torch.float32)
    torch.cuda.synchronize()
    return (time.time() - t0) / args.repeats * 1000.0


ta, tm = timeit(authors_fn), timeit(mine_fn)
print(f'\nforward + backward, batch {B}: authors kernel {ta:.1f} ms, my implementation {tm:.1f} ms '
      f'(ratio {tm / ta:.1f}x)')
