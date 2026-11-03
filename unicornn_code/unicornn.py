"""
UnICORNN (Rusch & Mishra, ICML 2021) -- readable pure-PyTorch implementation.

Purpose: understanding + verification, NOT speed. One Python iteration per
time step per layer. For full-scale runs use the authors' CUDA kernels:
https://github.com/tk-rusch/unicornn

Per layer l (paper eq. 6), per neuron, with x_n = V y^{l-1}_n + b:
    h   = dt * sigmoid(c)
    z_n = z_{n-1} - h * ( tanh(w*y_{n-1} + x_n) + alpha*y_{n-1} )
    y_n = y_{n-1} + h * z_n                  (uses the NEW z_n: symplectic Euler)

Run `python unicornn.py` to execute all sanity checks and a tiny training demo.
"""
import math
import torch
import torch.nn as nn
# 1. Reference recurrence (plain autograd; used to verify the custom backward)

def reference_recurrence(x, w, c, dt, alpha):
    """x: (N,B,m) already transformed input. Returns y sequence (N,B,m)."""
    N, B, m = x.shape
    h = dt * torch.sigmoid(c)
    y = x.new_zeros(B, m)
    z = x.new_zeros(B, m)
    ys = []
    for n in range(N):
        z = z - h * (torch.tanh(w * y + x[n]) + alpha * y)
        y = y + h * z
        ys.append(y)
    return torch.stack(ys)

# 2. Reversible recurrence with hand-written BPTT (what the CUDA kernel does)
class UnICORNNFn(torch.autograd.Function):
    """Forward stores ONLY the final (y_N, z_N) (plus x, w, c); the hidden
    states y_n, z_n are rebuilt in backward via the exact inverse (eq. 7)."""

    @staticmethod
    def forward(ctx, x, w, c, dt, alpha):
        N, B, m = x.shape
        h = dt * torch.sigmoid(c)
        y = x.new_zeros(B, m)
        z = x.new_zeros(B, m)
        out = torch.empty_like(x)
        for n in range(N):
            z = z - h * (torch.tanh(w * y + x[n]) + alpha * y)
            y = y + h * z
            out[n] = y
        ctx.save_for_backward(x, w, c)
        ctx.yN, ctx.zN = y, z
        ctx.dt, ctx.alpha = dt, alpha
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w, c = ctx.saved_tensors
        dt, alpha = ctx.dt, ctx.alpha
        N, B, m = x.shape
        sc = torch.sigmoid(c)
        h = dt * sc
        y, z = ctx.yN, ctx.zN                      # states after the last step
        Ybar = x.new_zeros(B, m)                   # adjoint of y after current step
        Zbar = x.new_zeros(B, m)                   # adjoint of z after current step
        gx = torch.empty_like(x)
        gw = x.new_zeros(B, m)
        gh = x.new_zeros(B, m)
        for n in range(N - 1, -1, -1):
            Ybar = Ybar + grad_out[n]              # loss term at y_n
            zt = Zbar + h * Ybar                   # total adjoint of z_n
            # --- reverse the step exactly (eq. 7): recover y_{n-1}, z_{n-1}
            y_prev = y - h * z
            pre = w * y_prev + x[n]
            t = torch.tanh(pre)
            s = 1.0 - t * t                        # sigma'(pre)
            g = t + alpha * y_prev
            z_prev = z + h * g
            # --- parameter / input gradients
            gx[n] = -h * zt * s
            gw = gw - h * zt * s * y_prev
            gh = gh + Ybar * z - zt * g
            # --- propagate adjoints to the previous step
            Ybar = Ybar - h * zt * (s * w + alpha)
            Zbar = zt
            y, z = y_prev, z_prev
        gc = gh.sum(0) * dt * sc * (1.0 - sc)      # chain rule through h = dt*sigmoid(c)
        return gx, gw.sum(0), gc, None, None



# 3. Model
class VarDropout(nn.Module):
    """Variational dropout: same mask for every time step. Input (N,B,m)."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(1, x.shape[1], x.shape[2]).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class UnICORNNLayer(nn.Module):
    def __init__(self, n_in, m, dt, alpha, reversible=True):
        super().__init__()
        self.V = nn.Linear(n_in, m)                     # V and bias b
        nn.init.kaiming_uniform_(self.V.weight, a=8)    # Appendix A
        nn.init.zeros_(self.V.bias)
        self.w = nn.Parameter(torch.rand(m))            # U(0,1)
        self.c = nn.Parameter(torch.empty(m).uniform_(-0.1, 0.1))
        self.dt, self.alpha, self.reversible = dt, alpha, reversible

    def forward(self, u):                               # u: (N,B,n_in)
        x = self.V(u)                                   # parallel over time
        if self.reversible:
            return UnICORNNFn.apply(x, self.w, self.c, self.dt, self.alpha)
        return reference_recurrence(x, self.w, self.c, self.dt, self.alpha)


class UnICORNN(nn.Module):
    def __init__(self, n_in, m, n_out, L, dt, alpha, dropout=0.1,
                 dt_first=None, reversible=True):
        super().__init__()
        layers, drops = [], []
        for l in range(L):
            layer_dt = dt_first if (l == 0 and dt_first is not None) else dt
            layers.append(UnICORNNLayer(n_in if l == 0 else m, m, layer_dt, alpha, reversible))
            drops.append(VarDropout(dropout))
        self.layers, self.drops = nn.ModuleList(layers), nn.ModuleList(drops)
        self.readout = nn.Linear(m, n_out)

    def forward(self, u):                               # u: (N,B,n_in)
        h = u
        for layer, drop in zip(self.layers, self.drops):
            h = drop(layer(h))
        return self.readout(h[-1])                      # last time step



# 4. Sanity checks
def check_gradients():
    torch.manual_seed(0)
    N, B, m = 50, 4, 6
    dt, alpha = 0.2, 3.0
    x = torch.randn(N, B, m, dtype=torch.float64, requires_grad=True)
    w = torch.rand(m, dtype=torch.float64, requires_grad=True)
    c = torch.randn(m, dtype=torch.float64, requires_grad=True)
    tgt = torch.randn(N, B, m, dtype=torch.float64)

    def loss(fn):
        return 0.5 * ((fn() - tgt) ** 2).sum()

    l1 = loss(lambda: reference_recurrence(x, w, c, dt, alpha))
    g1 = torch.autograd.grad(l1, [x, w, c])
    l2 = loss(lambda: UnICORNNFn.apply(x, w, c, dt, alpha))
    g2 = torch.autograd.grad(l2, [x, w, c])
    for name, a, b in zip("xwc", g1, g2):
        print(f"[grad check] d/d{name}: max abs diff = {(a - b).abs().max().item():.2e}")


def check_reversibility(dtype, N=1000):
    torch.manual_seed(0)
    B, m, dt, alpha = 2, 8, 0.1, 5.0
    x = torch.randn(N, B, m, dtype=dtype)
    w = torch.rand(m, dtype=dtype)
    c = torch.zeros(m, dtype=dtype)
    h = dt * torch.sigmoid(c)
    y = torch.zeros(B, m, dtype=dtype); z = torch.zeros(B, m, dtype=dtype)
    ys, zs = [], []
    for n in range(N):
        z = z - h * (torch.tanh(w * y + x[n]) + alpha * y); y = y + h * z
        ys.append(y); zs.append(z)
    err = 0.0
    for n in range(N - 1, 0, -1):                      # walk back, compare to stored
        y_prev = y - h * z
        z = z + h * (torch.tanh(w * y_prev + x[n]) + alpha * y_prev)
        y = y_prev
        err = max(err, (y - ys[n - 1]).abs().max().item(), (z - zs[n - 1]).abs().max().item())
    print(f"[reversibility] {str(dtype):15s} N={N}: max reconstruction error = {err:.2e}")


def check_jacobian_det():
    """One-step Jacobian of (y,z)->(y',z') for a random neuron: det must be 1."""
    torch.manual_seed(1)
    h, alpha = 0.15, 4.0
    for _ in range(3):
        w, x = torch.randn(1).double(), torch.randn(1).double()
        v = torch.randn(2, dtype=torch.float64)

        def step(v):
            y, z = v[0], v[1]
            z2 = z - h * (torch.tanh(w * y + x) + alpha * y)
            return torch.stack([(y + h * z2).squeeze(), z2.squeeze()])

        J = torch.autograd.functional.jacobian(step, v)
        print(f"[jacobian] det = {torch.det(J).item():.15f}   (trace = {torch.trace(J).item():.3f})")


def tiny_training_demo():
    """Toy long-memory task: classify sign of the FIRST input; the rest is noise."""
    torch.manual_seed(0)
    N, B = 200, 64
    model = UnICORNN(1, 32, 2, L=2, dt=0.1, alpha=1.0, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for it in range(301):
        u = 0.3 * torch.randn(N, B, 1)
        label = (torch.rand(B) > 0.5).long()
        u[0, :, 0] = (2 * label - 1).float() * 2.0
        loss = nn.functional.cross_entropy(model(u), label)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            acc = (model(u).argmax(1) == label).float().mean().item()
            print(f"[demo] iter {it:3d}  loss {loss.item():.3f}  acc {acc:.2f}")


if __name__ == "__main__":
    check_gradients()
    check_reversibility(torch.float64)
    check_reversibility(torch.float32)
    check_jacobian_det()
    tiny_training_demo()
