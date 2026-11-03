# UnICORNN study: implementation and partial reproduction

An independent study of **UnICORNN** (Undamped Independent Controlled Oscillatory RNN) from

> T. K. Rusch and S. Mishra. *UnICORNN: A recurrent model for learning very long time dependencies.* ICML 2021.
> [arXiv:2103.05487](https://arxiv.org/abs/2103.05487)

The authors' official code is at <https://github.com/tk-rusch/unicornn> and is **not included here**.
This repository contains my own PyTorch implementation, scripts that run and extend the authors' code,
and small experiments. It is a study and a *partial* reproduction, not a replacement for the original code.

## The idea in four lines

Each hidden unit is an undamped, driven oscillator (position `y`, velocity `z`). One time step is

```
z_new = z - h * ( tanh(w*y + x) + alpha*y )      # velocity first
y_new = y + h * z_new                            # then position, using the NEW velocity (symplectic Euler)
```

with `x = V*(input) + b` and a learnable step `h = dt * sigmoid(c)` per neuron. The matrix of derivatives of the new state with respect to the old one (the *transition Jacobian*) has
determinant exactly 1 for this step, which means the step neither stretches nor squeezes the volume of a cloud of states; the step also has an exact inverse (`y_old = y - h*z`, then `z_old = z + h*(tanh(w*y_old + x) + alpha*y_old)`),
which removes one source of exploding or vanishing gradients (the paper adds bounds on top of this; a determinant of 1 alone
does not limit stretching in one direction, so the size of the Jacobian matters too), and hidden states can be rebuilt in the backward pass.

## Files

| File | What it does | Status |
|---|---|---|
| `unicornn.py` | My implementation: reference loop, reversible custom autograd function with hand-written backward pass, model, numerical checks, toy demo | run on Colab, all checks passed |
| `psmnist_mine.py` | Permuted sequential MNIST with my implementation (slow: Python loop) | run on Colab (CPU), short runs only |
| `lstm_vs_unicornn.py` | "Remember the first input" task: UnICORNN vs LSTM vs tanh RNN, plus a gradient-versus-lag plot | run on Colab |
| `noisy_mnist_compare.py` | Noise-padded MNIST (10 classes): UnICORNN vs LSTM vs tanh RNN | run on Colab (CPU) |
| `patch_network.py` | Makes the authors' CUDA kernels compile with `cupy.RawModule` (the original `pynvrtc` did not work on current Colab) | applied on Colab (psMNIST folder) |
| `patch_psmnist_task.py` | Adds `--epochs` and `--reduce_point2` to the authors' `psMNIST_task.py` | the same edit was applied with `sed` on Colab; this script was only tested on a copy of the file |
| `patch_cifar_task.py` | Makes the learning-rate drop of the authors' `noisy_cifar10_task.py` configurable (`--reduce_point`) | applied on a GPU runtime; a 60-epoch run was lost at epoch 55 when the session ended |
| `patch_cifar_log.py` | Also logs the test accuracy of every epoch, so a cut-off run still gives a result | only tested on a copy of the file |
| `compare_kernel_vs_mine.py` | Compares the authors' CUDA kernel with my implementation on identical inputs (outputs, gradients, timing) and with a float64 reference | **not run yet** (needs a GPU) |
| `numpy_math_check.py` | Independent NumPy check of the backward formulas (finite differences), the inverse and `det = 1` | run in a sandbox, no PyTorch needed |

## Quick start (my implementation)

Requirements: Python 3, PyTorch, NumPy, matplotlib, torchvision (for MNIST). No GPU needed.

```bash
python unicornn.py                                      # gradient, reversibility and determinant checks + toy demo
python psmnist_mine.py --epochs 1 --max_batches 50      # short psMNIST run (slow; ~1 s per batch on a CPU)
python lstm_vs_unicornn.py --skip_train                 # gradient-versus-lag plot
python lstm_vs_unicornn.py --lengths 100 300 --iters 200 --skip_grad
python noisy_mnist_compare.py --lengths 0 100 --iters 300 --eval_every 150
```

Keep `unicornn.py` in the same folder as the other scripts (they import it).

## Running the authors' code (GPU needed)

The authors' kernels need an NVIDIA GPU. On a current Colab runtime:

```bash
git clone https://github.com/tk-rusch/unicornn
pip install cupy-cuda12x                                  # match the CUDA version shown by nvidia-smi
python patch_network.py unicornn/psMNIST/network.py
python patch_psmnist_task.py unicornn/psMNIST/psMNIST_task.py
cd unicornn/psMNIST
rm -f result/psMNIST_log.txt                              # the repo ships the authors' own log; the script appends to it
python psMNIST_task.py --nhid 128 --lr 0.00114 --batch 64 --dt 0.482 --alpha 12.53 --nlayers 3 \
       --reduce_point 150 --epochs 250
```

## Results

All numbers come from **single runs** unless stated otherwise.

**Checks of my implementation** (float64 unless noted): backward pass vs PyTorch autograd, max abs. difference
`7.8e-16` (x), `1.3e-15` (w), `4.4e-15` (c). Rebuilding all states of a 1000-step sequence from the final state with the
exact inverse: error `3.3e-15` (float64), `1.8e-6` (float32). Determinant of the transition Jacobian (the factor by which one update step stretches volume): `1.000000000000000`.

**psMNIST, authors' code, 3 layers, 128 units** (test accuracy at the best validation epoch):

| Run | Epochs | LR drops after epoch | Test accuracy |
|---|---|---|---|
| A | 4 | 1 | 80.93 % |
| B | 100 | 25 | 93.64 % |
| C | 250 | 150 | 96.22 % |
| D | 230 | 130 and 190 (second drop is not in the paper) | 96.48 % |
| Paper | 2600 | 650 | 97.8 % (10-run mean 97.7 ± 0.09 %) |

My own implementation was only trained for about one pass over the data (76 % test accuracy), so the numbers above are
from the authors' CUDA code. The remaining gap to the paper is probably the much shorter schedule, but I did not test that.

**"Remember the first input"** (label in step 1, then N noise steps; success = test accuracy > 0.9; ~4,700 parameters per model):
UnICORNN with untuned settings succeeded in 3/3 seeds at N = 300 and N = 1000 (500 iterations, no clipping).
A default LSTM and a tanh RNN never succeeded in those runs. The LSTM only learned after gradient clipping, a large
forget-gate bias and a lower learning rate (for example 3/3 at N = 100, 2/3 at N = 1000 with 3000 iterations), and was unreliable.
So the LSTM *can* solve the task; UnICORNN was more robust to its settings.

**Noise-padded MNIST** (10 classes, 2 layers, ~6,900 parameters per model, 2000 iterations, test accuracy %):

| Model | N = 0 | N = 100 | N = 300 |
|---|---|---|---|
| UnICORNN (dt 0.126, alpha 13) | 91.1 | 87.3 | 83.9 |
| LSTM (forget bias 5, clipping) | 96.1 | 94.1 | 91.3 |
| tanh RNN (clipping) | 88.3 | 11.2 | 11.3 |

UnICORNN is far more robust to noise padding than a plain RNN, but an LSTM with a tuned initialisation was ahead at every
length, including without noise. I found no evidence that UnICORNN beats a well-initialised LSTM on this task.

## Not done or not tested

- The paper's other benchmarks (noise-padded CIFAR-10, EigenWorms, vital-signs prediction, IMDB). The noise-padded CIFAR-10 scripts
  here have not been run because no GPU was available.
- A direct test of the paper's gradient bounds (Props. 3.1 and 3.2). `lstm_vs_unicornn.py` plots only a proxy
  (the derivative of the output with respect to the input) at initialisation.
- Memory and speed claims. From reading the authors' code, each layer still stores its input sequence, so the practical
  memory saving is probably smaller than the headline; this is not measured.
- Multiple seeds for psMNIST and noise-padded MNIST, and the full 2600-epoch psMNIST schedule.

## Caveats

- The numbers were produced on Google Colab (CPU runtime for my own implementation, a T4 GPU runtime for the authors' code),
  so speeds from the two are not comparable.
- UnICORNN's `dt` and `alpha` were not tuned in my experiments, while the LSTM received a small search over its settings.

## Credit

The model and the benchmark setups are from the paper above and the authors' repository. Please cite the paper if you use this work.
