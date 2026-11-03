"""Make the learning-rate drop of the authors' noisy_cifar10_task.py configurable (idempotent).
The original drops the learning rate only when (epoch+1) % 250 == 0, so a run shorter than
250 epochs never gets a drop. This adds --reduce_point (default 250 = original behaviour).
Usage:
    python patch_cifar_task.py unicornn/noisy_CIFAR10/noisy_cifar10_task.py
"""
import sys

path = sys.argv[1]
s = open(path).read()
if 'reduce_point' in s:
    print('already patched')
    sys.exit(0)

a = "args = parser.parse_args()"
assert a in s, 'parse_args line not found'
s = s.replace(a,
    "parser.add_argument('--reduce_point', type=int, default=250,\n"
    "                    help='divide the learning rate by 10 after this epoch')\n" + a, 1)

b = "if (epoch + 1) % 250 == 0:"
assert b in s, 'lr-drop condition not found'
s = s.replace(b, "if (epoch + 1) == args.reduce_point:", 1)

open(path, 'w').write(s)
print('patched', path)
