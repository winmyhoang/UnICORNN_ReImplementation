"""Add two options to the authors' psMNIST_task.py (idempotent):
    --epochs          total number of epochs (0 = the original 4 * reduce_point)
    --reduce_point2   optional second learning-rate drop (x0.1) after this epoch
Usage:
    python patch_psmnist_task.py unicornn/psMNIST/psMNIST_task.py
"""
import sys

path = sys.argv[1]
s = open(path).read()
if 'reduce_point2' in s:
    print('already patched')
    sys.exit(0)

# 1) new arguments, inserted before parse_args()
a = "args = parser.parse_args()"
assert a in s, 'parse_args line not found'
s = s.replace(a,
    "parser.add_argument('--epochs', type=int, default=0, help='total epochs (0 = 4*reduce_point)')\n"
    "parser.add_argument('--reduce_point2', type=int, default=0, help='optional 2nd lr drop (x0.1) after this epoch')\n"
    + a, 1)

# 2) total number of epochs
b = "for epoch in range(int(4*args.reduce_point)):"
assert b in s, 'epoch loop not found'
s = s.replace(b, "for epoch in range(args.epochs if args.epochs else int(4*args.reduce_point)):", 1)

# 3) second learning-rate drop
c = "if (epoch + 1) == args.reduce_point:"
assert c in s, 'lr-drop condition not found'
s = s.replace(c, "if (epoch + 1) == args.reduce_point or (epoch + 1) == args.reduce_point2:", 1)

open(path, 'w').write(s)
print('patched', path)
