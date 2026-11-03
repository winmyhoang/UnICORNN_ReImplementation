"""Log the test accuracy of every epoch in the authors' noisy_cifar10_task.py (idempotent).
The original script computes it each epoch but writes only the validation accuracy; the test
accuracy at the best validation epoch is written only after the last epoch, so a run that is cut
off (for example by a disconnected Colab session) loses its result. After this patch every epoch
writes two lines:
    eval accuracy: <validation accuracy>
    test accuracy: <test accuracy of the same epoch>
The test accuracy at the best validation epoch can then be read from the log.
Usage:
    python patch_cifar_log.py unicornn/noisy_CIFAR10/noisy_cifar10_task.py
"""
import sys

path = sys.argv[1]
s = open(path).read()
marker = "    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\\n')\n"
if "f.write('test accuracy: '" in s:
    print('already patched')
    sys.exit(0)
assert marker in s, 'logging line not found'
s = s.replace(marker, marker + "    f.write('test accuracy: ' + str(round(test_acc, 2)) + '\\n')\n", 1)
open(path, 'w').write(s)
print('patched', path)
