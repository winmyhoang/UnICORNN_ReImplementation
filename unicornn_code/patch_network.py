"""Patch the authors' network.py to compile its CUDA kernels with cupy.RawModule
(instead of the old pynvrtc), so it runs on current Colab. Usage:
    python patch_network.py path/to/network.py
"""
import re, sys
path = sys.argv[1]
s = open(path).read()
if 'RawModule' in s:
    print('already patched'); sys.exit(0)

old_imp = "from cupy.cuda import function\nfrom pynvrtc.compiler import Program\n"
assert old_imp in s, 'import block not found'
s = s.replace(old_imp, "import cupy\n")

start = s.index("class UnICORNN_compile():")
end = s.index("class UnICORNN_Compute_GPU(Function):")
new_cls = '''class _KernelWrapper():
    """Keeps the call signature of the old code: f(args=[...], block=, grid=, stream=)."""
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, args, block, grid, stream):
        with cupy.cuda.ExternalStream(stream.ptr):      # run on PyTorch's current stream
            self.kernel(grid, block, tuple(args))


class UnICORNN_compile():
    _MODULE = None

    def __init__(self):
        super(UnICORNN_compile, self).__init__()

    def get_functions(self):
        if UnICORNN_compile._MODULE is None:
            UnICORNN_compile._MODULE = cupy.RawModule(code=UnICORNN_CODE, options=('--std=c++11',))
        mod = UnICORNN_compile._MODULE
        Stream = namedtuple('Stream', ['ptr'])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        return (current_stream,
                _KernelWrapper(mod.get_function('unicornn_fwd')),
                _KernelWrapper(mod.get_function('unicornn_bwd')))


'''
s = s[:start] + new_cls + s[end:]
open(path, 'w').write(s)
print('patched', path)
