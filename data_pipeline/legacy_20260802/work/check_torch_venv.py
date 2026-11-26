import sys

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    print("threads:", torch.get_num_threads())
except Exception as e:
    print("no torch:", e)
import os
print("cpus:", os.cpu_count())
print("python:", sys.version)
