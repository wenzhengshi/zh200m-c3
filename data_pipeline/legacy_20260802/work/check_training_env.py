import os

print("nvidia-smi:")
print(os.popen("nvidia-smi 2>&1").read()[:1500])
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        props = torch.cuda.get_device_properties(0)
        print("mem GB:", round(props.total_memory / 1e9, 1))
except Exception as e:
    print("torch import error:", e)
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception:
    pass
