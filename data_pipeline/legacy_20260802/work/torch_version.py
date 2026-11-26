try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("no torch:", e)
