import numpy as np
from pathlib import Path

d = Path("/tmp/smoke_train_data")
d.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260802)
np.array(rng.integers(1, 32000, 16384), dtype=np.int32).tofile(d / "train.bin")
np.array(rng.integers(1, 32000, 4096), dtype=np.int32).tofile(d / "validation.bin")
print("synthetic train/val written")
