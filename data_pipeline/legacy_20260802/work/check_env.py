mods = ["numpy", "sentencepiece", "scipy", "sklearn", "minhash", "datasketch"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)

import shutil
print("ffprobe/poppler not needed")
import os
st = os.statvfs("/home/xmkjsoft")
print("disk free GB:", st.f_bavail * st.f_frsize / 1e9)
