for m in ("gguf", "onnx", "onnxruntime", "sentencepiece", "torch", "numpy"):
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
