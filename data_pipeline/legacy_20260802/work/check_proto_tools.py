mods = ["google.protobuf", "sentencepiece_model_pb2"]
for m in mods:
    try:
        __import__(m)
        print(m, "OK")
    except Exception as e:
        print(m, "MISSING", type(e).__name__, e)
