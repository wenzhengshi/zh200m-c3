#!/usr/bin/env python3
"""Minimal protobuf wire-format decoder for the SentencePiece model proto (read-only)."""
import json
import struct
import sys

import sentencepiece as spm

MODEL = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model"
sp = spm.SentencePieceProcessor(model_file=MODEL)
data = sp.serialized_model_proto()


def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def decode_message(buf, pos, end, depth=0):
    fields = []
    while pos < end:
        key, pos = read_varint(buf, pos)
        field_num = key >> 3
        wire = key & 7
        if wire == 0:
            val, pos = read_varint(buf, pos)
            fields.append((field_num, "varint", val))
        elif wire == 1:
            val = struct.unpack_from("<Q", buf, pos)[0]
            pos += 8
            fields.append((field_num, "fixed64", val))
        elif wire == 2:
            ln, pos = read_varint(buf, pos)
            payload = buf[pos:pos + ln]
            pos += ln
            try:
                sub = decode_message(payload, 0, len(payload), depth + 1)
                fields.append((field_num, "message", sub))
            except Exception:
                fields.append((field_num, "bytes", payload.decode("utf-8", errors="replace")))
        elif wire == 5:
            val = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            fields.append((field_num, "fixed32", val))
        else:
            raise ValueError(f"unsupported wire type {wire} at {pos}")
    return fields


fields = decode_message(data, 0, len(data))


def dump(fields, indent=0):
    pad = "  " * indent
    for fnum, wtype, val in fields:
        if wtype == "message":
            print(f"{pad}field {fnum}: message")
            dump(val, indent + 1)
        else:
            print(f"{pad}field {fnum} [{wtype}]: {val}")


dump(fields)
