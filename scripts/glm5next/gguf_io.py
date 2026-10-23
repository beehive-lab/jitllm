"""Minimal GGUF and safetensors I/O for the GLM-5.3-Flash (glm5next) tools.

Only numpy is required. The safetensors reader maps tensors lazily, so a
converter can read a few tensors from a shard without loading it. The GGUF
writer takes every tensor's shape and type up front, writes the header, and
then streams the tensor data in the same order.
"""

import json
import mmap
import os
import struct

import numpy as np

# GGUF value types
GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16 = 0, 1, 2, 3
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL = 4, 5, 6, 7
GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64 = 8, 9, 10, 11, 12

# GGML tensor types
F32, F16, Q4_0, Q8_0, BF16 = 0, 1, 2, 8, 30
TYPE_NAMES = {F32: "F32", F16: "F16", Q4_0: "Q4_0", Q8_0: "Q8_0", BF16: "BF16"}
TYPE_BY_NAME = {v: k for k, v in TYPE_NAMES.items()}
BLOCK = {F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q8_0: (32, 34), Q4_0: (32, 18)}

ALIGNMENT = 32


# ---------------------------------------------------------------- dtype decode

def bf16_to_f32(raw_u16):
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def _e4m3fn_table():
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0xF and man == 0x7:
            table[b] = np.nan
        elif exp == 0:
            table[b] = sign * (man / 8.0) * 2.0 ** -6
        else:
            table[b] = sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)
    return table


E4M3FN = _e4m3fn_table()


def f32_to_e4m3fn(x):
    """Round-to-nearest-even encode used only by the tiny reference checkpoint."""
    x = np.asarray(x, dtype=np.float32)
    finite = E4M3FN[:0x7F].copy()  # positive codes 0..126, 0x7F is NaN
    mag = np.clip(np.abs(x), 0, 448.0)
    idx = np.searchsorted(finite, mag)
    idx = np.clip(idx, 1, len(finite) - 1)
    lo, hi = finite[idx - 1], finite[idx]
    pick_hi = (mag - lo > hi - mag) | ((mag - lo == hi - mag) & (idx % 2 == 0))
    code = np.where(pick_hi, idx, idx - 1).astype(np.uint8)
    code = np.where(x < 0, code | 0x80, code).astype(np.uint8)
    return code


# ---------------------------------------------------------------- safetensors

_ST_DTYPES = {
    "F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (np.uint16, 2),
    "F8_E4M3": (np.uint8, 1), "I64": (np.int64, 8), "I32": (np.int32, 4),
    "U8": (np.uint8, 1), "I8": (np.int8, 1),
}


class SafetensorsFile:
    """Lazy reader for one .safetensors file."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "rb")
        n = struct.unpack("<Q", self._f.read(8))[0]
        header = json.loads(self._f.read(n))
        header.pop("__metadata__", None)
        self.header = header
        self.data_start = 8 + n
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return self.header.keys()

    def dtype(self, name):
        return self.header[name]["dtype"]

    def shape(self, name):
        return list(self.header[name]["shape"])

    def raw(self, name):
        info = self.header[name]
        np_dtype, _ = _ST_DTYPES[info["dtype"]]
        begin, end = info["data_offsets"]
        buf = np.frombuffer(self._mm, dtype=np.uint8, count=end - begin, offset=self.data_start + begin)
        return buf.view(np_dtype).reshape(info["shape"])

    def close(self):
        self._mm.close()
        self._f.close()


class Checkpoint:
    """A Hugging Face checkpoint directory read through model.safetensors.index.json."""

    def __init__(self, directory):
        self.directory = directory
        index_path = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = SafetensorsFile(os.path.join(directory, "model.safetensors"))
            self.weight_map = {k: "model.safetensors" for k in single.keys()}
            single.close()
        self._open = {}

    def has(self, name):
        return name in self.weight_map

    def shard_of(self, name):
        return self.weight_map[name]

    def _file(self, name):
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = SafetensorsFile(os.path.join(self.directory, shard))
        return self._open[shard]

    def shape(self, name):
        return self._file(name).shape(name)

    def dtype(self, name):
        return self._file(name).dtype(name)

    def tensor_f32(self, name):
        """Returns the tensor as float32, dequantizing FP8 block-scaled weights."""
        f = self._file(name)
        dt = f.dtype(name)
        raw = f.raw(name)
        if dt == "F32":
            out = np.array(raw, dtype=np.float32)
        elif dt == "F16":
            out = raw.astype(np.float32)
        elif dt == "BF16":
            out = bf16_to_f32(raw)
        elif dt == "F8_E4M3":
            out = E4M3FN[raw]
            scale_name = name + "_scale_inv"
            if self.has(scale_name):
                scale = self.tensor_f32(scale_name)
                out = dequant_block_scale(out, scale)
        else:
            raise ValueError(f"{name}: unsupported dtype {dt}")
        return out

    def close_unused(self, keep=()):
        for shard in list(self._open):
            if shard not in keep:
                self._open.pop(shard).close()


def dequant_block_scale(w, scale_inv):
    rows, cols = w.shape
    br = -(-rows // scale_inv.shape[0])
    bc = -(-cols // scale_inv.shape[1])
    s = np.repeat(np.repeat(scale_inv, br, axis=0), bc, axis=1)[:rows, :cols]
    return (w * s).astype(np.float32)


def write_safetensors(path, tensors):
    """tensors: dict name -> (st_dtype, numpy array holding the raw element values)."""
    header = {}
    offset = 0
    blobs = []
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


# ---------------------------------------------------------------- quantization

def _round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_q8_0(x):
    """llama.cpp quantize_row_q8_0_ref over the last axis; returns raw bytes."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)
    q = _round_away(x * inv[:, None]).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def dequantize_q8_0(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    q = b[:, 2:].view(np.int8).astype(np.float32)
    return (q * d[:, None]).reshape(-1)[:n]


def quantize_q4_0(x):
    """llama.cpp quantize_row_q4_0_ref over the last axis; returns raw bytes."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    idx = np.abs(x).argmax(axis=1)
    mx = x[np.arange(x.shape[0]), idx]
    d = mx / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)
    v = x * inv[:, None] + 8.5
    q = np.minimum(15, np.trunc(v).astype(np.int32)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()


def dequantize_q4_0(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = b[:, 2:]
    lo = (qs & 0x0F).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    return (np.concatenate([lo, hi], axis=1) * d[:, None]).reshape(-1)[:n]


def encode(x, ggml_type):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if ggml_type == F32:
        return x.tobytes()
    if ggml_type == F16:
        return x.astype(np.float16).tobytes()
    if ggml_type == BF16:
        return (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    if ggml_type == Q8_0:
        return quantize_q8_0(x)
    if ggml_type == Q4_0:
        return quantize_q4_0(x)
    raise ValueError(ggml_type)


def decode(raw, ggml_type, n):
    """Inverse of encode, used to make reference weights exactly representable."""
    if ggml_type == F32:
        return np.frombuffer(raw, dtype=np.float32)[:n].copy()
    if ggml_type == F16:
        return np.frombuffer(raw, dtype=np.float16)[:n].astype(np.float32)
    if ggml_type == BF16:
        return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)[:n])
    if ggml_type == Q8_0:
        return dequantize_q8_0(raw, n)
    if ggml_type == Q4_0:
        return dequantize_q4_0(raw, n)
    raise ValueError(ggml_type)


def nbytes(shape_ne, ggml_type):
    n = int(np.prod(shape_ne))
    block, size = BLOCK[ggml_type]
    if n % block:
        raise ValueError(f"{n} elements not a multiple of block {block}")
    return n // block * size


# ---------------------------------------------------------------- GGUF writer

def _pack_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_value(vtype, value):
    if vtype == GGUF_STRING:
        return _pack_string(value)
    fmt = {GGUF_UINT8: "<B", GGUF_INT8: "<b", GGUF_UINT16: "<H", GGUF_INT16: "<h",
           GGUF_UINT32: "<I", GGUF_INT32: "<i", GGUF_FLOAT32: "<f", GGUF_BOOL: "<?",
           GGUF_UINT64: "<Q", GGUF_INT64: "<q", GGUF_FLOAT64: "<d"}[vtype]
    return struct.pack(fmt, value)


class GGUFWriter:
    def __init__(self, path, architecture):
        self.path = path
        self.kv = []
        self.tensors = []  # (name, ne, type, nbytes)
        self._names = set()
        self._written = 0
        self._f = None
        self.add_string("general.architecture", architecture)
        self.add_uint32("general.alignment", ALIGNMENT)

    # metadata
    def add(self, key, vtype, value):
        self.kv.append((key, vtype, value))

    def add_string(self, key, v):
        self.add(key, GGUF_STRING, v)

    def add_uint32(self, key, v):
        self.add(key, GGUF_UINT32, int(v))

    def add_int32(self, key, v):
        self.add(key, GGUF_INT32, int(v))

    def add_float32(self, key, v):
        self.add(key, GGUF_FLOAT32, float(v))

    def add_bool(self, key, v):
        self.add(key, GGUF_BOOL, bool(v))

    def add_array(self, key, elem_type, values):
        self.add(key, GGUF_ARRAY, (elem_type, list(values)))

    # tensors
    def add_tensor_info(self, name, ne, ggml_type):
        if name in self._names:
            raise ValueError(f"duplicate tensor {name}")
        self._names.add(name)
        ne = [int(v) for v in ne]
        self.tensors.append((name, ne, ggml_type, nbytes(ne, ggml_type)))

    def write_header(self):
        self._f = open(self.path, "wb")
        f = self._f
        f.write(struct.pack("<IIQQ", 0x46554747, 3, len(self.tensors), len(self.kv)))
        for key, vtype, value in self.kv:
            f.write(_pack_string(key))
            f.write(struct.pack("<I", vtype))
            if vtype == GGUF_ARRAY:
                etype, values = value
                f.write(struct.pack("<IQ", etype, len(values)))
                for v in values:
                    f.write(_pack_value(etype, v))
            else:
                f.write(_pack_value(vtype, value))
        offset = 0
        self._offsets = []
        for name, ne, ggml_type, size in self.tensors:
            f.write(_pack_string(name))
            f.write(struct.pack("<I", len(ne)))
            for v in ne:
                f.write(struct.pack("<Q", v))
            f.write(struct.pack("<IQ", ggml_type, offset))
            self._offsets.append(offset)
            offset += size + (-size % ALIGNMENT)
        f.write(b"\0" * (-f.tell() % ALIGNMENT))
        self._data_start = f.tell()

    def write_tensor(self, name, data):
        """Writes the next tensor's raw bytes; tensors must come in add_tensor_info order."""
        self.write_tensor_parts(name, [data])

    def write_tensor_parts(self, name, parts):
        """Like write_tensor, with the bytes given as an iterable of pieces (e.g. one per expert)."""
        expected_name, _, _, size = self.tensors[self._written]
        if name != expected_name:
            raise ValueError(f"expected tensor {expected_name}, got {name}")
        written = 0
        for data in parts:
            self._f.write(data)
            written += len(data)
        if written != size:
            raise ValueError(f"{name}: {written} bytes, expected {size}")
        self._f.write(b"\0" * (-size % ALIGNMENT))
        self._written += 1

    def close(self):
        if self._written != len(self.tensors):
            raise ValueError(f"wrote {self._written} of {len(self.tensors)} tensors")
        self._f.close()


# ---------------------------------------------------------------- GGUF reader

class GGUFReader:
    """Reads metadata and tensors back; used to check converter output."""

    def __init__(self, path):
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._pos = 0
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", self._read(24))
        if magic != 0x46554747:
            raise ValueError("not a GGUF file")
        self.version = version
        self.kv = {}
        for _ in range(n_kv):
            key = self._string()
            vtype = struct.unpack("<I", self._read(4))[0]
            self.kv[key] = self._value(vtype)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._string()
            nd = struct.unpack("<I", self._read(4))[0]
            ne = list(struct.unpack(f"<{nd}Q", self._read(8 * nd)))
            ggml_type, offset = struct.unpack("<IQ", self._read(12))
            self.tensors[name] = (ne, ggml_type, offset)
        align = self.kv.get("general.alignment", ALIGNMENT)
        self.data_start = self._pos + (-self._pos % align)

    def _read(self, n):
        b = self._mm[self._pos:self._pos + n]
        self._pos += n
        return b

    def _string(self):
        n = struct.unpack("<Q", self._read(8))[0]
        return self._read(n).decode("utf-8")

    def _value(self, vtype):
        if vtype == GGUF_STRING:
            return self._string()
        if vtype == GGUF_ARRAY:
            etype, n = struct.unpack("<IQ", self._read(12))
            return [self._value(etype) for _ in range(n)]
        fmt = {GGUF_UINT8: "<B", GGUF_INT8: "<b", GGUF_UINT16: "<H", GGUF_INT16: "<h",
               GGUF_UINT32: "<I", GGUF_INT32: "<i", GGUF_FLOAT32: "<f", GGUF_BOOL: "<?",
               GGUF_UINT64: "<Q", GGUF_INT64: "<q", GGUF_FLOAT64: "<d"}[vtype]
        return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]

    def tensor_f32(self, name):
        ne, ggml_type, offset = self.tensors[name]
        n = int(np.prod(ne))
        size = nbytes(ne, ggml_type)
        start = self.data_start + offset
        raw = self._mm[start:start + size]
        return decode(raw, ggml_type, n).reshape(list(reversed(ne)))
