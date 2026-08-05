"""
Utility script to convert Python embeddings.npz to C binary database format (embeddings.bin)
and vice-versa.

Binary Format Specification:
- Header: 8 bytes magic ("FACES1\0\0")
- Count: int32 (number of employees)
- Dim: int32 (embedding dimension, usually 512)
- Body: Repeated per employee:
    - 128 bytes: null-padded UTF-8 employee_id string
    - 512 * 4 bytes: float32 L2-normalized embedding vector
"""

import sys
import struct
from pathlib import Path
import numpy as np


def convert_npz_to_bin(npz_path: str, bin_path: str) -> None:
    print(f"Reading '{npz_path}'...")
    data = np.load(npz_path)
    employee_ids = data["employee_ids"]
    templates = data["templates"]

    count = len(employee_ids)
    dim = templates.shape[1] if templates.ndim > 1 else len(templates[0])

    print(f"Loaded {count} employees with dimension {dim}.")
    print(f"Exporting to C binary format '{bin_path}'...")

    with open(bin_path, "wb") as f:
        # Magic: 8 bytes
        f.write(b"FACES1\x00\x00")
        # Header: count (int32), dim (int32)
        f.write(struct.pack("<ii", count, dim))

        for i in range(count):
            emp_id = str(employee_ids[i]).encode("utf-8")[:127]
            emp_id_padded = emp_id.ljust(128, b"\x00")
            f.write(emp_id_padded)

            vec = templates[i].astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            f.write(vec.tobytes())

    print(f"Successfully created '{bin_path}' with {count} employee templates.")


def dump_bin_info(bin_path: str) -> None:
    print(f"Inspecting C binary file: '{bin_path}'...")
    with open(bin_path, "rb") as f:
        magic = f.read(8)
        if magic[:6] != b"FACES1":
            print("Invalid magic header!")
            return
        count, dim = struct.unpack("<ii", f.read(8))
        print(f"Total Enrolled Employees: {count}")
        print(f"Embedding Dimension    : {dim}")
        print("-" * 50)
        for i in range(count):
            id_bytes = f.read(128).split(b"\x00")[0]
            vec_bytes = f.read(dim * 4)
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            emp_id = id_bytes.decode("utf-8", errors="replace")
            print(f"[{i+1:03d}] ID: {emp_id:<20} | L2 Norm: {np.linalg.norm(vec):.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_npz_to_bin.py <input.npz> [output.bin]")
        print("  python convert_npz_to_bin.py --info <input.bin>")
        sys.exit(1)

    if sys.argv[1] == "--info":
        dump_bin_info(sys.argv[2])
    else:
        in_file = sys.argv[1]
        out_file = sys.argv[2] if len(sys.argv) > 2 else "data/embeddings.bin"
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        convert_npz_to_bin(in_file, out_file)
