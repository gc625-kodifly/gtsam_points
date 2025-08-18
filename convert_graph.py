#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from pathlib import Path

def quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    # Normalize to be safe
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = w/n, x/n, y/n, z/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1.0 - 2.0*(yy + zz),     2.0*(xy - wz),         2.0*(xz + wy)],
        [    2.0*(xy + wz),   1.0 - 2.0*(xx + zz),       2.0*(yz - wx)],
        [    2.0*(xz - wy),       2.0*(yz + wx),     1.0 - 2.0*(xx + yy)]
    ], dtype=np.float64)
    return R

def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) != 8:
        raise ValueError(
            f"Expected 8 tokens: name tx ty tz qx qy qz qw. Got {len(parts)} -> {parts}"
        )
    name = parts[0]
    tx, ty, tz = map(float, parts[1:4])
    qx, qy, qz, qw = map(float, parts[4:8])  # XYZW
    return name, tx, ty, tz, qx, qy, qz, qw

def format_cloudcompare(name: str, T: np.ndarray) -> str:
    rows = [f"{name}:"]
    for r in range(4):
        rows.append(" " + "  ".join(f"{T[r,c]: .6f}" for c in range(4)))
    return "\n".join(rows) + "\n"

def format_brackets(name: str, T: np.ndarray) -> str:
    rows = []
    rows.append(f"{name}: [[ " + ",  ".join(f"{T[0,c]: .6f}" for c in range(4)) + "],")
    rows.append("         [ " + ",  ".join(f"{T[1,c]: .6f}" for c in range(4)) + "],")
    rows.append("         [ " + ",  ".join(f"{T[2,c]: .6f}" for c in range(4)) + "],")
    rows.append("         [ " + ",  ".join(f"{T[3,c]: .6f}" for c in range(4)) + "]]")
    return "\n".join(rows) + "\n"

def main():
    ap = argparse.ArgumentParser(
        description="Convert 'name tx ty tz qx qy qz qw' lines to 4x4 transforms."
    )
    ap.add_argument("input", help="Input .txt file")
    ap.add_argument("-o", "--output",
                    help="Output file (default: <input>_cloudcompare.txt)")
    ap.add_argument("--format", choices=["cloudcompare", "brackets"],
                    default="cloudcompare", help="Output format (default: cloudcompare)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else \
               in_path.with_name(in_path.stem + ("_cloudcompare.txt" if args.format=="cloudcompare" else "_brackets.txt"))

    lines = [ln for ln in in_path.read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]

    out_blocks = []
    for ln in lines:
        name, tx, ty, tz, qx, qy, qz, qw = parse_line(ln)
        R = quat_to_rot(qx, qy, qz, qw)  # your function, XYZW order
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        block = format_cloudcompare(name, T) if args.format == "cloudcompare" else format_brackets(name, T)
        out_blocks.append(block)

        # quick proof: orthogonality + determinant
        ortho_err = np.linalg.norm(R.T @ R - np.eye(3))
        det = np.linalg.det(R)
        out_blocks.append(f"# check {name}: ortho_err={ortho_err:.3e}, det={det:.6f}\n")

    out_path.write_text("\n".join(out_blocks))
    print(f"✅ Wrote {len(lines)} transforms to {out_path}")

if __name__ == "__main__":
    main()
