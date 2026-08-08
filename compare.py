import sys
import numpy as np

def load(path):
    # .npy = PyTorch oracle (has a header), anything else = raw float32 from C++
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32).ravel()
    return np.fromfile(path, dtype=np.float32)

a, b = load(sys.argv[1]), load(sys.argv[2])

if a.size != b.size:
    print(f"SIZE MISMATCH: {a.size} vs {b.size}")
    sys.exit(1)

diff = np.abs(a - b)
max_abs = diff.max()
max_rel = max_abs / (np.abs(a).max() + 1e-9)
worst = int(diff.argmax())

print(f"n={a.size}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
print(f"worst at index {worst}: oracle={a[worst]:.6f}  mine={b[worst]:.6f}")
print("MATCH" if max_rel < 1e-4 else "MISMATCH")
sys.exit(0 if max_rel < 1e-4 else 1)