"""Regenerate the independent numerical oracle using ObsPy 1.4.2 + NumPy.

Run from any working directory. The browser application never imports Python.
MoPaD's get_M() returns NED by default. NumPy eigh supplies independent axes
(MoPaD 1.4.2's get_p_axis() can fail on its one-dimensional vector shape).
"""
import json
from pathlib import Path
import numpy as np
import obspy
from obspy.imaging.scripts.mopad import MomentTensor

assert obspy.__version__ == "1.4.2", "Use the pinned reference version."
rng = np.random.default_rng(2580)
cases = [[0, 45, -90], [0, 45, 90], [0, 90, 0], [0, 90, 180],
         [35, 60, 125], [123, 0, 17], [360, 90, -180], [270, 90, 90],
         [42, 0.0001, -35], [85, 89.9999, 179.9999]]
cases += [[float(rng.uniform(0, 360)), float(rng.uniform(0, 90)),
           float(rng.uniform(-180, 180))] for _ in range(100)]
records = []
for sdr in cases:
    matrix = np.asarray(MomentTensor(sdr).get_M(), dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    records.append(dict(sdr=sdr, matrix=matrix.tolist(),
                        p=eigenvectors[:, 0].tolist(),
                        t=eigenvectors[:, 2].tolist()))
output = dict(reference="ObsPy 1.4.2 MoPaD get_M(NED) + numpy.linalg.eigh",
              source="https://github.com/obspy/obspy/blob/1.4.2/obspy/imaging/scripts/mopad.py",
              seed=2580, cases=records)
Path(__file__).with_name("reference.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(records)} reference mechanisms.")
