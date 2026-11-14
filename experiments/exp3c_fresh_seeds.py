"""Fresh-seed confirmation of the support-slack remedy (seeds disjoint from the pilot and the main run)."""
import json, sys, time
sys.path.insert(0, ".")
import experiments.exp3b_shift_default as b
import experiments.exp3c_shift_gate as c
from experiments.common import save_json
b.NULL_SEED0, b.SIGNAL_SEED0 = 50_000, 950_000
out = {}
for a in b.ACTIVITY_SHIFTS:
    null = b.null_part(a, c.NEW_OPTS, 1500, 3)
    sig = b.signal_part(a, c.NEW_OPTS, 200, 3)
    cell = b.assemble_cell(null, sig)
    out[str(a)] = cell
    print(a, f"PFA={cell['pfa']:.4f} CI={cell['pfa_ci']} {cell['verdict']} miss={cell['miss_rate']:.3f} cens={cell['add_censored_s']:.1f} abst={cell['abstention_null']:.3f}", flush=True)
save_json({"seed0_null": 50_000, "seed0_signal": 950_000, "variant": c.NEW_VARIANT, "opts": c.NEW_OPTS, "cells": out}, "exp3c_fresh_seeds")
print("wrote", flush=True)
