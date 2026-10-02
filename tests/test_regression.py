"""Episode-for-episode regression against recorded outcomes.

The physics is deterministic for a fixed scene, so a refactor that preserves
behaviour reproduces every episode exactly: outcome, furthest phase, sim time
and the cylinder's final pose. `golden.json` was recorded from the pre-cleanup
code (`dev` at 385a5b1); it covers the pure IK pipeline on `shipped` and
`hl_full` and the direct HL on `hl_full`, successes and a failure.

The episodes run in parallel, one fresh process each, so a commander installed
for an HL case can never leak into an IK case. Takes a few minutes.

    python -m pytest tests/test_regression.py -q
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())
HL_MODEL = REPO / "hl" / "models" / "direct.pt"


def _fly(case):
  if case["mode"] == "hl":
    os.environ["G1HL_COMMANDER"] = f"direct:{HL_MODEL}"
  else:
    os.environ.pop("G1HL_COMMANDER", None)
  sys.path.insert(0, str(REPO))
  sys.path.insert(0, str(REPO / "eval"))
  import sweep
  bounds, cyl_nominal = sweep.table_bounds(
    sweep.PRESETS[case["preset"]].get("cyl_margin_m", 0.045))
  spec = sweep.make_episode(case["ep"], case["seed"], case["preset"], bounds,
                            cyl_nominal)
  opts = {"video": False, "sim_limit": 300.0, "wall_limit": 600.0}
  rec, _ = sweep.run_episode((spec, opts))
  return rec


@pytest.fixture(scope="module")
def flown():
  with mp.get_context("spawn").Pool(len(GOLDEN), maxtasksperchild=1) as pool:
    return pool.map(_fly, GOLDEN)


@pytest.mark.parametrize("i", range(len(GOLDEN)),
                         ids=[f"{g['mode']}-{g['preset']}-ep{g['ep']}"
                              for g in GOLDEN])
def test_episode_reproduces(flown, i):
  want, got = GOLDEN[i], flown[i]
  assert not got.get("error"), got.get("error")
  for key in ("success", "failure", "phase", "sim_time_s", "cyl_final"):
    assert got[key] == want[key], (key, got[key], want[key])
