#!/usr/bin/env python3
"""Randomized evaluation harness for the G1 pick & place pipeline (spec 8.2).

Replaces the 5-case `e2e_v2_integrated.py --sweep`: every episode is drawn
from a *distribution* with a reproducible seed, scored on the spec's success
criterion (cylinder upright <= 10 deg, at rest on the blue table for 2 s), and
every non-success is auto-classified into one of six failure modes. Per-episode
records land in `eval/results/` so two controller versions can be compared.

    .venv/bin/python eval/sweep.py -n 10                  # default `spec` preset
    .venv/bin/python eval/sweep.py -n 10 --preset mild    # the documented 1/5 case
    .venv/bin/python eval/sweep.py -n 50 --jobs 8 --seed 7
    .venv/bin/python eval/sweep.py -n 2 --video           # write mp4s (slow)

Determinism: episode i draws from `default_rng([seed, i])`, so the scene for a
given (seed, i) is identical no matter what N or --jobs are, and workers never
share a stream. Physics itself is deterministic for a fixed scene.

Video is OFF by default: it dominates the wall clock (an episode is ~6 s
headless) and a parallel sweep would fight over the frame files.
"""

import argparse
import contextlib
import csv
import io
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RESULTS = Path(__file__).resolve().parent / "results"

# The six spec-8.2 buckets, in report order. Every non-success episode is
# assigned exactly one.
FAILURES = ("never_captured", "dropped_in_transport", "tipped_at_place",
            "out_of_reach", "robot_fell", "timeout")

# ------------------------------------------------------------------ #
# randomization presets
# ------------------------------------------------------------------ #
PRESETS = {
  # Spec 8.2: robot spawn +/-15 cm and +/-15 deg, cylinder anywhere on the
  # brown tabletop interior, cylinder mass and friction, table heights +/-2 cm.
  "spec": {
    "spawn_xy_m": 0.15, "spawn_yaw_deg": 15.0,
    "cyl_mode": "table", "cyl_margin_m": 0.045,
    "mass_scale": (0.6, 1.6), "friction": (1.5, 4.0), "table_dz_m": 0.02,
    "table_dxy_m": 0.0,
  },
  # What `--sweep` used to perturb: +/-8 cm spawn, +/-4 cm cylinder, nothing
  # else. Kept so the documented 1/5 baseline stays reproducible.
  "mild": {
    "spawn_xy_m": 0.08, "spawn_yaw_deg": 0.0,
    "cyl_mode": "near", "cyl_radius_m": 0.04,
    "mass_scale": None, "friction": None, "table_dz_m": 0.0,
    "table_dxy_m": 0.0,
  },
  # The hardest LEGITIMATE test: shipped physics, but the cylinder may sit
  # anywhere on the brown tabletop interior (not just near nominal). This is
  # `spec`'s spatial randomization with `shipped`'s physics; the two were
  # previously confounded, so no clean number existed for it.
  "shipped_table": {
    "spawn_xy_m": 0.15, "spawn_yaw_deg": 15.0,
    "cyl_mode": "table", "cyl_margin_m": 0.045,
    "mass_scale": None, "friction": None, "table_dz_m": 0.0,
    "table_dxy_m": 0.0,
  },
  # THE TASK AS SHIPPED. `scene.xml` fixes the cylinder's friction at
  # `3 0.1 0.01` (mu = 3.0), its density at 100, and both table heights, all
  # unchanged since the fork point 0e9d1f9. The mass/friction/table ranges in
  # the presets below are OURS, not the challenge's, and pricing the pipeline
  # against them was a self-inflicted difficulty: mu is drawn from [1.5, 4.0]
  # while the shipped value 3.0 is the band the pipeline handles best.
  # This preset keeps the legitimate robustness probe (the robot does not
  # start in a guaranteed pose, and the cylinder may sit anywhere near
  # nominal) and restores every physical property the repo shipped with.
  "shipped": {
    "spawn_xy_m": 0.15, "spawn_yaw_deg": 15.0,
    "cyl_mode": "near", "cyl_radius_m": 0.10,
    "mass_scale": None, "friction": None, "table_dz_m": 0.0,
    "table_dxy_m": 0.0,
  },
  # Spec spawn/mass/friction/table randomization, but the cylinder stays
  # within reach of the fixed pick stance, which isolates grasp robustness from
  # the "the stance is hard-coded" failure that swamps the full preset.
  "reachable": {
    "spawn_xy_m": 0.15, "spawn_yaw_deg": 15.0,
    "cyl_mode": "near", "cyl_radius_m": 0.10,
    "mass_scale": (0.6, 1.6), "friction": (1.5, 4.0), "table_dz_m": 0.02,
    "table_dxy_m": 0.0,
  },
  # `shipped` physics with the two TABLE HEIGHTS randomized +/-2 cm. The design
  # spec's 8.2 height axis existed only inside `spec`/`reachable`, so every one
  # of its 9062 recorded episodes also drew a self-chosen friction and mass,
  # and the last one ran at git 7aa5b5f -- 96 commits back, before M2.3 and all
  # of M3.x. This isolates the axis at the physics scene.xml ships with.
  #
  # It is a PAIRED contrast against `shipped`: the spawn, yaw and cylinder
  # draws come off the stream BEFORE the height draws and `mass_scale`/
  # `friction` are None in both, so for a given (seed, ep) this preset flies
  # `shipped`'s exact scene with only the tabletop heights moved.
  "shipped_dz": {
    "spawn_xy_m": 0.15, "spawn_yaw_deg": 15.0,
    "cyl_mode": "near", "cyl_radius_m": 0.10,
    "mass_scale": None, "friction": None, "table_dz_m": 0.02,
    "table_dxy_m": 0.0,
  },
  # `shipped` physics with the two TABLES SHIFTED horizontally +/-5 cm. Never
  # measured: `randomize_scene` had no XY argument before this preset existed.
  # The cylinder rides with the brown top, so this is purely "the furniture is
  # somewhere else" -- a test of whether the model-derived place point, nav
  # keep-outs and tabletop bounds really are position-agnostic. Expect
  # `STANCE_CLEAR_M` (pinned at 0.15 for the shipped 0.30 m inter-table gap)
  # and `NAV_BUDGET_S` to bind before the grasp does.
  "shipped_move": {
    "spawn_xy_m": 0.15, "spawn_yaw_deg": 15.0,
    "cyl_mode": "near", "cyl_radius_m": 0.10,
    "mass_scale": None, "friction": None, "table_dz_m": 0.0,
    "table_dxy_m": 0.05,
  },
  # M5 training distribution for the HL policy (spec 2026-09-11 §6): wide
  # spawn, both tabletops moved in height AND position, footprints scaled,
  # cylinder anywhere on the brown top. Physics as shipped. `scale`/`*_top`
  # are drawn AFTER the shared draws, so presets without them are unchanged.
  "hl_full": {
    "spawn_xy_m": 0.30, "spawn_yaw_deg": 45.0,
    "cyl_mode": "table", "cyl_margin_m": 0.05,
    "mass_scale": None, "friction": None, "table_dz_m": 0.0,
    "table_dxy_m": 0.15,
    "scale": (0.75, 1.25), "brown_top": (0.60, 0.80), "blue_top": (0.50, 0.72),
  },
}

# ------------------------------------------------------------------ #
# the table-geometry LADDER
# ------------------------------------------------------------------ #
# `shipped_dz` at +/-2 cm cost 4.2 points (425/450 -> 406/450, paired
# +18/-37, p = 0.0145) with every grasp instrument flat across the range --
# so the size of the offset did not predict the loss, and 12% of episodes
# churned. That leaves one question unanswered: is 4.2 points a HEIGHT
# sensitivity, or just this pipeline's chaos floor? Re-spawning the cylinder
# at its rest height (identical rest pose) already flips the nominal run.
#
# A ladder settles it, because `rng.uniform(-d, d)` is `d * uniform(-1, 1)`
# off the same stream position. Every rung therefore draws the SAME
# perturbation DIRECTION for a given (seed, ep) and scales only its
# magnitude, and all of them share `shipped`'s spawn, yaw and cylinder draws.
# The ladder is one dose-response curve on one set of episodes:
#
#   dz   0.0001 -> `shipped_dz0`   the chaos floor: a 0.1 mm shift, which no
#                                  mechanism can care about. Whatever this
#                                  rung churns is the floor every other rung
#                                  must be read against.
#        0.02   -> `shipped_dz`    the design spec's 8.2 axis (measured)
#        0.04   -> `shipped_dz4`
#        0.08   -> `shipped_dz8`   brown top at 63-79 cm, against a pelvis at
#                                  ~79 cm -- expected to break something
#
#   dxy  0.02   -> `shipped_move2`
#        0.05   -> `shipped_move`
#        0.10   -> `shipped_move10` worst case closes the 0.30 m inter-table
#                                  gap to 0.10 m. The robot is wider than
#                                  that, so `nav_path` should run out of
#                                  route before the grasp runs out of reach.
#
# NOTE: the tables are static bodies whose legs are children of the top, so a
# height rung leaves the legs hovering above (or sunk into) the floor. They
# are collision geoms with no dynamics of their own, so this changes nothing
# the robot can feel -- but it does mean a rendered video of a tall rung
# looks wrong, and that is cosmetic.
def _rung(base, **over):
  cfg = dict(PRESETS[base])
  cfg.update(over)
  return cfg


PRESETS.update({
  "shipped_dz0": _rung("shipped", table_dz_m=0.0001),
  "shipped_dz4": _rung("shipped", table_dz_m=0.04),
  "shipped_dz8": _rung("shipped", table_dz_m=0.08),
  "shipped_move2": _rung("shipped", table_dxy_m=0.02),
  "shipped_move10": _rung("shipped", table_dxy_m=0.10),
})

NOMINAL_SPAWN = (-1.2, 0.15)


def table_bounds(margin):
  """Brown tabletop interior, read from scene.xml (never hard-coded)."""
  import mujoco

  from ik import pipeline as e2
  model = mujoco.MjModel.from_xml_path(str(REPO / "scene.xml"))
  x0, x1, y0, y1 = e2.brown_table_bounds(model, margin=margin)
  cyl = model.body_pos[mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "red_block")][:2]
  return (x0, x1, y0, y1), (float(cyl[0]), float(cyl[1]))


def make_episode(idx, seed, preset, bounds, cyl_nominal, nominal=False):
  """Draw one episode's scene from an independent, reproducible stream.

  `nominal=True` returns the unperturbed scene instead: a control episode, so
  "did this change break the case the FSM was tuned on?" is one command.
  """
  import numpy as np
  cfg = PRESETS[preset]
  rng = np.random.default_rng([seed, idx])
  x0, x1, y0, y1 = bounds
  if nominal:
    # `nominal: True` makes run_episode pass cyl_xy=None, which leaves the
    # cylinder exactly as scene.xml spawns it (12 cm above the table). That
    # matters: re-spawning it at rest height instead is enough of a physics
    # difference to flip the outcome.
    return {"ep": idx, "seed": seed, "preset": preset + "/nominal",
            "nominal": True,
            "spawn": list(NOMINAL_SPAWN), "spawn_yaw_deg": 0.0,
            "cyl_xy": [round(v, 4) for v in cyl_nominal],
            "cyl_mass_scale": 1.0, "cyl_friction": None,
            "brown_dz": 0.0, "blue_dz": 0.0,
            "brown_dxy": [0.0, 0.0], "blue_dxy": [0.0, 0.0]}

  s = cfg["spawn_xy_m"]
  spawn = (NOMINAL_SPAWN[0] + rng.uniform(-s, s),
           NOMINAL_SPAWN[1] + rng.uniform(-s, s))
  yaw = np.radians(rng.uniform(-cfg["spawn_yaw_deg"], cfg["spawn_yaw_deg"]))

  if cfg["cyl_mode"] == "table":
    cyl = (rng.uniform(x0, x1), rng.uniform(y0, y1))
  else:
    r = cfg["cyl_radius_m"]
    cyl = (float(np.clip(cyl_nominal[0] + rng.uniform(-r, r), x0, x1)),
           float(np.clip(cyl_nominal[1] + rng.uniform(-r, r), y0, y1)))

  m = cfg["mass_scale"]
  f = cfg["friction"]
  dz = cfg["table_dz_m"]
  dxy = cfg["table_dxy_m"]

  def shift():
    """One table's horizontal offset, or no draw at all when the axis is off.

    The guard is what keeps every pre-existing preset replaying bit for bit:
    with `table_dxy_m = 0` the stream is never advanced.
    """
    if not dxy:
      return [0.0, 0.0]
    return [round(float(rng.uniform(-dxy, dxy)), 4),
            round(float(rng.uniform(-dxy, dxy)), 4)]

  spec = {
    "ep": idx, "seed": seed, "preset": preset,
    "spawn": [round(float(v), 4) for v in spawn],
    "spawn_yaw_deg": round(float(np.degrees(yaw)), 2),
    "cyl_xy": [round(float(v), 4) for v in cyl],
    "cyl_mass_scale": 1.0 if m is None else round(float(rng.uniform(*m)), 3),
    "cyl_friction": None if f is None else round(float(rng.uniform(*f)), 3),
    "brown_dz": 0.0 if not dz else round(float(rng.uniform(-dz, dz)), 4),
    "blue_dz": 0.0 if not dz else round(float(rng.uniform(-dz, dz)), 4),
    "brown_dxy": shift(),
    "blue_dxy": shift(),
  }

  if "scale" in cfg:
    lo, hi = cfg["scale"]
    spec["brown_scale"] = [round(float(rng.uniform(lo, hi)), 4),
                           round(float(rng.uniform(lo, hi)), 4)]
    spec["blue_scale"] = [round(float(rng.uniform(lo, hi)), 4),
                          round(float(rng.uniform(lo, hi)), 4)]
    spec["brown_dz"] = round(float(rng.uniform(*cfg["brown_top"])) - 0.733, 4)
    spec["blue_dz"] = round(float(rng.uniform(*cfg["blue_top"])) - 0.633, 4)
    # keep the cylinder on the SCALED footprint
    hx = 0.4 * spec["brown_scale"][0] - cfg["cyl_margin_m"]
    hy = 0.25 * spec["brown_scale"][1] - cfg["cyl_margin_m"]
    spec["cyl_xy"] = [round(0.351 + float(rng.uniform(-hx, hx)), 4),
                      round(float(rng.uniform(-hy, hy)), 4)]
  return spec


# ------------------------------------------------------------------ #
# failure taxonomy
# ------------------------------------------------------------------ #
def classify(rec):
  """Exactly one label per episode. Ordered rules, first match wins.

  `robot_fell` and `timeout` come first because they abort the episode
  wherever it was; the phase reached says nothing about *why* it stopped.
  Everything after that is read off the furthest phase reached:

    no side-grasp IK solution  -> out_of_reach   (geometry, not control)
    never lifted the cylinder  -> never_captured
    lifted, lost before place  -> dropped_in_transport
    arrived, never touched down-> out_of_reach if the place IK residual is
                                  large, else dropped_in_transport
    touched down, not upright  -> tipped_at_place
  """
  from ik import pipeline as e2
  if rec.get("success"):
    return "success"
  if rec.get("fell") or rec.get("abort") == "robot_fell":
    return "robot_fell"
  if rec.get("abort") == "timeout":
    return "timeout"
  if rec.get("abort") == "no_grasp_ik":
    return "out_of_reach"
  ph = rec.get("phase_idx", 0)
  idx = {n: i for i, n in enumerate(e2.PHASES)}
  if ph < idx["lifted"]:
    return "never_captured"
  if ph < idx["transported"]:
    return "dropped_in_transport"
  if not rec.get("touched"):
    resid = rec.get("place_resid_m")
    if resid is not None and resid > 0.05:
      return "out_of_reach"
    return "dropped_in_transport"
  return "tipped_at_place"


# ------------------------------------------------------------------ #
# one episode (runs in a worker process)
# ------------------------------------------------------------------ #
def run_episode(job):
  spec, opts = job
  t0 = time.time()
  buf = io.StringIO()
  try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
      # The direct HL, when asked: it owns the pick, the pipeline flies it.
      # Unset = the pure IK pipeline (`e2.COMMANDER = None`).
      from ik import pipeline as e2
      if os.environ.get("G1HL_COMMANDER"):
        from hl import commander
        commander.install().reset(spec)
      res = e2.run_once(
        spawn=tuple(spec["spawn"]),
        spawn_yaw=spec["spawn_yaw_deg"] * 3.141592653589793 / 180.0,
        cyl_xy=None if spec.get("nominal") else tuple(spec["cyl_xy"]),
        cyl_mass_scale=spec["cyl_mass_scale"],
        cyl_friction=spec["cyl_friction"],
        brown_dz=spec["brown_dz"], blue_dz=spec["blue_dz"],
        brown_dxy=tuple(spec.get("brown_dxy") or (0.0, 0.0)),
        blue_dxy=tuple(spec.get("blue_dxy") or (0.0, 0.0)),
        brown_scale=tuple(spec.get("brown_scale") or (1.0, 1.0)),
        blue_scale=tuple(spec.get("blue_scale") or (1.0, 1.0)),
        video=(f"sweep_ep{spec['ep']:03d}.mp4" if opts["video"] else None),
        snapshots=False, sim_limit=opts["sim_limit"],
        wall_limit=opts["wall_limit"])
  except BaseException:                    # a crashed episode is a data point
    res = {"phase": "start", "phase_idx": 0, "success": False, "ok": {},
           "abort": "crash", "error": traceback.format_exc(),
           "wall_time_s": round(time.time() - t0, 2)}
    buf.write(res["error"])
  rec = dict(spec)
  rec.update(res)
  rec["failure"] = classify(rec)
  rec["pid"] = os.getpid()
  return rec, buf.getvalue()


# ------------------------------------------------------------------ #
# reporting
# ------------------------------------------------------------------ #
def stats(values):
  vals = [v for v in values if v is not None]
  if not vals:
    return None
  vals.sort()
  return {"n": len(vals), "mean": round(statistics.fmean(vals), 4),
          "median": round(statistics.median(vals), 4),
          "p10": round(vals[int(0.1 * (len(vals) - 1))], 4),
          "p90": round(vals[int(0.9 * (len(vals) - 1))], 4),
          "max": round(vals[-1], 4)}


def summarize(records, meta):
  from ik import pipeline as e2
  n = len(records)
  n_ok = sum(1 for r in records if r.get("success"))
  out = []
  w = out.append
  w("")
  w("=" * 72)
  w(f"  SWEEP SUMMARY  preset={meta['preset']}  seed={meta['seed']}  "
    f"N={n}  jobs={meta['jobs']}")
  w("=" * 72)
  w(f"  SUCCESS RATE : {n_ok}/{n} = {100.0 * n_ok / max(n, 1):.0f}%"
    "   (upright <=10 deg, at rest on blue table 2 s)")
  w("")
  w("  Failure modes                     Phase reached (furthest)")
  counts = {f: 0 for f in FAILURES}
  for r in records:
    if not r.get("success"):
      counts[r["failure"]] = counts.get(r["failure"], 0) + 1
  phases = {p: 0 for p in e2.PHASES}
  for r in records:
    phases[r.get("phase", "start")] = phases.get(r.get("phase", "start"), 0) + 1
  frows = [(f, counts[f]) for f in FAILURES]
  prows = [(p, phases[p]) for p in e2.PHASES]
  for i in range(max(len(frows), len(prows))):
    left = right = ""
    if i < len(frows):
      f, c = frows[i]
      left = f"  {f:<22s} {c:>2d} {'#' * c}"
    if i < len(prows):
      p, c = prows[i]
      right = f"{p:<13s} {c:>2d} {'#' * c}"
    w(f"{left:<38s}{right}")
  n_err = sum(1 for r in records if r.get("error"))
  if n_err:
    w(f"  ({n_err} episode(s) raised an exception; see the logs)")
  w("")
  w("  Metric distributions            n   mean  median     p10     p90     max")
  metrics = [
    ("placement err (cm)", [r.get("place_err_m") for r in records
                            if r.get("touched")], 100),
    ("max tilt (deg)", [r.get("max_tilt_deg") for r in records], 1),
    ("final tilt (deg)", [r.get("final_tilt_deg") for r in records], 1),
    ("max in-palm slip (cm)", [r.get("max_slip_m") for r in records], 100),
    ("pick stance err (cm)", [r.get("pick_pos_err_m") for r in records], 100),
    ("sim time (s)", [r.get("sim_time_s") for r in records], 1),
    ("wall time (s)", [r.get("wall_time_s") for r in records], 1),
  ]
  for name, vals, scale in metrics:
    s = stats([v * scale if v is not None else None for v in vals])
    if s is None:
      w(f"  {name:<28s}   -")
    else:
      w(f"  {name:<28s} {s['n']:>3d} {s['mean']:>6.1f} {s['median']:>7.1f} "
        f"{s['p10']:>7.1f} {s['p90']:>7.1f} {s['max']:>7.1f}")
  w("")
  w("  Episodes")
  w("  ep  spawn(x,y,yaw)        cylinder(x,y)   m/mu     phase        "
    "result")
  for r in sorted(records, key=lambda r: r["ep"]):
    res = "SUCCESS" if r.get("success") else r["failure"]
    w(f"  {r['ep']:>2d}  ({r['spawn'][0]:+.2f},{r['spawn'][1]:+.2f},"
      f"{r['spawn_yaw_deg']:+5.1f})  ({r['cyl_xy'][0]:+.2f},"
      f"{r['cyl_xy'][1]:+.2f})  {r['cyl_mass_scale']:.2f}/"
      f"{(r['cyl_friction'] or 3.0):.1f}  {r.get('phase', '?'):<12s} {res}")
  w("")
  return "\n".join(out)


CSV_FIELDS = ["ep", "seed", "preset", "success", "failure", "phase",
              "phase_idx", "place_err_m", "place_err_nominal_m",
              "max_tilt_deg", "final_tilt_deg", "max_slip_m", "final_slip_m",
              "lift_rise_m", "pick_pos_err_m", "pick_yaw_err_deg",
              "place_resid_m", "crouched", "touched", "fell", "abort",
              "sim_time_s", "wall_time_s", "cyl_mass_scale", "cyl_friction",
              "brown_dz", "blue_dz", "brown_dxy", "blue_dxy",
              "spawn_yaw_deg", "brown_scale", "blue_scale"]


def write_results(run_id, records, meta, logs, report):
  RESULTS.mkdir(parents=True, exist_ok=True)
  jpath = RESULTS / f"{run_id}.json"
  with open(jpath, "w") as f:
    json.dump({"meta": meta, "episodes": sorted(records,
                                                key=lambda r: r["ep"])},
              f, indent=1)
  cpath = RESULTS / f"{run_id}.csv"
  with open(cpath, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    wr.writeheader()
    for r in sorted(records, key=lambda r: r["ep"]):
      row = dict(r)
      row["spawn_x"], row["spawn_y"] = r["spawn"]
      wr.writerow(row)
  ldir = RESULTS / "logs" / run_id
  ldir.mkdir(parents=True, exist_ok=True)
  for ep, text in logs.items():
    (ldir / f"ep{ep:03d}.log").write_text(text)
  (RESULTS / f"{run_id}.txt").write_text(report)
  return jpath, cpath, ldir


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("-n", "--episodes", type=int, default=10)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--preset", choices=sorted(PRESETS), default="spec")
  ap.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                  help="parallel worker processes (episodes are single-threaded)")
  ap.add_argument("--video", action="store_true",
                  help="write an mp4 per episode (slow; off by default)")
  ap.add_argument("--sim-limit", type=float, default=300.0,
                  help="per-episode sim-seconds budget -> `timeout`")
  ap.add_argument("--wall-limit", type=float, default=600.0,
                  help="per-episode wall-seconds budget -> `timeout`")
  ap.add_argument("--nominal-first", action="store_true",
                  help="make episode 0 the unperturbed nominal scene (control)")
  ap.add_argument("--tag", default="", help="label prefix for the result files")
  # PROBE INSTRUMENT (throwaway). The presets draw a table offset at RANDOM
  # from +/-d, which mixes heights inside a rung and moves both tables at
  # once -- useless for asking "how far down can standing still reach?".
  # These force ONE fixed offset on every episode and override whatever the
  # preset would have drawn, so a rung is a single geometry and the two
  # tables can be walked down independently.
  ap.add_argument("--brown-dz", type=float, default=None,
                  help="force the brown (pick) tabletop offset, metres")
  ap.add_argument("--blue-dz", type=float, default=None,
                  help="force the blue (place) tabletop offset, metres")
  args = ap.parse_args()

  bounds, cyl_nominal = table_bounds(PRESETS[args.preset].get("cyl_margin_m",
                                                              0.045))
  specs = [make_episode(i, args.seed, args.preset, bounds, cyl_nominal,
                        nominal=(i == 0 and args.nominal_first))
           for i in range(args.episodes)]
  if args.brown_dz is not None or args.blue_dz is not None:
    # Applied AFTER the draw, so the spawn and cylinder streams are the ones
    # the preset would have produced and the rung stays paired with its
    # control.
    for sp in specs:
      if args.brown_dz is not None:
        sp["brown_dz"] = float(args.brown_dz)
      if args.blue_dz is not None:
        sp["blue_dz"] = float(args.blue_dz)
  opts = {"video": args.video, "sim_limit": args.sim_limit,
          "wall_limit": args.wall_limit}
  run_id = (f"{args.tag + '_' if args.tag else ''}"
            f"{args.preset}"
            f"_n{args.episodes}_seed{args.seed}_"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")

  print(f"sweep: preset={args.preset} N={args.episodes} seed={args.seed} "
        f"jobs={args.jobs} video={args.video}")
  print(f"brown tabletop interior: x [{bounds[0]:+.3f},{bounds[1]:+.3f}] "
        f"y [{bounds[2]:+.3f},{bounds[3]:+.3f}]")
  t0 = time.time()
  records, logs = [], {}

  def note(rec):
    print(f"  ep {rec['ep']:>2d}  {rec.get('phase', '?'):<12s} "
          f"{'SUCCESS' if rec.get('success') else rec['failure']:<21s} "
          f"{rec.get('wall_time_s', 0):>5.1f}s  "
          f"[{len(records)}/{len(specs)}]", flush=True)

  jobs = [(s, opts) for s in specs]
  if args.jobs > 1 and len(jobs) > 1:
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.jobs, len(jobs))) as pool:
      for rec, log in pool.imap_unordered(run_episode, jobs):
        records.append(rec)
        logs[rec["ep"]] = log
        note(rec)
  else:
    for job in jobs:
      rec, log = run_episode(job)
      records.append(rec)
      logs[rec["ep"]] = log
      note(rec)

  meta = {"preset": args.preset, "seed": args.seed, "episodes": args.episodes,
          "nominal_first": args.nominal_first,
          "jobs": args.jobs, "video": args.video,
          "sim_limit_s": args.sim_limit, "wall_limit_s": args.wall_limit,
          "ranges": PRESETS[args.preset],
          "forced_brown_dz": args.brown_dz,
          "forced_blue_dz": args.blue_dz,
          "commander": os.environ.get("G1HL_COMMANDER", ""),
          "table_bounds": [round(b, 4) for b in bounds],
          "started": datetime.now().isoformat(timespec="seconds"),
          "sweep_wall_s": round(time.time() - t0, 1),
          "git": os.popen("git -C %s rev-parse --short HEAD" % REPO).read().strip()}
  report = summarize(records, meta)
  print(report)
  jpath, cpath, ldir = write_results(run_id, records, meta, logs, report)
  print(f"  wrote {jpath.relative_to(REPO)}")
  print(f"        {cpath.relative_to(REPO)}")
  print(f"        {ldir.relative_to(REPO)}/  (per-episode stdout)")
  print(f"  sweep wall time {meta['sweep_wall_s']:.1f} s")
  n_ok = sum(1 for r in records if r.get("success"))
  return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
  sys.exit(main())
