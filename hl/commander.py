"""Direct high-level policy over the IK pipeline: the HL owns the pick.

The HL never moves a joint. Through `ik.pipeline.COMMANDER` it issues, per
episode:

  pick   (yaw, fwd, lat, crouch)   a stance relative to the cylinder,
                                   base = cyl - R(yaw) @ (fwd, lat), and a
                                   pelvis height (above `STAND_ABOVE` = stand)
  grasp  (tilt deg, curl side, azimuth deg), chosen AFTER the walk from the
         cylinder's realized body-frame offset
  move   or decline that grasp and command a new stance (`move_instead`)

The pipeline's replanner is skipped (no feasibility search, no corrective
legs, no IK-residual abort, no script default); the walk, the IK arm path,
the carry and the place are the pipeline's.

    G1HL_COMMANDER=direct:<model.pt>           argmax of P(success)
    G1HL_COMMANDER=direct_thompson:<model.pt>  one bootstrap member's argmax
                                               (on-policy data collection)
    G1HL_COMMANDER=direct_explore              uniform commands (data)
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ik import pipeline as e2  # noqa: E402

D_CMD = 4
STAND_ABOVE = 0.72
STAND_H = 0.766                      # the standing pelvis, for the features
DPICK_H = (0.55, 0.80)
GRASP_LO = np.array([15.0, -1.0, -45.0])
GRASP_HI = np.array([45.0, 1.0, 45.0])




def pick_to_world(ctx, cmd):
  yaw, fwd, lat = (float(v) for v in cmd[:3])
  return e2.offset_to_base(ctx["cyl"][:2], yaw, fwd, lat), yaw


def pick_clear(ctx, cmd):
  """The stance keeps `STANCE_CLEAR_M` off both tabletops, as the script's
  own basin search requires -- a command may not park the base in a table."""
  xy, _ = pick_to_world(ctx, cmd)
  return not e2._table_blocker(ctx["geo"])(xy)


def _wrap(a):
  return float(np.arctan2(np.sin(a), np.cos(a)))


def pick_features(ctx):
  """Scene the pick command is chosen against, in the cylinder's frame."""
  g, c = ctx["geo"], ctx["cyl"]
  rel_b = np.asarray(ctx["base_xy"]) - c[:2]
  return np.array([
    c[0] - g["brown_center"][0], c[1] - g["brown_center"][1],
    g["brown_half"][0], g["brown_half"][1], g["brown_top_z"],
    rel_b[0], rel_b[1], _wrap(ctx["base_yaw"])], np.float32)


def pick_cmd_rel(ctx, xy, yaw, h_floor):
  """World stance -> (yaw, fwd, lat, h_floor): inverse of `pick_to_world`."""
  c, s = np.cos(yaw), np.sin(yaw)
  v = np.asarray(ctx["cyl"], float)[:2] - np.asarray(xy, float)
  return np.array([_wrap(yaw), c * v[0] + s * v[1], -s * v[0] + c * v[1],
                   h_floor], np.float32)


def bank_key(ctx):
  """Where the cylinder sits on what table: the scene a stance is reused in."""
  g, c = ctx["geo"], np.asarray(ctx["cyl"], float)
  return np.array([c[0] - g["brown_center"][0], c[1] - g["brown_center"][1],
                   g["brown_half"][0], g["brown_half"][1], g["brown_top_z"]],
                  np.float32)


# Place candidates: the script's own command, moved on a small grid. The place
# stance barely moved success across hl_full's +/-15 cm table shift, so the
# search stays local to it rather than roaming the whole box.


def facing_edge_yaw(ctx):
  """Heading that faces the brown table across the edge nearest the cylinder."""
  g, c = ctx["geo"], np.asarray(ctx["cyl"], float)
  r = c[:2] - np.asarray(g["brown_center"][:2], float)
  h = np.asarray(g["brown_half"][:2], float)
  gaps = [h[0] - r[0], h[0] + r[0], h[1] - r[1], h[1] + r[1]]   # +x -x +y -y
  return (np.pi, 0.0, -np.pi / 2, np.pi / 2)[int(np.argmin(gaps))]


def sample_dpick(rng, ctx, n):
  """Stances facing the nearest edge +/-60 deg, inside the stance grid."""
  out = []
  y0 = facing_edge_yaw(ctx)
  for _ in range(40):
    m = max(n, 64)
    c = np.stack([y0 + rng.uniform(-np.radians(60), np.radians(60), m),
                  rng.uniform(0.22, 0.45, m), rng.uniform(-0.33, 0.03, m),
                  rng.uniform(*DPICK_H, m)], 1).astype(np.float32)
    c[:, 0] = np.angle(np.exp(1j * c[:, 0]))
    out += [x for x in c if pick_clear(ctx, x)]
    if len(out) >= n:
      break
  return np.array(out[:n], np.float32).reshape(-1, 4)


def grasp_features(ctx):
  """The grasp as the walk left it: cylinder in the body frame, heights
  against the pelvis, and the table edge the path must clear."""
  g, c = ctx["geo"], np.asarray(ctx["cyl"], float)
  yaw = float(ctx["base_yaw"])
  v = c[:2] - np.asarray(ctx["base_xy"], float)
  pz = float(ctx.get("pelvis_z", STAND_H))
  return np.array([
    np.cos(yaw) * v[0] + np.sin(yaw) * v[1],
    -np.sin(yaw) * v[0] + np.cos(yaw) * v[1],
    c[2] - pz, g["brown_top_z"] - pz,
    c[0] - g["brown_center"][0], c[1] - g["brown_center"][1],
    g["brown_half"][0], g["brown_half"][1]], np.float32)


GRASP_GRID = np.array([(t, s, a) for t in (15, 20, 25, 30, 35, 40, 45)
                       for s in (-1.0, 1.0)
                       for a in range(-45, 46, 15)], np.float32)




def _set_crouch(cmdr, h):
  cmdr.pick_crouch_h = None if float(h) > STAND_ABOVE else float(h)


class DirectExploreCommander:
  """Uniform direct commands: data for the direct models."""
  direct = True

  def reset(self, spec):
    key = f"direct/{spec['seed']}/{spec['ep']}/{spec.get('preset')}".encode()
    self.rng = np.random.default_rng(
      int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
    self.pick_crouch_h = None

  def pick_stance(self, ctx):
    c = sample_dpick(self.rng, ctx, 1)
    c = c[0] if len(c) else np.array([facing_edge_yaw(ctx), 0.35, -0.2, 0.8])
    _set_crouch(self, c[3])
    return pick_to_world(ctx, c)

  def grasp(self, ctx):
    t, s, a = self.rng.uniform(GRASP_LO, GRASP_HI)
    return float(t), (1.0 if s >= 0 else -1.0), float(a)

  def place_stance(self, ctx):
    return np.array(e2.PLACE_STANCE[0], float), float(e2.PLACE_STANCE[1])


class DirectLearnedCommander:
  """argmax of the learned P(success) at each decision, no default, no margin.

  pick   candidates: `N_CAND` stances from the exploration distribution plus
         the hindsight bank's stances in similar scenes, each at every crouch
  grasp  candidates: `GRASP_GRID`
  `thompson=True` flies one bootstrap member's argmax instead (collection).
  """
  direct = True
  N_CAND = 256
  K_NN = 24
  CROUCHES = (0.80, 0.70, 0.65, 0.60, 0.55)

  def __init__(self, path, thompson=False):
    from hl.train import load_models, load_bank
    self.models = load_models(path)
    try:
      self.bank_q, self.bank_s, self.bank_mu, self.bank_sd = load_bank(path)
    except KeyError:
      self.bank_q = None
    self.thompson = thompson
    self.member = None
    self.rng = np.random.default_rng(0)

  def reset(self, spec):
    self.rng = np.random.default_rng([int(spec["seed"]), int(spec["ep"]), 7])
    self.pick_crouch_h = None
    if self.thompson:
      self.member = int(self.rng.integers(len(self.models["dpick"].nets)))

  def _argmax(self, stage, feat, cands):
    m = self.models[stage]
    if self.member is not None:
      p = m.probs(feat, cands)[self.member % len(m.nets)]
    else:
      p = m.score(feat, cands)
    return cands[int(np.argmax(p))]

  def pick_stance(self, ctx):
    cands = [sample_dpick(self.rng, ctx, self.N_CAND)]
    if self.bank_q is not None:
      q = (bank_key(ctx) - self.bank_mu) / self.bank_sd
      d = np.linalg.norm((self.bank_q - self.bank_mu) / self.bank_sd - q, 1)
      for yaw, fwd, lat in self.bank_s[np.argsort(d)[:self.K_NN]]:
        for h in self.CROUCHES:
          c = np.array([yaw, fwd, lat, h], np.float32)
          if pick_clear(ctx, c):
            cands.append(c[None])
    cands = np.vstack(cands)
    c = self._argmax("dpick", pick_features(ctx), cands)
    _set_crouch(self, c[3])
    return pick_to_world(ctx, c)

  def grasp(self, ctx):
    t, s, a = self._argmax("grasp", grasp_features(ctx), GRASP_GRID)
    return float(t), float(s), float(a)

  # The HL's own correction. If even its best grasp from where the walk landed
  # is predicted below `MOVE_TAU`, it commands a new stance instead (up to
  # `MAX_MOVES`). Calibrated on 197 held-out on-policy episodes: best p < 0.20
  # succeeded 2 times in 40 (5%), against ~30% for a fresh attempt.
  MOVE_TAU = float(os.environ.get("G1HL_DIRECT_TAU", "0.20"))
  MAX_MOVES = int(os.environ.get("G1HL_DIRECT_MOVES", "2"))

  def move_instead(self, ctx, k):
    """None = grasp from here; otherwise the best grasp's predicted P."""
    if k >= self.MAX_MOVES:
      return None
    p = float(self.models["grasp"].probs(grasp_features(ctx), GRASP_GRID)
              .mean(0).max())
    return None if p >= self.MOVE_TAU else p

  def place_stance(self, ctx):
    return np.array(e2.PLACE_STANCE[0], float), float(e2.PLACE_STANCE[1])


def install(name=None):
  """Set `e2.COMMANDER` from `G1HL_COMMANDER` (or `name`); returns it."""
  name = name or os.environ.get("G1HL_COMMANDER", "")
  if not name:
    return None
  if getattr(e2.COMMANDER, "_name", None) == name:
    return e2.COMMANDER          # one worker flies many episodes; load once
  if name == "direct_explore":
    cmd = DirectExploreCommander()
  elif name.startswith("direct_thompson:"):
    cmd = DirectLearnedCommander(name.split(":", 1)[1], thompson=True)
  elif name.startswith("direct:"):
    cmd = DirectLearnedCommander(name.split(":", 1)[1])
  else:
    raise ValueError(f"unknown commander {name!r}")
  cmd._name = name
  e2.COMMANDER = cmd
  return cmd
