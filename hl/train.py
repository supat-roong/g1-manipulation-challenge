"""Fit the direct HL: P(success | scene, command) for the pick and the grasp.

Reads sweep JSONs and rebuilds every decision the direct commander made from
the `ctx_pick` / `ctx_grasp` records `ik.pipeline` logs at the seam. The
features go through `commander.pick_features` / `grasp_features`, the SAME
functions the deployed commander calls, so train and deploy cannot read
different observations.

  dpick  (scene, stance + crouch)            -> spec success
  grasp  (walked-to offset, tilt/side/az)    -> spec success
  bank   stances the pipeline lifted from, reused as pick candidates

Both stages are labelled with SUCCESS, not with the lift: the grasp's tilt and
azimuth set the in-grip pose the carry and the place inherit. Lift-trained
commands lifted 44% of 250 on-policy episodes but converted only 45% of those
lifts, against the script's 61% with the same place.

    python -m hl.train eval/results/dx_*.json eval/results/dts_*.json \
        --out hl/models/direct.pt
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hl import commander as C  # noqa: E402

LIFTED = C.e2.PHASES.index("lifted")


def _ctx(rec):
  geo = {k: np.asarray(rec[k], float) for k in
         ("brown_center", "brown_half", "blue_center", "blue_half")}
  geo["brown_top_z"] = float(rec["brown_top_z"])
  geo["blue_top_z"] = float(rec["blue_top_z"])
  return {"geo": geo, "cyl": np.asarray(rec["cyl"], float),
          "base_xy": np.asarray(rec["base_xy"], float),
          "base_yaw": float(rec["base_yaw"])}


pick_cmd_rel = C.pick_cmd_rel


def enc_cmd(stage, cmds):
  """Headings go in as (cos, sin): -pi and pi are the same stance."""
  if stage == "grasp":          # (tilt deg, curl side, azimuth deg)
    cmds = np.asarray(cmds, np.float32).reshape(-1, 3)
    a = np.radians(cmds[:, 2])
    return np.stack([cmds[:, 0] / 45.0, cmds[:, 1], np.cos(a), np.sin(a)], 1)
  cmds = np.asarray(cmds, np.float32).reshape(-1, C.D_CMD)
  if stage == "dpick":          # every crouch above STAND_ABOVE is "stand"
    h = np.where(cmds[:, 3] > C.STAND_ABOVE, C.STAND_H, cmds[:, 3])
    y = cmds[:, 0]
    return np.stack([np.cos(y), np.sin(y), cmds[:, 1], cmds[:, 2], h], 1)
  raise ValueError(f"unknown stage {stage!r}")


def extract(paths):
  """-> {"dpick": (feat, cmd, y), "grasp": (...)} over every direct episode."""
  rows = {"dpick": [], "grasp": []}
  for p in paths:
    for rec in json.load(open(p))["episodes"]:
      if rec.get("abort") == "crash":
        continue
      ok = float(bool(rec.get("success")))
      if rec.get("cmd_grasp") is not None and rec.get("ctx_pick"):
        ctx = _ctx(rec["ctx_pick"])
        x, y, yd = rec["cmd_pick"]
        ch = rec.get("cmd_pick_crouch_h")
        rows["dpick"].append((C.pick_features(ctx), pick_cmd_rel(
          ctx, (x, y), np.radians(yd), C.STAND_H if ch is None else ch),
          ok))
        gctx = _ctx(rec["ctx_grasp"])
        gctx["pelvis_z"] = float(rec["ctx_grasp"]["pelvis_z"])
        rows["grasp"].append((C.grasp_features(gctx),
                              np.asarray(rec["cmd_grasp"], np.float32),
                              ok))
  out = {}
  for k, r in rows.items():
    if r:
      out[k] = tuple(np.array([row[i] for row in r], np.float32)
                     for i in range(3))
  return out


def extract_bank(paths):
  """Hindsight stances: where the replanner REACHED a feasible grasp, in the
  pick command frame (yaw, fwd, lat), keyed by the scene it was reached in.
  Only episodes that went on to lift -- a feasible plan that then failed to
  capture is not a stance worth reusing."""
  q, st = [], []
  for p in paths:
    for rec in json.load(open(p))["episodes"]:
      if not (rec.get("ctx_pick") and rec.get("cyl_in_body_grasp")
              and rec.get("pick_heading_deg") is not None
              and rec.get("phase_idx", 0) >= LIFTED):
        continue
      fwd, lat = rec["cyl_in_body_grasp"]
      q.append(C.bank_key(_ctx(rec["ctx_pick"])))
      st.append([np.radians(rec["pick_heading_deg"]), fwd, lat])
  return np.array(q, np.float32), np.array(st, np.float32)


def load_bank(path):
  b = torch.load(path, weights_only=False)["_bank"]
  return b["q"], b["s"], b["mu"], b["sd"]


def _net(d_in, hidden=64):
  return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                       nn.Linear(hidden, hidden), nn.SiLU(),
                       nn.Linear(hidden, 1))


class StageModel:
  """Ensemble of MLP classifiers; `score` is mean - `k_std` * std of P."""

  def __init__(self, stage, mu, sd, nets, k_std=1.0):
    self.stage, self.mu, self.sd, self.nets = stage, mu, sd, nets
    self.k_std = k_std

  def _x(self, feat, cmds):
    feat = np.asarray(feat, np.float32).reshape(-1, feat.shape[-1])
    c = enc_cmd(self.stage, cmds)
    if len(feat) == 1:
      feat = np.repeat(feat, len(c), 0)
    x = np.concatenate([feat, c], 1)
    return torch.as_tensor((x - self.mu) / self.sd)

  @torch.no_grad()
  def probs(self, feat, cmds):
    x = self._x(feat, cmds)
    return np.stack([torch.sigmoid(n(x)).squeeze(1).numpy()
                     for n in self.nets])

  def score(self, feat, cmds):
    p = self.probs(feat, cmds)
    return p.mean(0) - self.k_std * p.std(0)


# Held-out log-loss on 450 pick episodes (5-fold): 400 epochs at wd 1e-3 scored
# 1.44 against a base rate of 0.649 -- confidently wrong, and it saturated every
# score at 1.0. 50 epochs at wd 1.0: 0.605, AUC 0.714.
def fit_stage(stage, feat, cmd, y, n_ens=5, epochs=50, seed=0, wd=1.0):
  x = np.concatenate([feat, enc_cmd(stage, cmd)], 1)
  mu, sd = x.mean(0), x.std(0) + 1e-6
  xt = torch.as_tensor((x - mu) / sd)
  yt = torch.as_tensor(y)[:, None]
  nets = []
  for m in range(n_ens):
    g = torch.Generator().manual_seed(seed * 100 + m)
    idx = torch.randint(0, len(xt), (len(xt),), generator=g)   # bootstrap
    torch.manual_seed(seed * 100 + m)
    net = _net(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    for _ in range(epochs):
      opt.zero_grad()
      loss = nn.functional.binary_cross_entropy_with_logits(net(xt[idx]),
                                                            yt[idx])
      loss.backward()
      opt.step()
    nets.append(net.eval())
  return StageModel(stage, mu.astype(np.float32), sd.astype(np.float32), nets)


def cv_auc(stage, feat, cmd, y, folds=5, seed=0):
  """Held-out AUC: does the model rank outcomes at all, before we act on it?"""
  rng = np.random.default_rng(seed)
  perm = rng.permutation(len(y))
  pred = np.zeros(len(y))
  for f in range(folds):
    te = perm[f::folds]
    tr = np.setdiff1d(perm, te)
    m = fit_stage(stage, feat[tr], cmd[tr], y[tr], n_ens=3, seed=seed + f)
    pred[te] = m.probs(feat[te], cmd[te]).mean(0)
  pos, neg = pred[y > 0.5], pred[y < 0.5]
  if not len(pos) or not len(neg):
    return float("nan")
  return float((pos[:, None] > neg[None]).mean()
               + 0.5 * (pos[:, None] == neg[None]).mean())


def save_models(models, path, bank=None):
  blob = {k: {"mu": m.mu, "sd": m.sd, "k_std": m.k_std,
              "d_in": int(m.mu.shape[0]),
              "nets": [n.state_dict() for n in m.nets]}
          for k, m in models.items()}
  if bank is not None:
    q, st = bank
    blob["_bank"] = {"q": q, "s": st, "mu": q.mean(0),
                     "sd": q.std(0) + 1e-6}
  torch.save(blob, path)


def load_models(path):
  blob = torch.load(path, weights_only=False)
  out = {}
  for k, b in blob.items():
    if k.startswith("_"):
      continue
    nets = []
    for sd in b["nets"]:
      n = _net(b["d_in"])
      n.load_state_dict(sd)
      nets.append(n.eval())
    out[k] = StageModel(k, b["mu"], b["sd"], nets, b["k_std"])
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("inputs", nargs="+", help="sweep JSONs (globs ok)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--k-std", type=float, default=1.0)
  args = ap.parse_args()
  paths = sorted({p for g in args.inputs for p in glob.glob(g)})
  # seed 3 is the held-out evaluation seed: never let it into a fit
  leak = [p for p in paths if "_seed3_" in Path(p).name]
  if leak:
    sys.exit(f"refusing to train on the evaluation seed: {leak}")
  data = extract(paths)
  models = {}
  for stage, (feat, cmd, y) in data.items():
    auc = cv_auc(stage, feat, cmd, y) if len(y) >= 20 else float("nan")
    print(f"{stage:5s}  n={len(y):4d}  base rate {y.mean():.3f}  "
          f"held-out AUC {auc:.3f}")
    models[stage] = fit_stage(stage, feat, cmd, y)
    models[stage].k_std = args.k_std
  bank = extract_bank(paths)
  print(f"bank   n={len(bank[0]):4d}  hindsight stances that lifted")
  save_models(models, args.out, bank)
  print(f"wrote {args.out} from {len(paths)} file(s)")


if __name__ == "__main__":
  main()
