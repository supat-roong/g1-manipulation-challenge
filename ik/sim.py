"""Simulation layer for the IK pipeline: scene, runner, arm IK and the walk.

  - walk: P-controller over the walker's velocity commands with a 0.30 m/s
    floor (the gait deadbands smaller commands: it steps in place), break on
    arrival; yaw is used for aiming (accurate), strafe is not.
  - arm: discrete kinematic 6-DoF DLS IK (mj_jacSite; mj_comPos is REQUIRED on
    scratch data or the Jacobian is silently zero) + aim-and-correct rounds,
    with gravity sag cancelled in closed form by adding qfrc_bias/kp to the
    arm's POSITION SETPOINT. No qfrc_applied wrench: every term is computable
    on a real robot (qfrc_bias from inverse dynamics on the encoders, kp from
    the controller config).
  - `Runner` steps the physics at 200 Hz, the policies at 50 Hz, and records
    the telemetry, video and snapshots the pipeline reports.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent  # repo root, for `run.py`
sys.path.insert(0, str(PROJECT))

from run import SCRIPT_DIR, G1Controller, ONNXPolicy, set_armature  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "frames"
OUT_DIR.mkdir(exist_ok=True)
DECIMATION = 4

PLACE_XY = np.array([-0.15, -0.62])  # blue table, ~7 cm inside the NE corner
BLUE_TOP_Z = 0.633
CYL_HALF = 0.037

# ------------------- M1.6: feedforward Cartesian palm path ------------------ #
# `ramp_to` used to clip PER-JOINT deltas toward a single endpoint solution, so
# every joint marched independently at the same rate and joints with a small
# delta finished early: the palm traced an uncontrolled arc between two poses
# that both solved perfectly. Measured worst case, with 0.0 mm IK residual at
# BOTH endpoints: the palm was carried 10 cm ABOVE the cylinder and came down on
# top of it, sweeping it 78 cm off the table.
#
# The fix is feedforward, in two layers, and deliberately NOT a task-space
# servo (a continuous Cartesian loop oscillates up to 20 cm against the soft
# actuator PD; discrete solve -> ramp -> measure -> re-aim is what is stable):
#
#   SYNC_RAMP:   inside one ramp, scale each joint's rate by its own delta so
#                every joint arrives on the SAME control step. Removes the
#                "short-travel joints finish early" arc within a segment.
#   CART_PATH:   interpolate the PALM POSE (position linearly, orientation by
#                rotation-vector interpolation) into waypoints, solve IK for
#                each one OFFLINE on scratch MjData seeded from the previous
#                solution, then ramp between consecutive joint solutions.
#                Consecutive solutions are close, so the residual arcing
#                between them is negligible and the palm follows the line.
#
# Both are module switches so the before/after ablation is one assignment.
CART_PATH = True
SYNC_RAMP = True
# Move tags kept ENTIRELY on the legacy joint-space shape: no waypoints and no
# rate synchronisation. EMPTY since M1.7: the whole episode now flies planned
# Cartesian paths, so the foul model applies everywhere.
#
# M1.6 had to keep the grasp approach here, because the tuned grasp needed the
# arc that the per-joint ramp bowed 4-14 cm ABOVE the straight line: captured
# 7/25 with the arc, 2/25 flown straight, since a straight 25 deg diagonal
# ploughs the fingers 9-10 cm along the table. M1.7 measured WHY (the pre-curled
# finger pair sits inside the cylinder's radius, so the cage cannot admit it
# from the front, only from above) and replaced the accident with an explicit
# standoff-apex-descend path whose apex height is derived from the hand
# geometry. Set this back to ("pick_raise", "pick_approach") to recover the
# M1.6 shape for an ablation.
CART_SKIP_TAGS = ()
CART_STEP_M = 0.035     # one waypoint per ~3.5 cm of palm travel
CART_STEP_DEG = 20.0    # ... or per 20 deg of palm rotation
CART_MAX_WP = 10        # waypoints are not free: each is an IK solve + a ramp
CART_SEED_ITERS = 120   # DLS iterations for a waypoint seeded from its
                        # predecessor (the endpoint still gets the full 300)
# A straight palm line is NOT always available. When an interior waypoint has no
# IK solution, or when consecutive solutions sit on different IK branches, the
# straight line is not a path the arm can fly and forcing it is worse than the
# arc: the plan is rejected and the move falls back to the single endpoint solve
# (with the synchronized ramp). Both thresholds are measured, see the M1.6 notes.
CART_WP_TOL = 0.015    # max interior-waypoint IK residual, m
CART_JUMP_RAD = 0.45   # max joint step between consecutive solutions
# --- M1.11: how the ramp STOPS -------------------------------------------- #
# `ramp_to` marches `q_cmd` at a constant `rate` and then stops in one control
# step, so the commanded joint velocity steps from `rate` to 0. The arm is
# behind by kv*v/kp at that moment and springs forward: measured on the lift at
# mu 4.0, the palm overshoots its own command by +3.4 mm, recoils to -2.4 mm and
# rings at ~1 Hz for longer than the 0.8 s settle. That ring, not a sag, is the
# "arm sags 0.5 cm" M1.10 recorded, and it is what throws a high-friction
# cylinder out of the cage.
#
# `RAMP_TAPER` decelerates the last `RAMP_TAPER` of the ramp's peak-joint delta
# at CONSTANT deceleration (v = rate*sqrt(rem/d)), so the command arrives with
# `RAMP_TAPER_FLOOR` of the rate instead of all of it. `RAMP_TAPER_IN` does the
# same to the ramp's START, where the command velocity otherwise steps 0 ->
# rate. Cost is bounded: ~0.9*taper of the move's own duration per tapered end.
RAMP_TAPER = 0.0
RAMP_TAPER_IN = 0.0
RAMP_TAPER_FLOOR = 0.1  # never below this fraction of `rate` -- the ramp has to
                        # terminate, and the final snap is at most rate*floor
# --- M1.11: the grip is a VICE on a feather -------------------------------- #
# The cylinder weighs 8.9 g (0.087 N). At full closure the finger position
# actuators sit 130 mrad (middle_0) and 62 mrad (middle_1) short of their
# commanded angle, i.e. kp*dq = 0.20 N.m of blocked torque, a pinch normal of
# 0.2-0.5 N of summed contact normal, 2-6x the object's own weight. At the
# randomized cylinder<->finger friction of 4.0 the tangential CAPACITY of that
# pinch is ~1-2 N -- **10-20x the weight of the thing it is holding** -- so any
# solver residual or asymmetry is an enormous acceleration on a near-massless
# body: measured, a perfectly seated grasp
# (misalignment 0.1 deg, 14 finger contacts, 15 cm in the air) grows its net
# contact torque 30x in one second of a STATIONARY hold and pivots out.
#
# `GRIP_CAP` bounds the command to `cap` radians beyond each finger's own
# MEASURED angle, re-evaluated every step. That is a torque limit -- the drive
# torque is kp*cap regardless of geometry -- and it is per finger, so it
# expresses a uniform preload rather than a uniform angle, which is what
# `set_grip`'s single alpha cannot do. It only ever CAPS the closing command:
# a smaller alpha (the compliant seat) is already inside the cap and passes
# through untouched.
GRIP_CAP = None
# Which finger actuators the cap applies to, by index into
# `ctrl.right_finger_actuators` (0-2 thumb_0/1/2, 3-4 index_0/1, 5-6
# middle_0/1), or None for all of them. The over-drive is not spread evenly:
# at full closure on the cylinder the blocked position error is +130 mrad on
# middle_0 and +62 on middle_1 against +7 / 0 on the index pair, so capping the
# middle pair alone is a way to bound the squeeze without touching the fingers
# that are already at their contact equilibrium.
GRIP_CAP_ONLY = None


def rot_from_vec(w):
  """Rodrigues: rotation matrix from a rotation vector."""
  th = float(np.linalg.norm(w))
  if th < 1e-12:
    return np.eye(3)
  n = np.asarray(w, float) / th
  K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
  return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def cartesian_waypoints(p0, R0, p1, R1, n=None, step_m=None, step_deg=None,
                        max_wp=None):
  """Palm poses along the straight line p0 -> p1 with rotation-vector
  (slerp-equivalent) orientation interpolation. Excludes the start, includes
  the exact endpoint. `n=None` picks the count from the travel distance."""
  p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
  w = rotvec_between(R0, R1)
  ang = float(np.linalg.norm(w))
  if n is None:
    n = max(int(np.ceil(np.linalg.norm(p1 - p0)
                        / (step_m or CART_STEP_M))),
            int(np.ceil(np.degrees(ang) / (step_deg or CART_STEP_DEG))), 1)
    n = min(n, max_wp or CART_MAX_WP)
  out = []
  for k in range(1, n + 1):
    f = k / n
    if k == n:
      out.append((p1.copy(), R1.copy()))
    else:
      out.append((p0 + f * (p1 - p0), rot_from_vec(f * w) @ R0))
  return out


def seg_dev_z(trace, a, b):
  """Signed vertical deviation of an executed path from the segment a -> b:
  (max above the line, max below). Tells you which WAY a path bows, which is
  what decides whether an approach descends onto an object or ploughs into it.
  """
  if not len(trace):
    return 0.0, 0.0
  P = np.asarray(trace, float)
  a = np.asarray(a, float); b = np.asarray(b, float)
  d = b - a
  L2 = float(d @ d)
  t = np.zeros(len(P)) if L2 < 1e-12 else np.clip(((P - a) @ d) / L2, 0.0, 1.0)
  dz = P[:, 2] - (a[2] + t * d[2])
  return float(np.max(dz)), float(np.min(dz))


def seg_dev(trace, a, b):
  """Max distance from an executed path to the straight SEGMENT a -> b.

  The direct measure of whether the palm flew the line it was aimed along.
  Distance to the segment (not the infinite line), so an overshoot past the
  endpoint registers as the deviation it is.
  """
  if not len(trace):
    return 0.0
  P = np.asarray(trace, float)
  a = np.asarray(a, float); b = np.asarray(b, float)
  d = b - a
  L2 = float(d @ d)
  if L2 < 1e-12:
    return float(np.max(np.linalg.norm(P - a, axis=1)))
  t = np.clip(((P - a) @ d) / L2, 0.0, 1.0)
  return float(np.max(np.linalg.norm(P - (a + t[:, None] * d), axis=1)))


# --------------------------------------------------------------------- #
# sim setup
# --------------------------------------------------------------------- #
def build_sim(spawn=(-1.2, 0.15)):
  with open(SCRIPT_DIR / "model_config.json") as f:
    config = json.load(f)
  joint_names = config["joint_names"]
  model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / "scene.xml"))
  model.opt.timestep = 0.005
  set_armature(model, joint_names)
  data = mujoco.MjData(model)
  data.qpos[0], data.qpos[1], data.qpos[2] = spawn[0], spawn[1], 0.76
  data.qpos[3:7] = [1, 0, 0, 0]
  for name, value in config["default_joint_pos"].items():
    if name in joint_names:
      data.qpos[7 + joint_names.index(name)] = value
  mujoco.mj_forward(model, data)
  ctrl = G1Controller(
    model, data,
    ONNXPolicy(str(SCRIPT_DIR / "walker.onnx")),
    ONNXPolicy(str(SCRIPT_DIR / "croucher.onnx")),
    ONNXPolicy(str(SCRIPT_DIR / "rotator.onnx")),
    config,
    right_reacher=ONNXPolicy(str(SCRIPT_DIR / "right_reacher.onnx")),
  )
  return model, data, ctrl


def rotvec_between(R_cur, R_tgt):
  R_err = R_tgt @ R_cur.T
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, R_err.flatten())
  w = np.clip(q[0], -1, 1)
  v = q[1:4]
  s = np.linalg.norm(v)
  if s < 1e-9:
    return np.zeros(3)
  angle = 2 * np.arctan2(s, w)
  if angle > np.pi:
    angle -= 2 * np.pi
  return (v / s) * angle


def R_from_axes(x, y):
  x = np.array(x, float); x /= np.linalg.norm(x)
  y = np.array(y, float); y -= x * (x @ y); y /= np.linalg.norm(y)
  return np.column_stack([x, y, np.cross(x, y)])


def pelvis_to_world(data, p):
  """Map a point from the pelvis frame into world coordinates."""
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, data.qpos[3:7])
  return data.qpos[:3] + R.reshape(3, 3) @ np.asarray(p, float)


def base_yaw(data):
  w, x, y, z = data.qpos[3:7]
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# --------------------------------------------------------------------- #
# 6-DoF kinematic IK for the right arm
# --------------------------------------------------------------------- #
class ArmIK6:
  def __init__(self, model, ctrl):
    self.model = model
    self.site_id = ctrl.right_palm_site_id
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ctrl.right_arm_joint_names]
    self.dof_idx = [model.jnt_dofadr[j] for j in jids]
    self.qpos_idx = [model.jnt_qposadr[j] for j in jids]
    self.lo = np.array([model.jnt_range[j][0] for j in jids])
    self.hi = np.array([model.jnt_range[j][1] for j in jids])
    # Arm actuators (named after their joints) and their proportional gains
    # -- what turns a bias torque into a position-command offset.
    self.act_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in ctrl.right_arm_joint_names]
    self.kp = model.actuator_gainprm[self.act_idx, 0].copy()
    # Alternative IK seeds for `solve_ms`. DLS is a local method, so which
    # posture it starts from decides whether it finds a solution at all.
    self.q_default = np.array(
      [ctrl.default_joint_pos[i] for i in ctrl.right_arm_indices], float)
    self.alt_seeds = [self.q_default,
                      np.clip(0.5 * (self.lo + self.hi), self.lo, self.hi)]
    self.scratch = mujoco.MjData(model)
    self.jacp = np.zeros((3, model.nv))
    self.jacr = np.zeros((3, model.nv))

  def sag_correction(self, data, q_star=None):
    """Position-command offset `qfrc_bias / kp` that cancels gravity sag.

    The arm runs on `position` actuators: tau = kp (q_cmd - q) - kv qdot. At
    rest the actuator itself must supply the bias (gravity) torque, so a bare
    command q* settles at q* - qfrc_bias/kp -- roughly 36 deg of shoulder
    droop at kp = 14.25. That is the reacher's "biased low" error, and it is
    what the old `qfrc_applied = qfrc_bias` wrench papered over.

    Nothing privileged is read either way: q* comes from IK, qfrc_bias from
    inverse dynamics on the URDF at a joint configuration, kp from the
    controller config.

    Two evaluation points, same formula:

    - `q_star` given -- predictive: qfrc_bias is evaluated on scratch MjData
      at the IK solution, before the arm is there. This is the form derived in
      the design spec (S3.2); its fixed point is only correct to first order,
      because the true settle pose is where kp (q_cmd - q) = qfrc_bias(q).
    - `q_star` omitted -- measured: qfrc_bias is read straight off `data`,
      where `mj_step` has already computed it for the current (q, qdot). It
      costs nothing, and its fixed point is EXACT: commanding
      q_cmd = q* + qfrc_bias(q)/kp leaves kp (q* - q) = 0, i.e. q = q*.
      It also cancels the velocity-dependent part of the bias, so the arm
      holds its pose while the base walks. This is what the controller uses.
    """
    if q_star is None:
      return data.qfrc_bias[self.dof_idx] / self.kp
    d = self.scratch
    d.qpos[:] = data.qpos
    d.qpos[self.qpos_idx] = q_star
    d.qvel[:] = 0.0
    mujoco.mj_forward(self.model, d)
    return d.qfrc_bias[self.dof_idx] / self.kp

  def solve_ms(self, data, target_p, target_R=None, w_ori=0.5, iters=300,
               seed=None, tol=1e-3):
    """Multi-start `solve`. DLS is local, so its residual can report a target
    as unreachable when the arm merely cannot get there FROM ITS CURRENT
    POSTURE. Measured: a 15 cm straight-up lift solved to 0.002 mm from one
    grasp posture and stalled 299 mm short from another 4 cm away, and the
    controller flew the 299 mm solution: 45 cm of palm excursion, cylinder
    thrown off the table. Retrying from the default arm pose (and from
    mid-range) makes the residual a property of the TARGET, which is what both
    the controller and the feasibility search need it to be.

    Retries only fire when the first solve misses, so a reachable target costs
    exactly what it did before.
    """
    q, rp, rr = self.solve(data, target_p, target_R, w_ori, iters, seed)
    if rp <= tol:
      return q, rp, rr
    for alt in self.alt_seeds:
      q2, rp2, rr2 = self.solve(data, target_p, target_R, w_ori, iters, alt)
      if rp2 < rp:
        q, rp, rr = q2, rp2, rr2
      if rp <= tol:
        break
    return q, rp, rr

  def solve(self, data, target_p, target_R=None, w_ori=0.5, iters=300,
            seed=None):
    """DLS IK on scratch MjData. `seed` starts the arm from a given joint
    configuration instead of the measured one; that is what keeps consecutive
    Cartesian waypoints on the same IK branch (and makes them cheap: a
    waypoint 3.5 cm from its predecessor converges in a few iterations)."""
    d = self.scratch
    d.qpos[:] = data.qpos
    if seed is not None:
      d.qpos[self.qpos_idx] = np.clip(seed, self.lo, self.hi)
    for _ in range(iters):
      mujoco.mj_kinematics(self.model, d)
      mujoco.mj_comPos(self.model, d)  # REQUIRED before mj_jacSite
      e_p = target_p - d.site_xpos[self.site_id]
      mujoco.mj_jacSite(self.model, d, self.jacp, self.jacr, self.site_id)
      if target_R is None:
        e = e_p
        J = self.jacp[:, self.dof_idx]
      else:
        R_cur = d.site_xmat[self.site_id].reshape(3, 3)
        e = np.concatenate([e_p, w_ori * rotvec_between(R_cur, target_R)])
        J = np.vstack([self.jacp[:, self.dof_idx],
                       w_ori * self.jacr[:, self.dof_idx]])
      if np.linalg.norm(e) < 1e-4:
        break
      dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(J.shape[0]), e)
      d.qpos[self.qpos_idx] = np.clip(
        d.qpos[self.qpos_idx] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
    mujoco.mj_kinematics(self.model, d)
    resid_p = float(np.linalg.norm(target_p - d.site_xpos[self.site_id]))
    resid_r = 0.0
    if target_R is not None:
      resid_r = float(np.degrees(np.linalg.norm(
        rotvec_between(d.site_xmat[self.site_id].reshape(3, 3), target_R))))
    return d.qpos[self.qpos_idx].copy(), resid_p, resid_r


# --------------------------------------------------------------------- #
# runner: physics loop + arm override + ramped grip + video
# --------------------------------------------------------------------- #
class Runner:
  def __init__(self, model, data, ctrl, ik, video_every=8, fps=25,
               video_name="e2e_run.mp4"):
    self.model, self.data, self.ctrl, self.ik = model, data, ctrl, ik
    self.state = {"cs": 0, "tp": ctrl.default_joint_pos.copy()}
    self.q_cmd = None
    self.grip_alpha = 0.0        # thumb (first 3 finger actuators)
    self.grip_alpha_f = 0.0      # index + middle (last 4)
    self.grip_cap = GRIP_CAP     # rad of preload beyond the measured angle,
                                 # or None for the raw position command
    # qpos address of each finger actuator's joint, for the cap
    self._fq = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]])
                for a, _ in ctrl.right_finger_actuators]
    self.renderer = mujoco.Renderer(model, 480, 640)
    self.cam = mujoco.MjvCamera()
    self.cyl_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     "red_block")
    self.video_every = video_every
    self.palm_trace = None       # set to [] to record the executed palm path
    self.ik_solves = 0           # waypoint IK solves spent by the controller
    # M1.6 straightness telemetry: one row per goto/goto_track/touchdown.
    # `dev0_m` is THE measurement of the milestone: how far the executed palm
    # path strayed from the straight line between the two poses it was aimed
    # between. `move_tag` labels the next move; the pipeline sets it.
    self.move_log = []
    self.move_tag = ""
    self.writer = cv2.VideoWriter(
      str(OUT_DIR / video_name), cv2.VideoWriter_fourcc(*"mp4v"),
      fps, (640, 480))
    self.recording = True

  def step_once(self):
    if self.state["cs"] % DECIMATION == 0:
      self.state["tp"] = self.ctrl.step()
      if self.q_cmd is not None:
        for k, full_idx in enumerate(self.ctrl.right_arm_indices):
          self.state["tp"][full_idx] = self.q_cmd[k]
    self.ctrl.apply_pd_control(self.state["tp"])
    for k, (act_id, closed_val) in enumerate(self.ctrl.right_finger_actuators):
      a = self.grip_alpha if k < 3 else self.grip_alpha_f  # thumb | fingers
      cmd = a * closed_val
      cap = self.grip_cap
      if cap is not None and (GRIP_CAP_ONLY is None or k in GRIP_CAP_ONLY):
        # Never drive more than `cap` rad past where the finger actually is.
        q = float(self.data.qpos[self._fq[k]])
        cmd = (min(cmd, q + cap) if closed_val >= 0.0
               else max(cmd, q - cap))
      self.data.ctrl[act_id] = cmd
    # Gravity compensation, folded into the arm's POSITION SETPOINT rather
    # than injected as an external wrench. The actuator produces
    # kp (ctrl - q) - kv qdot, so adding qfrc_bias/kp to ctrl adds exactly
    # qfrc_bias to the joint torque -- the same quantity the old
    # `qfrc_applied = qfrc_bias` cheat pushed in, but delivered through the
    # real actuator from quantities a robot has: encoders -> inverse dynamics,
    # and kp from the controller config. Applied once, here, so `q_cmd`,
    # `ramp_to`, `touchdown` and `goto_track`'s aim-and-correct all keep
    # working in true joint angles.
    self.data.qfrc_applied[self.ik.dof_idx] = 0.0
    self.data.ctrl[self.ik.act_idx] += self.ik.sag_correction(self.data)
    mujoco.mj_step(self.model, self.data)
    self.state["cs"] += 1
    if self.palm_trace is not None and self.state["cs"] % DECIMATION == 0:
      self.palm_trace.append(self.data.site_xpos[self.ik.site_id].copy())
    if self.recording and self.state["cs"] % self.video_every == 0:
      self.renderer.update_scene(self.data, camera="side_view")
      self.writer.write(cv2.cvtColor(self.renderer.render(),
                                     cv2.COLOR_RGB2BGR))

  def close_video(self):
    self.recording = False
    self.writer.release()

  def run(self, seconds):
    for _ in range(int(seconds / self.model.opt.timestep)):
      self.step_once()

  def set_grip(self, alpha_target, seconds=0.8, thumb=True, fingers=True):
    n = int(seconds / self.model.opt.timestep)
    a0_t, a0_f = self.grip_alpha, self.grip_alpha_f
    for i in range(n):
      frac = (i + 1) / n
      if thumb:
        self.grip_alpha = a0_t + (alpha_target - a0_t) * frac
      if fingers:
        self.grip_alpha_f = a0_f + (alpha_target - a0_f) * frac
      self.step_once()

  def ramp_grip_cap(self, target, seconds=0.0):
    """Ease `grip_cap` down to `target` rad of preload.

    Dropping the cap in one step relieves a squeeze that has been holding a
    130 mrad deflection, and the cylinder is launched by whatever the contact
    had stored -- at low friction there is nothing to stop it. Starting from
    the deflection the fingers are ACTUALLY at makes the cap non-binding on
    its first step, so the squeeze bleeds off instead of snapping.
    """
    if seconds <= 0.0 or target is None:
      self.grip_cap = target
      return
    dq = 0.0
    for k, (act_id, closed_val) in enumerate(self.ctrl.right_finger_actuators):
      q = float(self.data.qpos[self._fq[k]])
      cmd = (self.grip_alpha if k < 3 else self.grip_alpha_f) * closed_val
      dq = max(dq, abs(cmd - q))
    start = max(dq, target)
    n = max(int(seconds / self.model.opt.timestep), 1)
    for i in range(n):
      self.grip_cap = start + (target - start) * (i + 1) / n
      self.step_once()

  def grasp_center_offset(self):
    """Close the hand in free air; centroid of distal links in palm frame."""
    self.set_grip(1.0)
    self.run(0.5)
    palm_p = self.data.site_xpos[self.ik.site_id].copy()
    palm_R = self.data.site_xmat[self.ik.site_id].reshape(3, 3).copy()
    pts = []
    for bname in ("right_hand_index_1_link", "right_hand_middle_1_link",
                  "right_hand_thumb_2_link"):
      bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
      pts.append(palm_R.T @ (self.data.xpos[bid] - palm_p))
    self.set_grip(0.0)
    self.run(0.5)
    return np.mean(pts, axis=0)

  def ramp_to(self, q_star, settle=0.8, rate=0.012, sync=None, on_step=None):
    """Ramp the arm's joint command onto `q_star`.

    `sync` scales each joint's rate by its own delta so all of them arrive on
    the same control step. The legacy behaviour (`sync=False`) clipped every
    joint at the SAME rate, so joints with a small delta finished early and
    the palm arced away from the line between the two poses even when both
    solved to 0.0 mm. `on_step` is polled every physics step and returning
    True aborts the ramp (the contact-monitored descent uses it).
    """
    if self.q_cmd is None:
      self.q_cmd = self.data.qpos[self.ik.qpos_idx].copy()
    if sync is None:
      sync = SYNC_RAMP
    dq = np.asarray(q_star, float) - self.q_cmd
    peak = float(np.max(np.abs(dq)))
    if sync and peak > 1e-9:
      rates = np.maximum(rate * np.abs(dq) / peak, 1e-4)
    else:
      rates = np.full(len(self.q_cmd), float(rate))
    # Ease-in / ease-out distances, in the peak joint's own delta.
    d_out = RAMP_TAPER * peak
    d_in = RAMP_TAPER_IN * peak
    fl = RAMP_TAPER_FLOOR
    while np.max(np.abs(q_star - self.q_cmd)) > 1e-6:
      if self.state["cs"] % DECIMATION == 0:
        rem = q_star - self.q_cmd
        f = 1.0
        if d_out > 1e-9:      # constant deceleration onto the target
          f = min(f, max(fl, float(np.sqrt(
            min(1.0, float(np.max(np.abs(rem))) / d_out)))))
        if d_in > 1e-9:       # ... and constant acceleration off the start
          gone = peak - float(np.max(np.abs(rem)))
          f = min(f, max(fl, float(np.sqrt(min(1.0, max(gone, 0.0) / d_in)))))
        step = rates * f
        self.q_cmd = self.q_cmd + np.clip(rem, -step, step)
      self.step_once()
      if on_step is not None and on_step():
        return True
    self.run(settle)
    return False

  def palm_pose(self):
    """Measured palm pose (position, rotation matrix)."""
    return (self.data.site_xpos[self.ik.site_id].copy(),
            self.data.site_xmat[self.ik.site_id].reshape(3, 3).copy())

  def plan_cartesian(self, target_p, target_R, n=None, iters=300, ray=False):
    """Feedforward Cartesian plan: palm-pose waypoints from where the palm is
    now to (target_p, target_R), each solved OFFLINE on scratch MjData and
    seeded from its predecessor. Returns (joint solutions, endpoint residual,
    diagnostics).

    No simulation runs here and no measurement is fed back; this is not a
    task-space servo (which oscillates up to 20 cm against the soft actuator
    PD); it is a precomputed path the existing ramp plays back.

    **The plan is TRUNCATED at the last waypoint the arm can actually follow**,
    and that is as important as the interpolation itself. A straight line is
    not always available: the un-tuck target sits at the edge of the workspace,
    the place descent aims 10 cm INTO the tabletop on purpose, and a DLS solve
    seeded from an awkward posture can stall in a local minimum 300 mm from a
    target the same arm reaches cleanly from elsewhere. The old controller
    commanded those solutions anyway: measured, a 15 cm lift whose endpoint
    solved 299 mm short swung the palm 45 cm and threw the cylinder off the
    table. So a waypoint whose residual exceeds `CART_WP_TOL`, or which sits
    more than `CART_JUMP_RAD` from its predecessor (an IK branch flip), ends
    the plan: the palm flies the line as far as it can follow it and stops,
    and the aim-and-correct round then re-aims from there.

    `ray=True` is the exception, for "descend until something stops you": the
    contact-monitored place descent aims 10 cm INTO the tabletop deliberately,
    so its far waypoints have no exact solution and the best-effort DLS pose at
    each is the deepest reachable point along the line, a progressively firmer
    press, which is what seats the object. A ray is therefore never truncated
    and never rejected: measured, truncating it on a branch flip stopped the
    descent 1.5 cm above the table, and a cylinder released 1.5 cm up lands
    upright only by luck. It is sampled twice as finely instead, so the
    reconfiguration that the flip represents is spread over small steps.
    """
    p0, R0 = self.palm_pose()
    if target_R is None:
      target_R = R0
    target_p = np.asarray(target_p, float)
    # No multi-start on a ray: its far targets are unreachable BY DESIGN, so
    # retrying from other seeds only burns iterations and risks answering with
    # a wildly different posture instead of the deepest press along the line.
    solve = self.ik.solve if ray else self.ik.solve_ms
    q_free, rp_free, rr_free = solve(self.data, target_p, target_R,
                                     iters=iters)
    self.ik_solves += 1
    wps = cartesian_waypoints(p0, R0, target_p, target_R, n,
                              step_m=CART_STEP_M / 2 if ray else None)
    qs, seed = [], None
    worst = jump = 0.0
    truncated = False
    for p, R in wps[:-1]:            # interior only; the endpoint is separate
      q, rq, _ = (self.ik.solve(self.data, p, R, iters=CART_SEED_ITERS,
                                seed=seed) if ray else
                  self.ik.solve_ms(self.data, p, R, iters=CART_SEED_ITERS,
                                   seed=seed, tol=CART_WP_TOL))
      self.ik_solves += 1
      step = 0.0 if seed is None else float(np.max(np.abs(q - seed)))
      if not ray and (step > CART_JUMP_RAD or rq > CART_WP_TOL):
        truncated = True
        break
      worst, jump, seed = max(worst, rq), max(jump, step), q
      qs.append(q)
    rp, rr = rp_free, rr_free
    if not truncated:
      # **The destination is sacred.** The waypoints shape the route; they must
      # never cost endpoint accuracy, because the grasp window is +/-0.8 cm and
      # stopping one waypoint (~3.5 cm) short closes the fingers on the rim.
      # Measured: dropping the endpoint whenever its solution sat on a
      # different IK branch from the end of the chain cost 1.5-2 cm of palm
      # error and took capture from 7/25 to 0/25. So the endpoint is always
      # commanded when it is reachable (the better of the chain-seeded
      # solution and the unseeded one the joint-space controller would have
      # flown) and a branch change over that last segment is accepted.
      q_end, rp_end, rr_end = solve(self.data, target_p, target_R,
                                    iters=iters, seed=seed)
      self.ik_solves += 1
      step = 0.0 if seed is None else float(np.max(np.abs(q_end - seed)))
      if ray or rp_end <= max(rp_free + 0.003, CART_WP_TOL):
        qs.append(q_end)
        jump, rp, rr = max(jump, step), rp_end, rr_end
      elif rp_free <= CART_WP_TOL:
        qs.append(q_free)            # chain-seeded endpoint was worse
        jump = max(jump, 0.0 if seed is None
                   else float(np.max(np.abs(q_free - seed))))
      else:
        truncated = True             # the destination itself is unreachable
    if not qs:
      # Not even the first step along the line is followable. Refusing to move
      # is the honest answer for an unreachable destination: the alternative is
      # the 45 cm flail this guards against.
      qs = [q_free] if rp_free <= CART_WP_TOL else [
        self.q_cmd.copy() if self.q_cmd is not None
        else self.data.qpos[self.ik.qpos_idx].copy()]
    return qs, (rp, rr), {
      "wp_resid_m": worst, "jump_rad": jump, "n_wp": len(qs), "ray": ray,
      "truncated": truncated, "free_resid_m": rp_free,
      "flyable": not truncated,
    }

  def ramp_through(self, q_list, rate=0.012, settle=0.8, on_step=None,
                   sync=None):
    """Ramp through consecutive joint solutions. Only the last one settles:
    an intermediate settle would multiply the move's wall clock by the
    waypoint count for no benefit (the path is already straight)."""
    for i, q in enumerate(q_list):
      last = i == len(q_list) - 1
      if self.ramp_to(q, settle=(settle if last else 0.0), rate=rate,
                      on_step=on_step, sync=sync):
        return True
    return False

  def goto(self, target_p, target_R, rounds=2, max_shift=0.06, verbose="",
           rate=0.012, clamp_z_down=True, cart=None, on_step=None,
           settle=0.8):
    return self.goto_track(lambda: target_p, target_R, rounds, max_shift,
                           verbose, rate, clamp_z_down, cart, on_step,
                           settle=settle)

  def goto_track(self, target_fn, target_R, rounds=2, max_shift=0.06,
                 verbose="", rate=0.012, clamp_z_down=True, cart=None,
                 on_step=None, settle=0.8):
    """Aim-and-correct rounds, each flown as a straight Cartesian palm path.

    The rounds are unchanged and load-bearing: the one-shot 4.85 cm error is
    the PELVIS drifting ~2 cm as the arm's mass extends, after the IK was
    solved against the old base pose, and only re-measuring fixes that. What
    M1.6 changes is the shape of the move inside each round.
    """
    legacy_shape = self.move_tag in CART_SKIP_TAGS
    if cart is None:
      cart = CART_PATH and not legacy_shape
    sync = False if legacy_shape else None
    shift = np.zeros(3)
    p_start, _ = self.palm_pose()
    prev_trace, self.palm_trace = self.palm_trace, [p_start.copy()]
    n_wp, aim0, dev0, t0, s0 = 0, None, 0.0, time.time(), self.ik_solves
    wp_resid = wp_jump = 0.0
    n_fallback = 0
    for r in range(rounds):
      target_p = np.asarray(target_fn(), float)
      if cart:
        qs, (rp, rr), diag = self.plan_cartesian(target_p + shift, target_R)
        wp_resid = max(wp_resid, diag["wp_resid_m"])
        wp_jump = max(wp_jump, diag["jump_rad"])
        if diag["truncated"]:
          n_fallback += 1            # flew the followable prefix and stopped
      else:
        q_star, rp, rr = self.ik.solve(self.data, target_p + shift, target_R)
        qs = [q_star]
      n_wp = max(n_wp, len(qs))
      # `on_step` aborting the ramp ends the MOVE, not just this round: it is a
      # contact stop ("descend until the hand reaches the tabletop"), and
      # re-aiming from there would only push the arm back into whatever stopped
      # it. Same semantics as `touchdown`.
      stopped = self.ramp_through(qs, rate=rate, sync=sync, on_step=on_step,
                                  settle=settle)
      if r == 0:  # the long move; rounds 1+ deliberately aim somewhere else
        aim0 = target_p + shift
        dev0 = seg_dev(self.palm_trace, p_start, aim0)
      target_p = np.asarray(target_fn(), float)
      e_p = target_p - self.data.site_xpos[self.ik.site_id]
      if verbose:
        print(f"    {verbose} round{r}: palm err "
              f"{np.linalg.norm(e_p) * 100:.1f} cm "
              f"vec {np.round(e_p * 100, 1).tolist()} "
              f"(ik resid {rp * 1000:.0f} mm/{rr:.0f} deg)")
      shift = shift + np.clip(e_p, -max_shift, max_shift)
      if clamp_z_down:  # pick descent: never re-aim downward into the table
        shift[2] = max(shift[2], 0.0)
      if stopped:
        break
    trace, self.palm_trace = self.palm_trace, prev_trace
    target_p = np.asarray(target_fn(), float)
    err = float(np.linalg.norm(target_p - self.data.site_xpos[self.ik.site_id]))
    self._log_move(p_start, aim0, target_p, trace, dev0, err, rp, cart, n_wp,
                   time.time() - t0, self.ik_solves - s0,
                   extra={"wp_resid_mm": round(wp_resid * 1000, 1),
                          "wp_jump_rad": round(wp_jump, 3),
                          "fallbacks": n_fallback, "rounds": rounds})
    return err

  def _log_move(self, p_start, aim0, target_p, trace, dev0, err, resid, cart,
                n_wp, wall, solves, tag=None, extra=None):
    """One straightness record. `dev0_m` is the deviation of the executed palm
    path from the straight line of the FIRST aim (the long move); `dev_all_m`
    also covers the aim-and-correct rounds, which deliberately re-aim."""
    self.move_log.append({
      "tag": tag if tag is not None else (self.move_tag or "?"),
      "cart": bool(cart), "n_wp": int(n_wp),
      "dist_m": round(float(np.linalg.norm(
        (aim0 if aim0 is not None else target_p) - p_start)), 4),
      "dev0_m": round(float(dev0), 4),
      "dev_all_m": round(seg_dev(trace, p_start, target_p), 4),
      "err_m": round(float(err), 4),
      "dev0_up_m": round(seg_dev_z(
        trace, p_start, aim0 if aim0 is not None else target_p)[0], 4),
      "dev0_dn_m": round(seg_dev_z(
        trace, p_start, aim0 if aim0 is not None else target_p)[1], 4),
      "resid_mm": round(float(resid) * 1000, 2),
      "wall_s": round(float(wall), 2), "ik_solves": int(solves),
      "n_samples": len(trace),
      **(extra or {}),
    })

  def snap(self, name, lookat=None):
    if lookat is None:
      self.renderer.update_scene(self.data, camera="side_view")
    else:
      self.cam.lookat[:] = lookat
      self.cam.distance = 1.2
      self.cam.azimuth = 160
      self.cam.elevation = -20
      self.renderer.update_scene(self.data, camera=self.cam)
    cv2.imwrite(str(OUT_DIR / f"{name}.png"),
                cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR))

  def cyl_pos(self):
    return self.data.xpos[self.cyl_bid].copy()

  def cyl_tilt_deg(self):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, self.data.xquat[self.cyl_bid])
    return float(np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1, 1))))

  def contact_pairs(self, needle="right_"):
    pairs = set()
    for i in range(self.data.ncon):
      c = self.data.contact[i]
      b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                             self.model.geom_bodyid[c.geom1]) or "?"
      b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                             self.model.geom_bodyid[c.geom2]) or "?"
      if (needle in b1 or needle in b2) and "ankle" not in b1 + b2 \
         and "world" not in (b1, b2):
        pairs.add(f"{b1}<->{b2}")
    return pairs

  def pair_contact(self, name_a, name_b):
    for i in range(self.data.ncon):
      c = self.data.contact[i]
      b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                             self.model.geom_bodyid[c.geom1]) or ""
      b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                             self.model.geom_bodyid[c.geom2]) or ""
      if {name_a, name_b} <= {b1, b2}:
        return True
    return False

  def touchdown(self, R_hold, max_drop=0.06, seat=0.5, rate=0.003,
                seat_grip=None, table="table_white"):
    """Lower the palm slowly (orientation held) until the cylinder touches
    the table, then keep pressing gently for `seat` seconds. Returns True if
    contact was made.

    `seat_grip` makes the seat COMPLIANT: from the moment of table contact the
    grip is ramped to that level over the press, so the pinch becomes something
    close to a free pivot while the palm is still pushing the cylinder down
    onto the tabletop, and the table's normal force does the righting. Leave it
    None for the rigid press (the pinch holds whatever tilt it arrived with,
    which is what M1.8 measured tipping the cylinder on release).

    `table` names the tabletop to land on. The default is the blue destination;
    the M1.14 regrip sets the cylinder back down on the BROWN pick table, and
    with the name hard-coded the press never saw contact and the release let go
    of a cylinder still tilted 24-87 deg (it then lay flat in 12 of 18 cases).
    """
    target = self.data.site_xpos[self.ik.site_id] + np.array([0, 0, -max_drop])
    seat_steps = max(int(seat / self.model.opt.timestep), 1)
    st = {"contact": False, "n": 0, "g0": 0.0, "g0f": 0.0}

    def monitor():
      if not st["contact"]:
        if self.pair_contact("red_block", table):
          st["contact"] = True
          st["g0"], st["g0f"] = self.grip_alpha, self.grip_alpha_f
        elif any(table in p for p in self.contact_pairs("right_hand")):
          print("    [touchdown] stopping: hand hit the table first")
          return True  # jammed; pressing harder only fights the tabletop
      else:
        st["n"] += 1
        if seat_grip is not None:
          f = min(1.0, st["n"] / seat_steps)
          self.grip_alpha = st["g0"] + (seat_grip - st["g0"]) * f
          self.grip_alpha_f = st["g0f"] + (seat_grip - st["g0f"]) * f
        if st["n"] >= seat_steps:
          return True
      return False

    # The descent is a straight vertical line with the orientation HELD (the
    # 3-finger pinch does not constrain the cylinder's rotation, so re-aiming
    # while holding spins it 2 -> 48 -> 81 deg). That makes it the single move
    # Cartesian interpolation matters most for: the joint-space version bowed
    # the palm sideways on the way down.
    p_start, _ = self.palm_pose()
    prev_trace, self.palm_trace = self.palm_trace, [p_start.copy()]
    t0, s0, resid = time.time(), self.ik_solves, 0.0
    legacy_shape = self.move_tag in CART_SKIP_TAGS
    if CART_PATH and not legacy_shape:
      qs, (resid, _), diag = self.plan_cartesian(target, R_hold, ray=True)
    else:
      q_star, resid, _ = self.ik.solve(self.data, target, R_hold)
      qs, diag = [q_star], {"wp_resid_m": 0.0, "jump_rad": 0.0,
                            "truncated": False}
    n_wp = len(qs)
    self.ramp_through(qs, rate=rate, settle=0.0, on_step=monitor,
                      sync=False if legacy_shape else None)
    trace, self.palm_trace = self.palm_trace, prev_trace
    p_end, _ = self.palm_pose()
    # The descent stops on contact, so measure straightness against the part of
    # the line it actually flew: p_start -> where it stopped.
    self._log_move(p_start, p_end, p_end, trace, seg_dev(trace, p_start, p_end),
                   0.0, resid, CART_PATH and not legacy_shape, n_wp,
                   time.time() - t0, self.ik_solves - s0,
                   tag=self.move_tag or "touchdown",
                   extra={"wp_resid_mm": round(diag["wp_resid_m"] * 1000, 1),
                          "wp_jump_rad": round(diag["jump_rad"], 3),
                          "fallbacks": int(bool(diag.get("truncated"))),
                          "rounds": 1})
    self.run(0.3)
    return self.pair_contact("red_block", "table_white")

  def finger_contacts(self):
    n = 0
    for i in range(self.data.ncon):
      c = self.data.contact[i]
      b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                             self.model.geom_bodyid[c.geom1]) or ""
      b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                             self.model.geom_bodyid[c.geom2]) or ""
      if ("red_block" in b1 + b2) and ("right_hand" in b1 + b2):
        n += 1
    return n


# --------------------------------------------------------------------- #
# walk-to-position (gait deadband compensated)
# --------------------------------------------------------------------- #
# M1.14. The walker has a DEAD ATTRACTOR: it can stand perfectly still under a
# 0.40 m/s forward command, indefinitely. Traced over the corrective leg, the
# base position is frozen to the millimetre for the whole remaining 16 s of a
# 20 s budget while `lin_vel_x` reads 0.396 and the policy keeps emitting a
# static action. The observation is Markovian (base velocities, projected
# gravity, joint state, `last_action`, command) so this is a genuine fixed
# point of the closed loop, not a transient. It never once fired on a RETREAT
# (0 of 102) and fired on 21 of 102 forward legs.
#
# `WALK_STALL_S` / `WALK_STALL_M` define the detector (no progress toward the
# goal over this window while a real command is being issued) and it always
# runs: every detection is appended to `STALL_LOG`, which costs nothing and
# makes the failure visible in the episode record. `WALK_STALL_KICKS` arms the
# RECOVERY, which is what changes behaviour: zero the policy's `last_action`
# memory, which is a term of its own observation, and stand for
# `WALK_STALL_PAUSE` s. That is the same handoff `set_crouch` already does when
# it gives the walker back control.
WALK_STALL_S = 2.0        # window with no progress that counts as a stall
WALK_STALL_M = 0.03       # ... and the progress that clears it
# Only count a stall while the goal is far enough that the commanded speed is
# the full `cap`. `speed = max(min(2*dist, cap), 0.30)`, so below dist 0.20 the
# command is riding the 0.30 floor, which is the walker's own deadband edge --
# standing still there is the documented "walk_to floors out at 7-9 cm", a
# different phenomenon, and 671 of the 745 detections over 150 episodes are it.
WALK_STALL_MIN_D = 0.20
WALK_STALL_KICKS = 0      # recovery kicks a single walk_to may spend (0 = off)
WALK_STALL_PAUSE = 0.3    # seconds the kick's own command is held
# What the kick actually commands for `WALK_STALL_PAUSE` seconds. `last_action`
# is zeroed in every mode -- it is a term of the policy's own observation, so
# clearing it is the cheapest available perturbation -- but on its own it only
# escapes some of the fixed points (measured: 3 kicks left five episodes still
# frozen 23-29 cm out), so the mode also commands a way out.
#   "reset"  - zero command (the `set_crouch` handoff, and nothing more)
#   "back"   - walk backward at `cap`; no retreat leg has ever stalled
#   "turn"   - yaw burst at the clip
#   "strafe" - sidestep at the lateral clip
WALK_STALL_MODE = "reset"
STALL_LOG = []            # (goal_dist_at_detection, sim_time, kicked)


def walk_to(runner, goal_xy, goal_yaw=0.0, timeout=20.0, tol=0.07, cap=0.4):
  model, data, ctrl = runner.model, runner.data, runner.ctrl
  goal_xy = np.asarray(goal_xy, float)
  n = int(round(timeout / model.opt.timestep))
  mark_t = float(data.time)
  mark_d = float(np.linalg.norm(goal_xy - data.qpos[:2]))
  kicks = 0
  for i in range(n):
    if runner.state["cs"] % DECIMATION == 0:
      err_w = goal_xy - data.qpos[:2]
      yaw = base_yaw(data)
      c, s = np.cos(yaw), np.sin(yaw)
      err_b = np.array([c * err_w[0] + s * err_w[1],
                        -s * err_w[0] + c * err_w[1]])
      yaw_err = np.arctan2(np.sin(goal_yaw - yaw), np.cos(goal_yaw - yaw))
      dist = np.linalg.norm(err_b)
      scale = err_b / max(dist, 1e-6)
      speed = max(min(2.0 * dist, cap), 0.30)
      ctrl.lin_vel_x = float(speed * scale[0])
      ctrl.lin_vel_y = float(np.clip(speed * scale[1], -0.3, 0.3))
      ctrl.ang_vel_z = float(np.clip(1.5 * yaw_err, -0.6, 0.6))
      if WALK_STALL_S and float(data.time) - mark_t >= WALK_STALL_S:
        stalled = (mark_d - dist < WALK_STALL_M
                   and dist > WALK_STALL_MIN_D)
        if stalled:
          do_kick = kicks < WALK_STALL_KICKS
          STALL_LOG.append((round(dist, 3), round(float(data.time), 1),
                            bool(do_kick)))
          if do_kick:
            kicks += 1
            ctrl.lin_vel_x = ctrl.lin_vel_y = ctrl.ang_vel_z = 0.0
            ctrl.last_action[:] = 0.0
            if WALK_STALL_MODE == "back":
              ctrl.lin_vel_x = -float(cap)
            elif WALK_STALL_MODE == "turn":
              ctrl.ang_vel_z = 0.6 if kicks % 2 else -0.6
            elif WALK_STALL_MODE == "strafe":
              ctrl.lin_vel_y = 0.3 if kicks % 2 else -0.3
            runner.run(WALK_STALL_PAUSE)
        mark_t, mark_d = float(data.time), dist
    runner.step_once()
    if np.linalg.norm(goal_xy - data.qpos[:2]) < tol:
      break
  ctrl.lin_vel_x = ctrl.lin_vel_y = ctrl.ang_vel_z = 0.0
  runner.run(1.5)
  return float(np.linalg.norm(goal_xy - data.qpos[:2]))

