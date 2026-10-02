"""Base-pose primitives over the frozen ONNX body experts.

Measured division of labour:
  walker   translates well (>= 0.3 m/s commands) and turns while walking, but
           is dead for in-place yaw.
  rotator  turns in place fast and accurately (90 deg in ~3 s, ~6 deg error),
           but falls if asked to translate and drifts 8-35 cm while spinning,
           so it is used in short bursts only.
  croucher the walker's observation plus two extra dims (commanded and
           measured pelvis height); trained at zero velocity, so it cannot
           walk.
"""

import numpy as np

from ik import sim as ep
from run import SCRIPT_DIR, ONNXPolicy

ROTATOR = ONNXPolicy(str(SCRIPT_DIR / "rotator.onnx"))
CROUCHER = ONNXPolicy(str(SCRIPT_DIR / "croucher.onnx"))


def crouch_step(ctrl, extra2):
  """Walker-layout obs + 2 extra dims, croucher action pipeline."""
  lin_vel, ang_vel = ctrl._get_base_velocities()
  proj_gravity = ctrl._get_projected_gravity()
  obs = np.concatenate([
    lin_vel, ang_vel, proj_gravity,
    ctrl._get_joint_positions(), ctrl._get_joint_velocities(),
    ctrl.last_action, np.zeros(3, np.float32),  # cmd: trained at zero
    np.asarray(extra2, np.float32),
  ]).astype(np.float32)
  action = CROUCHER(obs)
  target = ctrl.default_joint_pos + action * ctrl.action_scales
  for idx in ctrl.arm_indices:
    target[idx] = ctrl.default_joint_pos[idx]
  ctrl.last_action = action.copy()
  return target


def yaw_err_to(data, goal_yaw):
  return np.arctan2(np.sin(goal_yaw - ep.base_yaw(data)),
                    np.cos(goal_yaw - ep.base_yaw(data)))


def rotate_to(runner, goal_yaw, tol=np.radians(8), timeout=8.0):
  """Rotator burst: swap it into the walker slot, P-control yaw, swap back."""
  ctrl, model, data = runner.ctrl, runner.model, runner.data
  saved = ctrl.walker_policy
  ctrl.walker_policy = ROTATOR
  ctrl.lin_vel_x = ctrl.lin_vel_y = 0.0
  n = int(timeout / model.opt.timestep)
  for i in range(n):
    if runner.state["cs"] % ep.DECIMATION == 0:
      e = yaw_err_to(data, goal_yaw)
      ctrl.ang_vel_z = float(np.clip(1.2 * e, -0.5, 0.5))
    runner.step_once()
    if abs(yaw_err_to(data, goal_yaw)) < tol:
      break
    if data.qpos[2] < 0.5:  # losing balance: hand back to the walker now
      print("    [rotate_to] ABORTING burst: pelvis dropped below 0.5")
      break
  ctrl.walker_policy = saved  # back to walker, settle standing
  ctrl.ang_vel_z = 0.0
  runner.run(1.0)
  return float(np.degrees(abs(yaw_err_to(data, goal_yaw))))


def walk_to_pose(runner, goal_xy, goal_yaw, standoff=0.60,
                 direct_yaw_tol=30.0, direct_min_dist=0.5):
  """Full (x, y, yaw) base pose from the measured strengths of each policy.

  The walker is most accurate on ONE long straight leg and does converge
  modest yaw while translating (4 cm / few deg); the rotator is the only way
  to change heading substantially, but drifts 8-35 cm doing it.

  So: walk direct when the walker can handle the heading itself; otherwise
  dock: stage `standoff` behind the goal along the final heading, spend all
  the rotation there (drift is harmless), then one straight leg in.
  """
  data = runner.data
  goal_xy = np.asarray(goal_xy, float)
  dyaw = abs(np.degrees(yaw_err_to(data, goal_yaw)))
  dist = float(np.linalg.norm(goal_xy - data.qpos[:2]))

  if dyaw < direct_yaw_tol and dist > direct_min_dist:
    ep.walk_to(runner, goal_xy, goal_yaw=goal_yaw, timeout=30.0)
  else:
    heading = np.array([np.cos(goal_yaw), np.sin(goal_yaw)])
    stage = goal_xy - standoff * heading
    # Translate to staging WITHOUT turning; the walker is omnidirectional,
    # and every extra in-place spin is both drift and a fall risk (a ~177 deg
    # rotator burst while holding a loaded outstretched arm fell the robot).
    if np.linalg.norm(stage - data.qpos[:2]) > 0.15:
      hold_yaw = ep.base_yaw(data)
      ep.walk_to(runner, stage, goal_yaw=hold_yaw, timeout=30.0)
    if abs(yaw_err_to(data, goal_yaw)) > np.radians(12):
      rotate_to(runner, goal_yaw)           # the single rotation, at staging
    ep.walk_to(runner, goal_xy, goal_yaw=goal_yaw, timeout=20.0)

  pos_err = float(np.linalg.norm(goal_xy - data.qpos[:2]))
  yaw_e = float(np.degrees(abs(yaw_err_to(data, goal_yaw))))
  return pos_err, yaw_e
