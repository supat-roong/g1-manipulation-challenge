"""The pure IK pick-and-place pipeline: brown table -> blue table.

Base pose  hybrid walker + rotator, docking-style `walk_to_pose` (the walker
           cannot turn in place, the rotator cannot translate), routed around
           the tables when the straight line crosses one
Pick       a stance SEARCHED from the pose actually reached: a kinematic
           feasibility map over cylinder-in-body offsets, corrective legs and
           approach-side changes until a side grasp is reachable, crouching
           when standing cannot reach (`replan_pick`)
Grasp      side grasp flown as an explicit path -- standoff behind and above,
           horizontal traverse, vertical descent -- with the geometry derived
           from the finger mesh (`grasp_geometry`)
Arm        6-DoF kinematic DLS IK + gravity compensation + aim-and-correct
           (the pretrained reacher has an 8-14 cm bias)
Carry      cradle: fingers pitched `CRADLE_PITCH` above horizontal so gravity
           seats the cylinder in the cage, hand tucked near the body
Place      the reachable point on the blue top, crouching only if the IK
           residual says standing cannot reach it, then a contact-monitored
           descent, a seat press and a staged thumb-first release

Success (spec 8.2): the cylinder on the blue table, tilt <= 10 deg, at rest
for 2 s. `run_once` flies one randomized episode and returns its telemetry;
`eval/sweep.py` is the harness.

`COMMANDER` is the seam a high-level policy drives the pipeline through: with
a `direct` commander the replanner is skipped and the commander owns the
stance, crouch and grasp (see `hl/commander.py`).
"""

import contextlib
import sys
import time
import traceback
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ik import base  # noqa: E402
from ik import sim as ep  # noqa: E402

PICK_STANCE = ([-0.40, 0.10], 0.35)
PICK_YAW = 0.35
# Where the cylinder sits in the robot's frame at the stance the grasp was
# tuned at. Deriving the stance from this to track a moved cylinder was tried
# and REGRESSED everything (0/5, nominal included): the offset here comes from
# the stance TARGET, but the passing run actually stood 4.2 cm off it, and the
# grasp is tuned to that real geometry. Kept as telemetry only; closing this
# gap needs the grasp to adapt to the measured offset, not a better stance.
PICK_BODY_OFFSET = np.array([0.35, -0.21])
PLACE_STANCE = ([-0.41, -0.35], -0.35)
PLACE_XY = ep.PLACE_XY
BLUE_TOP_Z = ep.BLUE_TOP_Z
CYL_HALF = ep.CYL_HALF
CROUCH_BIAS = 0.04      # croucher undershoots its command by ~4 cm
# Cradle roll, applied about the palm's OWN z axis, so it rolls the mouth of
# the cage upwards without changing where palm z points. Measured consequence
# (M1.8): the held cylinder's axis stays locked to palm z to within 4 deg for
# the whole carry, so the tilt the place has to seat is set by the GRASP tilt,
# not by this number. 15 deg still let a long loaded translation creep the grip
# 4.1 -> 8.2 cm; 40 deg froze slip but let the cylinder rotate until it lay
# along the pinch axis (tilt 95 deg, held by its end caps).
CRADLE_PITCH = 20
# M1.12: how FAST that roll is flown. It is the single largest source of
# `dropped_in_transport`: sampling the move every 10 physics steps on the seed-0
# episodes that drop, the cylinder is lost 1.2-1.4 s INTO the cradle ramp, with
# the arm marching at a steady 6 mrad per control step and tracking to 15-18
# mrad -- not in a settle, not at the ramp's stop, and not during the walk. The
# roll rotates the FINGERS around a cylinder whose axis is pinned to palm z, so
# a `tilt + CRADLE_PITCH` = 35-40 deg roll asks a mu 3-4 pinch to slide ~1.4 cm
# circumferentially, and at 0.006 rad/step that takes 2.1 s -- longer than the
# ~1.3 s the loss needs to develop. `CRADLE_RATE` is the knob that decides
# which of those two clocks wins. 0.006 replays M1.11.
CRADLE_RATE = 0.006
# M1.12: `CRADLE_STEPS` splits the roll into that many gotos. M1.13: the split
# is tied to `CRADLE_TIGHTEN_FIRST`, so it fires only on the LOW-friction half,
# and it has settle 0.0 on every step but the last, so it is one continuous
# ramp, not a series of stops.
CRADLE_STEPS = 8
# M1.12: WHEN the carry's tighten to 1.08 happens. Its own comment says the
# extra normal force is what "resists rotation about the pinch axis" -- and the
# cradle roll IS a rotation about the pinch axis, flown at grip 1.00 with the
# tighten applied afterwards. `CRADLE_TIGHTEN_FIRST` moves it in front of the
# roll, and `CRADLE_STEPS` then applies (splitting the roll helps only when the
# grip is already up: measured seed 0, steps 4 alone is 16 successes against the
# incumbent 20, and 22 with the tighten in front of it).
#
# Measured over 150 `reachable` episodes it is a 1/mu trade, the same shape as
# `HOLD_CAP`'s: `dropped_in_transport` 50 -> 44 overall, mu 1.5-2.0 success
# 6 -> 10 with drops 19 -> 12, but mu 3.5-4.0 success 4 -> 1 with drops
# 15 -> 19. So it takes the same treatment: True / False for a fixed choice, or
# a `(key, delta_max_m)` pair to decide per episode off the close's depth delta
# (see the M1.12 close-delta note above `CARRY_SETTLE` for what the two keys
# measure and separate).
CRADLE_TIGHTEN_FIRST = ("squeeze", -0.0155)
CRADLE_GRIP = 1.08       # the carry grip, whichever side of the roll it lands
# M1.13: WHETHER the roll happens at all, as a scale on its angle. The whole
# move is `pitch_about(R_s, R_s[:, 2], -(tilt + CRADLE_PITCH))`, and M1.12
# priced the ANGLE through `CRADLE_PITCH` (0 / 10 / 30) without ever pricing
# ZERO -- `CRADLE_PITCH = 0` still rolls by `tilt`, because the angle is the
# sum. `CRADLE_ROLL_SCALE` multiplies the whole angle, so 0.0 carries in the
# GRASP frame and skips the move (and its settle) entirely; 1.0 replays M1.12
# bit-identically.
CRADLE_ROLL_SCALE = 1.0
# M1.13: WHERE the roll is flown. The incumbent gives it a `goto` of its own,
# 2.1 s of pure rotation with a lifted cylinder in the cage and nothing else
# happening -- and that is where 28 of the 42 remaining `dropped_in_transport`
# go. The destination orientation is what the carry needs; the dedicated move
# is not.
#   "own"   -- its own `goto` after the lift (M1.12).
#   "lift"  -- fold it into the LIFT's ramp: the palm rises 15 cm and turns at
#              the same time, so the roll costs no extra seconds and happens
#              while the lift's own acceleration is pressing the cylinder into
#              the cage.
#   "tuck"  -- skip the roll entirely and let the TUCK's `goto` carry the
#              orientation change along with its translation.
CRADLE_AT = "own"

# ------------------------------- M1.8: the seat --------------------------- #
# COMPLIANT SEAT. The grip is ramped to `SEAT_GRIP` over the seat press, from
# the moment the cylinder touches the tabletop, so the pinch stops being a rigid
# clamp while the palm is still pressing down and the table's normal force
# rights the cylinder. `None` restores the M1.7 rigid press.
#
# This is what `place_grid.py` was built to measure, and it is the whole of
# M1.8's gain: over the 36-cell seating grid, upright-and-seated went
# 4/36 -> 12/36 (2/17 -> 10/20 of the cells still holding the cylinder at the
# pre-place), and 11 of the 15 cells that touch down at <= 25 deg of tilt now
# seat, against 1 of the 6 above 30. The M1.6 hypothesis this replaces, "the cradle
# presents 20-28 deg and the press cannot right that", is contradicted by the
# same grid: the RIGID press tipped cells delivered at 2-5 deg of tilt just as
# reliably as ones at 25, so the tilt was never the driver. What tips a rigidly
# clamped cylinder is the release: with the finger pair still at 1.08 the
# thumb-first stage pushes it over (measured tilt 2 -> 25-38 deg at `thumb open`
# in every rigid cell), and opening the pair then sweeps it (one cell threw it
# 27 cm onto the floor). Relaxing the clamp while the table is still being
# pushed against removes the stored force before either stage runs.
#
# 0.5 sits in the middle of a flat plateau, not on a tuned peak: 0.15/0.3/0.4/
# 0.5/0.6 score 13/12/12/14/12 of 36 and only 0.7 falls off (7/36), measured
# before the crouch gate moved, so read the spread, not the absolute numbers.
SEAT_GRIP = 0.5
SEAT_S = 0.8
# The place crouch. Standing residual above which the pelvis drops, and the
# height it drops to. Both are parameters so `place_grid.py` can force the
# crouch branch on a stance that does not need it (`--set
# e2.PLACE_CROUCH_RESID=0.0`) and measure it: on the 30-episode sweep every
# crouched place failed and every success was standing, 0/5 against 4/4.
PLACE_CROUCH_RESID = 0.04
PLACE_CROUCH_H = 0.65
# Un-tuck clearance. The reach-out point is `pelvis + [0.34, -0.22, -0.05]`,
# which at a standing pelvis puts the palm at z ~0.745, and the cradle hangs
# the cylinder ~6.5 cm BELOW the palm, so its base passes within 2-4 cm of the
# blue tabletop at 0.633 (+/-2 cm of randomization).
# MEASURED AND REJECTED: clamping the reach-out z up so the cylinder's base
# clears the tabletop (by 3 / 1 cm, +4.9 / +2.9 cm of raise) does fix exactly
# the two predicted episodes, but the reach-out point sits at the edge of the
# workspace and moving it churns everything downstream: `reachable` seed 0 went
# 5 successes -> 4 -> 3 and `dropped_in_transport` 0 -> 2 -> 2.
# MEASURED AND REJECTED AGAIN, on the cylinder's MEASURED lowest collision
# vertex and a bigger sample: a 0.5 cm clearance on `shipped_table` at n = 150
# is 130/150 = 86.7%, paired +3/-9 (p = 0.15). Raising the reach-out point moves
# the pose the place-point search starts from, 142 of 150 episodes fly a
# different place, and on four of the nine lost the clearance the clamp exists
# to raise went DOWN. The scrape it removes is not what loses the cylinder: the
# cylinder touches the blue top while still held on 471 of 838 un-tucks and 458
# of those succeed.
UNTUCK_OFFSET = (0.34, -0.22, -0.05)
# M3.5 instrument. §3.10's 0.5 cm was computed from the reach-out point and a
# treat-the-cylinder-as-vertical hang; it was never MEASURED in flight, and the
# regime it was filed in (23% success, `GRASP_TILTS` ascending) no longer
# exists. `CARRY_CLEAR_TRACE` samples the real minimum distance between the
# carried cylinder's lowest collision vertex and the blue tabletop through the
# un-tuck, the place crouch and the pre-place move, so "the cylinder is scraped
# off on the blue top during the un-tuck" is a measurement rather than an
# inference from two numbers measured in different regimes. Telemetry only:
# one analytic support-function evaluation per existing monitor sample, no
# extra physics, and `False` removes the samples entirely.
CARRY_CLEAR_TRACE = True
# Horizontal pad, in m, on the blue tabletop's footprint when deciding whether
# the cylinder is over the table at all. The cylinder's widest horizontal
# extent from its axis is the cap boxes' corner, 0.018 * sqrt(2) = 2.55 cm, so
# clearance to the blue top plane only means anything inside that halo.
BLUE_FOOTPRINT_PAD_M = 0.026
# Release order. M1.7's staged thumb-then-pair, or both together.
# MEASURED AND REJECTED: releasing symmetrically scores +2 cells on the seating
# grid (16/36 vs 14) and is WORSE on the sweep: `reachable` over seeds 0/1/2
# went successes 7 -> 4 and `dropped_in_transport` 3 -> 6. The grid's static
# stance cannot see that, which is the one place its verdict and the sweep's
# disagreed in this milestone; the sweep is the one that counts.
RELEASE_STAGED = True
RELEASE_GRIP = 0.30
# Multi-start IK in the place-point search. Off reproduces the M1.7 search
# exactly, which is what `place_grid.py --set e2.PLACE_IK_MS=false` needs.
PLACE_IK_MS = True
# ------------- M3.3: the seated residual, as an instrument ----------------- #
# `best_place_point` chooses the tabletop point by the residual at the PLACE
# pose (cylinder `PLACE_UP_M` above its rest height) and at the pre-place pose
# 6 cm above that. The pose the descent must actually FINISH in -- the cylinder
# ON the tabletop -- is neither scored nor reported, and it is the one that was
# failing: over the 150 verified M3.2 `shipped` episodes `touchdown` reported
# `table contact=False` on 8, and 6 of those 8 were losses, all six labelled
# `dropped_in_transport`. Five stalled after 6.2-6.9 cm of descent with the
# cylinder still 1.4-3.4 cm above the tabletop; the compliant seat is armed by
# contact, so it never fired, and the fingers opened on a cylinder in mid-air.
# The chosen point's PLACE residual is 0.0 mm on three of them, so nothing in
# the existing score could see it.
#
# `rp_seat` is therefore reported on every run -- one extra kinematic solve on
# the IK's own scratch, which cannot perturb the episode (proved: 300/300
# bit-identical). Adding it to the per-cell SCORE was drafted and then not
# needed: the M3.3 tilt order takes `table contact=False` from 8/150 to 1/150
# at `shipped`, because the mechanism is upstream (see `GRASP_TILTS`). It is
# left unmeasured rather than shipped unmeasured.
PLACE_UP_M = 0.02        # m, how far above rest the PLACE pose is commanded

# ==================== M1.7: the explicit grasp path ======================== #
# M1.6 measured that the tuned grasp captured only because `ramp_to` bowed the
# palm 4-14 cm ABOVE the straight pre-grasp -> grasp line and descended onto the
# cylinder: 7/25 captures with that arc, 2/25 flown straight. Every constant it
# was tuned with (GRASP_DZ, PREGRASP_BACK, the tilt sweep) therefore encoded a
# rate-limiter artefact. M1.7 replaces the accident with a path whose shape is
# PARAMETERS, all of them derived from two measured points on the pre-curled
# hand plus the cylinder and tabletop geometry, not fitted to a sweep.
#
# The two hand points are mesh vertices in the palm frame, expressed relative to
# the GRASP CENTRE (the closed-hand fingertip centroid `grasp_center_offset`
# returns, i.e. the point the palm target places on the cylinder), as (x, z):
#   HAND_TIP  the deepest point of the whole hand at 0.25 pre-curl
#   PAIR_TIP  the mean of the index and middle fingertip links: the pinch
# A palm-frame (x, z) at tilt t maps to world up = -x sin t + z cos t and
# horizontal forward = x cos t + z sin t, which is all the algebra below is.
#
# Three measurements drove the design (all from mesh vertices at the IK-solved
# grasp pose, hand pre-curled to the 0.25 the approach uses):
#
# 1. THE CAGE CANNOT ADMIT THE CYLINDER FROM THE FRONT, ONLY FROM ABOVE.
#    The index/middle surfaces sit 1.3-2.0 cm from the grasp centre across the
#    pinch, INSIDE the cylinder's 2.0 cm radius, so a straight-in approach at
#    any tilt drives the fingers into the cylinder's side; that is M1.6's 9-10
#    cm plough. The VERTICAL corridor over the cylinder's axis, by contrast, is
#    empty at every tilt <= 25 deg: no hand vertex passes within 2.0 cm of the
#    axis. So the last segment is a vertical descent, and its clearance
#    requirement is zero.
# 2. THE APEX THE ARC WAS REACHING BY ACCIDENT IS A GEOMETRIC QUANTITY. To put
#    the cage over the cylinder the fingertips must cross its position, so a
#    horizontal traverse must carry the whole hand above the cylinder's top:
#    measured 9.8 cm above a dead-centre grasp pose at 25 deg tilt, 7.6 at 15
#    and 11.7 at 45, and the formula `CYL_HALF - dz - tip_up` reproduces all
#    three to 1 mm. M1.6's arc bowed 4-14 cm and captured exactly when it
#    happened to bow enough.
# 3. A DEAD-CENTRE GRASP IS NOT REACHABLE ANYWHERE. `GRASP_DZ = 0` aimed the
#    grasp centre at the cylinder's centre, on the documented grounds that the
#    pose "sits 3.7 cm above the tabletop by construction". It does not: at
#    every IK-feasible cell of the offset slab the pre-curled hand is 3-7 cm
#    INSIDE the tabletop there (25 deg tilt: the middle fingertip 4.3 cm under
#    the top), because the fingertips overshoot a 4 cm cylinder and dive past
#    it. So the tabletop, not the constant, was setting the grasp height, and it
#    set it by stalling the arm. The height is now commanded explicitly as the
#    higher of the two geometric requirements:
#      * centre the pinch on the cylinder: the grasp centre sits `pair_up`
#        ABOVE the fingertip pair's mid-point, so dead-centre aiming actually
#        puts the pair 2.1 cm low at 25 deg;
#      * keep the hand off the tabletop by TABLE_CLEAR.
#    The table term wins everywhere (by 2.4-2.9 cm at every tilt), which is the
#    honest statement of the conflict: on a table this hand grips a 7.4 cm
#    cylinder with the LOWER finger and the thumb, and the upper finger clears
#    the top. Both are still on the body: the lower fingertip lands within
#    0.5 cm of the cylinder's mid-height at every tilt, which is what the
#    +/-0.8 cm window actually buys here.
CYL_R = 0.02              # scene.xml: red_cylinder radius
# Deepest point of the pre-curled hand, per CURL SIDE. The curl-side sign rolls
# the palm 180 deg about its finger axis (y flips, so z = x cross y flips too),
# which swaps which finger hangs lowest: the middle fingertip for sgn<0, the
# LONGER index fingertip for sgn>0, 1.6-1.8 cm deeper. Using one number for both
# under-clamps the sgn>0 grasp height and the hand fouls the tabletop by 3-12 mm
# (measured), and it was costing the search every small-tilt candidate at the
# inward cells (it fell through to a 45 deg tilt, whose pinch lands above the
# cylinder's rim and captures nothing).
HAND_TIP = {-1.0: (0.0851, -0.0282), 1.0: (0.0830, 0.0461)}
PAIR_TIP = (0.0665, 0.0084)    # index/middle fingertip mean (palm x, z)
TABLE_CLEAR = 0.005      # tabletop clearance the grasp pose keeps
APEX_MARGIN = 0.02       # apex clearance over the geometric minimum
STANDOFF_MARGIN = 0.02   # standoff clearance behind the cylinder
STANDOFF_UP = 0.02       # ... and above the apex, so the traverse arrives at
                         # the apex from ABOVE. Measured: a standoff reached
                         # 3.7 cm low turns the traverse into a climb that
                         # crosses the cylinder's top from underneath and knocks
                         # it over (tilt 0 -> 90 deg), even with 2 cm of nominal
                         # apex clearance.
# --- M1.10: aim the pinch 1 cm shallower in the palm -----------------------
# The quantity that decides the pick is the cylinder's IN-PALM DEPTH AT THE
# CLOSE, and M1.8's cage holds it only over ~3-4.5 cm of it. Measured over the
# widened grid x the friction axis (9 levels, 1.5-4.0), the DESCENT delivers the
# cylinder to 4.3-4.8 cm of depth at EVERY friction (the pre-close state is
# friction-invariant) and the CLOSE then moves it by -0.6 cm at mu >= 2.5 and
# -3.0 cm at mu <= 2.0, where the curling fingers drag it across the tabletop
# and out the bottom of the window.
#
# So the disturbance is not observable before the close (a partial-close probe
# does not separate the buckets either: median slide at grip 0.81 is 0.91 /
# 0.73 / 0.68 cm at mu 1.5 / 2.0 / 3.0 against totals of 3.05 / 3.00 / 0.83),
# and there is nothing friction-dependent to correct at the pre-close pose. What
# a friction-invariant pre-close state DOES admit is a constant: aim shallower
# by roughly the slide the mid-band suffers, so the same close ends inside the
# window over more of the friction draw. Seated cells over the friction axis,
# 34-cell grid: 37 -> 54, with no bucket worse (mu 2.0: 2 -> 11). A plateau, not
# an edge: 0.005 / 0.010 / 0.0125 give 49 / 54 / 53, and 0.02-0.04 overshoot the
# window at mu >= 2.5 and collapse it (5 / 3 seated at mu 3.0).
GRASP_BACK_M = 0.01      # m, back along the palm's finger axis
# How far the fingers are curled during the approach, as a fraction of closed.
# 0.25 is the shipped value and the comment in `fly_grasp_path` explains the
# UPPER bound (0.4 closes the vertical corridor over the cylinder). The lower
# bound is the tabletop: fully open fingertips graze it. It is a constant so
# the descent's finger clearance can be swept against the pinch depth -- a
# deeper pinch only pays if the fingers stop catching the cylinder on the way
# down, which is what topples it (measured: at dz -0.015 the IK reaches 48%
# index contact but 39 of 47 are a fallen cylinder).
GRASP_PRECURL = 0.25
DESCENT_DEG = 90.0       # 90 = vertical; the corridor measurement says vertical
APEX_ROUNDS = 3          # aim-and-correct rounds on the standoff and the apex

# ------------------- M1.9: the pinch-height ceiling ------------------------ #
# M1.8 shipped with `GRASP_TILTS` ascending as a PREFERENCE, so when no shallow
# tilt was IK-feasible from the reached stance the search silently degraded to a
# steep one. Measured over the 30 `reachable` episodes of the M1.8 tree, grouped
# by the tilt the search selected: 15 deg captured 13/17 and owns every success
# in the project; 20 deg 1/2; 25 deg 0/4; 35 deg 0/5; 45 deg 0/1. Ten episodes
# chose a tilt >= 25 and NOT ONE captured, at a median IK residual of 0.0 mm.
#
# The tilt is not the mechanism, though: `dz` is, and the two are not the same
# axis. `grasp_geometry` sets `dz` (how far above the cylinder's centre the pinch
# is commanded) as the larger of "centre the pinch" and "clear the tabletop", and
# the tabletop term depends on the CURL SIDE as strongly as on the tilt, because
# the sign flip swaps the longer index finger in as the hand's deepest point:
#
#   dz (cm)      sgn -1 (zs +1)   sgn +1 (zs -1)
#   tilt 15           1.7              3.4
#   tilt 20           2.3              4.0
#   tilt 25           3.0              4.5
#   tilt 35           4.0              5.3
#   tilt 45           4.8              5.9
#
# So the (tilt, sgn) sweep order is NOT the `dz` order: (35, -1) at 4.0 cm is a
# shallower pinch than (25, +1) at 4.5, and the sweep's own preference was
# spending the good half of the (tilt x sgn) product on candidates that cannot
# work.
#
# `pick_grid.py --force-tilt/--force-sgn` measures the axis directly, replaying
# all ten (tilt, sgn) candidates over the SAME 8 on-table cells:
#
#   tilt sgn   dz cm   captured   IK residual over the 8 cells
#     15  -1    1.94      5/8     3 misses are 32-51 mm, i.e. unreachable
#     20  -1    2.57      8/8     0 mm
#     25  -1    3.17      8/8     0 mm
#     15  +1    3.61      5/8     2 misses are 23 / 59 mm
#     20  +1    4.18      0/8     0 mm
#     35  -1    4.20      0/8     0 mm at EVERY cell
#     25  +1    4.70      0/8     0 mm
#     45  -1    5.02      0/8     0 mm, and zero finger contacts everywhere
#     35  +1    5.55      0/8     0 mm
#     45  +1    6.14      0/8     0 mm
#
# Conditioned on the IK actually solving: 26/27 at dz <= 3.61 cm and 0/47 at
# dz >= 4.18. The tilt is a CONFOUND: (25, -1) captures 8/8 at dz 3.17 while
# (35, -1) captures 0/8 at dz 4.20, and the cliff sits between 3.61 and 4.18,
# which is the cylinder's own half-height. That is the whole mechanism:
#
#   THE PINCH MUST LAND BELOW THE CYLINDER'S TOP FACE.
#
# `dz` is measured from the cylinder's CENTRE, so `dz > CYL_HALF` puts the pinch
# over the rim. `hand_geometry_report` shows what that looks like on the mesh: at
# (20, +1) only `index_0` has a vertex inside the cylinder's radius and below its
# top, with the thumb chain's lowest point 0.3 cm ABOVE the top face, so the
# close lands one jaw on the body and the other on air: `close_contacts` reads
# `['red_block<->index_1']`, a single contact with nothing opposing it. At
# (45, -1) it is the mirror image with `middle_0/1`, and the close registers
# zero contacts at all 8 cells.
#
# So the ceiling is not a tuned number, it is `CYL_HALF`. A candidate above it is
# treated as INFEASIBLE, which hands the problem to the stance search (walk to
# a stance from which a shallow pinch IS reachable) instead of accepting a
# grasp that closes on air. It also makes the search ~2x cheaper, because four of
# the ten candidates never get solved (measured 4023 -> 2112 solves per map).
GRASP_DZ_MAX = CYL_HALF    # m; None = no ceiling (the M1.8 behaviour)

# ------------------------- M1.5: grasp feasibility ------------------------- #
# The pipeline used to commit to a grasp plan computed BEFORE walking and never
# re-planned on arrival, so the whole thing rode on `walk_to_pose` landing on
# the stance the grasp was tuned at. It does not: 4.2 cm from the nominal spawn
# but 5.8-17.6 cm once the spawn moves. Below is the machinery that searches for
# a grasp the arm can actually fly FROM THE POSE IT REACHED, and, when there
# isn't one, re-approaches a stance chosen for the WIDTH of its feasible region.
# Tilt sweep, used in this order as a PREFERENCE: the search early-exits on the
# first candidate that is clean and inside `GRASP_TOL`, so the order decides
# which grasp is flown wherever more than one is feasible.
#
# It was ASCENDING from M1.7 to M3.2, on a kinematic argument: the required apex
# height, the tabletop conflict and the pinch's offset from the cylinder's
# centre all shrink monotonically as the tilt does (apex 7.6 cm at 15 deg vs
# 11.7 at 45), while the IK-reachable set shrinks the other way (a horizontal
# palm is unreachable at every cell of the slab: residual 26-136 mm at 0 deg,
# 0-16 mm at 15 deg, 0-10 mm at 20-25 deg). Nobody re-measured it after M1.9's
# ceiling changed the admitted set, and it was costing 11 episodes in 150.
#
# M3.3 measured it. `GRASP_DZ_MAX` closes the curl-side dimension outright at
# the shipped scene, so exactly three candidates survive -- (15,-1) dz 1.94 cm,
# (20,-1) 2.57, (25,-1) 3.17 -- and tilt and `dz` are perfectly COLLINEAR over
# them. Forcing ONE candidate over the SAME 150 `shipped` episodes (seeds 0/1/2
# at N=50), with a constant `dz` bias breaking the collinearity:
#
#   forced candidate            tilt   dz     end-to-end        dropped
#   (15,-1)                      15   1.94   131/150 = 87.3%      12
#   (15,-1) + dz bias +6.3 mm    15   2.57   135/150 = 90.0%       7
#   (20,-1)                      20   2.57   145/150 = 96.7%       1
#   (20,-1) + dz bias +6.0 mm    20   3.17   143/150 = 95.3%       2
#   (25,-1)                      25   3.17   142/150 = 94.7%       4
#   ascending preference (M3.2)                135/150 = 90.0%     10
#
# Read as a 2x2 on (tilt, dz), paired episode by episode with a sign test:
#
#   dz 1.94 -> 2.57 at tilt 15        131 -> 135   +13/-9   p = 0.52   NULL
#   dz 2.57 -> 3.17 at tilt 20        145 -> 143    +3/-5   p = 0.73   NULL
#   tilt 15 -> 20 at a FIXED dz 2.57  135 -> 145   +15/-5   p = 0.041
#   tilt 20 -> 25 at a FIXED dz 3.17  145 -> 142    +4/-7   p = 0.55   NULL
#
# So `dz` is a NULL on both rungs and the PALM ANGLE is the axis. This runs the
# other way from §3.12 -- there tilt was the confound and `dz` the cause; here,
# with every candidate already below the rim, the pinch height is exhausted.
# Preferring on `dz`, the obvious reading of §3.12, would have bought nothing
# that separates from noise. Both were measured rather than assumed.
#
# 20 against 25 does NOT separate either (p = 0.55). 20 leads on the point
# estimate, and it is the only condition with zero `table contact=False` in 150
# (see below), which is why it is first; 25 sits behind it, not below it.
#
# The mechanism reaches all the way to the PLACE. The 15 deg pinch holds the
# cylinder higher in the hand -- median palm-to-cylinder hang at pre-place 6.19
# cm against 6.46 at 20 deg and 6.66 at 25 -- so the palm must descend further
# to seat it and runs out of down-reach: `touchdown` reports no table contact on
# 10/150 at forced 15 and 0/150 at forced 20. That is where 6 of M3.2's 10
# `dropped_in_transport` came from. They were released 1.4-3.4 cm above the
# tabletop with the compliant seat never armed, and are not transport drops.
#
# 15 deg stays in the list as a FALLBACK rather than being dropped: it is
# sometimes the only candidate a stance offers, and 87.3% is far better than
# refusing. 35/45 are ceiling-filtered at this scene but the constant is not
# scene-specific, so they keep the place the ascending order gave them.
GRASP_TILTS = (20, 25, 15, 35, 45)
GRASP_SIGNS = (-1.0, 1.0)        # separation-axis (curl-side) sign
GRASP_TOL = 0.015                # max IK residual accepted, in m
SEARCH_ITERS = 90                # DLS iterations for grid cells (300 to decide)
PATH_STEPS = 6                   # joint-space waypoints per approach segment
PICK_CROUCH_HEIGHTS = (0.70, 0.65)   # pelvis heights tried after standing fails
# Candidate cylinder-in-body offsets. This is the (dx, dy, dyaw) base-pose grid
# in the coordinates that matter: arm IK in the pelvis frame is YAW-INVARIANT
# (the pelvis is upright, so yaw only rotates the whole problem), which collapses
# a 3-D base-pose grid to a 2-D offset grid and makes the search ~15x cheaper.
STANCE_FWD = np.round(np.arange(0.22, 0.4501, 0.025), 4)
STANCE_LAT = np.round(np.arange(-0.33, 0.0301, 0.03), 4)
BASIN_NEED_M = 0.08     # margin to infeasibility we insist on if we can get it
# Below this margin a standing clean-path basin is not a basin at all; it is a
# point the gait cannot hit (it carries ~13 cm of error), and the padded-grid
# margin of a lone feasible cell is exactly one grid ring, 2.5 cm. Only then is
# the wider path-fouling map preferred; above it, path cleanliness still wins.
# See `basin`, and the M1.7 `spec` regression it repairs.
BASIN_CLEAN_FLOOR_M = 0.04
STANCE_CLEAR_M = 0.22   # keep the base this far outside a tabletop footprint
# M1.14: 2 -> 4. This constant had never actually been measurable -- it was a
# default argument, bound before `_apply_env_overrides` could touch it (see
# `replan_pick`). Read at call time, `reachable` seeds 0/1/2 at N=50 give
# success 61 / 65 / 65 / 66 and `out_of_reach` 12 / 5 / 1 / 0 at a budget of
# 2 / 3 / 4 / 6, i.e. a plateau from 3 on. Paired, the change from 2 to 4
# touches EXACTLY the 12 `out_of_reach` episodes and leaves the other 138
# outcome for outcome identical. Cost: 102 -> 119 legs, median episode sim time
# 107 -> 110 s, worst case 187 -> 210 s against the 300 s limit.
MAX_PICK_LEGS = 4       # corrective legs before an honest `out_of_reach`
# ------------------------------------------------------------------------- #
# M3.2: the palm's approach AZIMUTH, chosen on a measurement
#
# M3.1 made the approach SIDE a search dimension and left `never_captured` as
# the binding failure: 25 of the 35 remaining `shipped_table` losses, against 0
# on `shipped`. What they are is NOT a grip problem and not a stance problem.
#
# What the diagnosis found, in order:
#
# 1. All 25 reach the stance and commit to a grasp the search scores at 0.0 mm
#    residual. 24 of 25 never leave the `aligned` phase, `nudge_m` is 0.000 --
#    the hand never touches the cylinder -- and the pinch ends up a median
#    12.9 cm ABOVE the cylinder's top face. The approach simply does not arrive:
#    executed apex error 19.6 cm median against 1.3 cm on every success.
# 2. The arm is not fighting the actuators. Instrumented on one episode, it
#    tracks its joint command to 0.001 rad in the later rounds. What happens in
#    round 0 is a CONTACT: `table<->right_hand_index_0_link`, 0.374 rad of
#    tracking error, palm stalled 30 cm short. The straight-line raise from the
#    arm's home posture to the standoff drags the hand across the tabletop, and
#    the aim-and-correct rounds then re-aim past a palm that is already jammed
#    and reconfigure the arm onto a branch that flies it 29 cm clear of the
#    table. The search checks the raise at ONE palm-space midpoint, so it sees
#    none of this.
# 3. **Four pre-commit predictors were tried and all four failed to separate**
#    the 25 failures from the 251 successes of the two presets: `plan_cartesian`
#    truncation (0/25), the plan's endpoint residual (0/25), the commanded
#    endpoint's penetration (0/25), and the worst-waypoint penetration -- which
#    fires on 24/25 but ALSO on 21 of the 35 `shipped` episodes that commit to a
#    fouling candidate and place the cylinder anyway. The straight-line raise
#    drags 13-48 mm through the tabletop on every single one of those 35. The
#    drag is normal; its depth predicts nothing.
#    (Fouling itself is equally useless as a gate for the same reason: 35 of 150
#    `shipped` episodes accept an all-fouling candidate and 34 of them succeed.)
#
# ------------------------------------------------------------------------- #
# M3.2's actual repair. The capture failures are the robot WALKING AWAY while
# the arm flies the approach -- see the note in `fly_grasp_path`. Lift the palm
# clear of the tabletop before traversing to the standoff, so the hand never
# jams on the table and the standing controller is never perturbed.
GRASP_PRELIFT = True          # False = M3.1 exactly
# Clearance over the tabletop the pre-lift insists on, and the amount the table
# footprint is inflated by for the "does this line cross the table" test. The
# clearance has to exceed how far the pre-curled fingers hang below the palm
# (2-4 cm, M1.9's `hand_geom`), and the pad covers the hand's own width.
GRASP_PRELIFT_CLEAR_M = 0.08
GRASP_PRELIFT_PAD_M = 0.06
# M3.4. The pre-lift runs ONE aim-and-correct round and M3.2 recorded that it
# sometimes misses badly (12.4 and 22.9 cm). A second round was declined there
# because it would cost sim time on every episode; gating it on "the pre-lift
# undershot" does not fix that -- the pre-lift error is 8-15 cm on 108 of the
# 150 `shipped_table` episodes and 24 of 150 `shipped` ones, almost all of them
# successes, so that gate buys two episodes and moves 130.
#
# The quantity that separates is the one the pre-lift EXISTS to produce: the
# clearance actually reached over the tabletop. Measured over the 149 episodes
# the pre-lift fires on, palm z minus tabletop z at the end of the move:
#
#   | preset          | successes            | non-successes            |
#   | `shipped`       | 7.6 / 10.8 / 17.7 cm | (none below 5 cm)        |
#   | `shipped_table` | 5.1 / 10.7 / 17.7 cm | -3.7, +4.7, then >=10 cm |
#
# so the floor needs no new number at all: it is `GRASP_PRELIFT_CLEAR_M`, the
# clearance the pre-lift already asks for. "Did this move achieve its own goal?"
# fires on 12 of 150 `shipped_table` episodes -- three of them failures, at
# -3.5, +6.0 and +6.8 cm -- and on **0 of 150 `shipped`** ones, where the
# pre-lift meets its target on all 30 fires (min 8.2 cm). The headline is out of
# range of this change by construction, not by tuning.
#
# TWO rounds, not more, and that cap is a measurement. Eleven of the twelve
# episodes the gate fires on are over the floor within two extra rounds (nine
# need two, two need one). The twelfth misses by 23 cm on its first round --
# palm z 0.550, a 37 cm lift -- and it is not "nearly there": the rounds do
# converge on it (-3.5 -> -1.3 -> +1.5 -> +3.9 -> +5.9 -> +9.4 cm, standoff
# error 24.5 -> 1.1 cm, base drift 27.2 -> 0.9 cm), and at five rounds it then
# captures, lifts, and FALLS -- the first `robot_fell` on this preset since
# M3.1. At two it stops at +1.5 cm, reverts to the `never_captured` it already
# was, and costs 2.3 s of sim. Same 136/150 either way, so the cap is set where
# it does not manufacture a fall. `None` disables the retry exactly.
GRASP_PRELIFT_MIN_M = GRASP_PRELIFT_CLEAR_M
GRASP_PRELIFT_RETRIES = 2     # extra rounds allowed when the floor is missed
GRASP_PRELIFT_GAIN_M = 0.005  # ... and stop early once a round stops gaining
# The apex error that says the traverse did not arrive. `shipped` successes max
# at 1.72 cm and `shipped_table` successes at 1.58 cm over 251 episodes, so 3 cm
# is a threshold with a 1.3 cm margin on the side that must not move.
GRASP_APEX_TOL_M = 0.03
RETREAT_M = 0.55        # staging distance back along the approach heading
MIN_RETREAT_M = 0.45    # ... and the shortest retreat the gait can actually do

# ===================================================================== #
# M3.1: the approach SIDE is a search dimension, and getting there
# ===================================================================== #
# S3.23: the 90.7% at shipped physics is position-limited, and the limit is
# structural. `basin` searches cylinder-in-body offsets at ONE heading (the
# one the robot happens to be standing at, which is always the brown table's
# west face), so past ~20 cm from nominal no feasible stance exists at any
# crouch height and the episode aborts `out_of_reach` (56 of the 72
# `shipped_table` failures; 28 of the 29 on seed 0 are `basin` returning None
# on its first call, before a single corrective leg is spent).
#
# Measured coverage over 12 tabletop cells x 12 headings (2026-09-01), best
# available basin margin from ANY heading:
#
#   cyl (x, y)        best yaw   margin   | cyl (x, y)       best yaw  margin
#   (-0.00, -0.21)      + 30     12.5 cm  | (+0.47, -0.21)     +120    12.5 cm
#   (-0.00, +0.00)      + 30     12.5 cm  | (+0.47, +0.00)     +120     2.5 cm
#   (-0.00, +0.21)      + 30     12.5 cm  | (+0.47, +0.21)     +300    12.5 cm
#   (+0.23, -0.21)      +150     10.3 cm  | (+0.71, -0.21)     +120    12.5 cm
#   (+0.23, +0.00)      + 30      2.5 cm  | (+0.71, +0.00)     +210    12.5 cm
#   (+0.23, +0.21)      +300     12.5 cm  | (+0.71, +0.21)     +210    12.5 cm
#
# Ten of the twelve cells have a 10-12.5 cm basin from SOME heading, i.e. the
# capability was never a reach problem, only a heading problem. The two that do
# not are the table's y-centreline at mid-x, and they are not a reach failure
# either: standing `STANCE_CLEAR_M` clear of a tabletop edge that is 0.25 m from
# the centreline puts the cylinder 0.47 m out in pure +y/-y, which only the
# far diagonal corner of the offset grid reaches. That band is the honest
# residual gap of this milestone, not something the yaw sweep can close.
APPROACH_SEARCH = True        # behaviour-preservation switch: False = M1.14
# World headings tried, 30 deg apart. Not "the four faces": the incumbent
# stance's own heading is 20 deg off the west face normal, and the coverage
# table above is won at +120/+150/+210/+300 as often as at an axis face: a
# 0.5 m table with a 0.22 m stand-off is approached from its corners.
APPROACH_YAWS_DEG = tuple(range(0, 360, 30))
APPROACH_NEED_M = BASIN_NEED_M  # margin that ends the heading sweep early
# The basin margin the plan is aimed at is a near-BINARY predictor of the whole
# episode. Pooled over `shipped_table` seeds 0-1, 100 episodes, M3.1 as first
# built:
#
#   margin aimed at   n   success   rate    dominant failure
#   <= 2.5 cm        18      2      11%     never_captured 14
#   2.6 - 4.0 cm     12      1       8%     never_captured 10
#   4.1 - 6.0 cm     23     17      74%
#   6.1 - 8.0 cm     13     10      77%
#   >  8.0 cm        24     18      75%
#
# A cliff at 4 cm: 3/30 = 10% below it, 45/60 = 75% above, and 24 of the 30
# below it die `never_captured`, i.e. the arm flies a clean approach and closes
# on nothing because the gait cannot land inside a one-cell basin. The number is
# not new: `BASIN_CLEAN_FLOOR_M` is already 4 cm for exactly this reason ("the
# padded-grid margin of a lone feasible cell is one grid ring, 2.5 cm"), and
# M1.7 already recorded that a 1-cell basin is a stance the gait will miss.
#
# So the heading sweep fires on a NARROW basin, not only on no basin at all.
# That is what reaches the 20-30 cm band, which the first build left at 3/11 and
# 3/11 against the incumbent's 2/11 and 2/11 -- the incumbent was not aborting
# there, it was aiming at a single cell and missing it.
APPROACH_MIN_MARGIN_M = BASIN_CLEAN_FLOOR_M
APPROACH_MIN_FREE = 4         # non-blocked offset cells a heading must offer
APPROACH_MAX_MAPS = 6         # feasibility maps the heading sweep may build
APPROACH_TURN_M = 0.35        # route cost charged per 90 deg of heading change
# A face change is not a corrective leg and does not spend that budget: the
# leg budget prices the gait's ~13 cm landing error, this prices the walk.
MAX_FACE_CHANGES = 1
# ... but only if it IS one; see `REFACE_NEAR_YAW_DEG` below `NAV_TURN_DEG`.
# Never turn an `out_of_reach` into a `timeout` (`eval/sweep.py` allows 300 s
# and a circling episode runs long): past this much sim time, decline the
# route and abort honestly instead.
FACE_CHANGE_DEADLINE_S = 150.0
NAV_BUDGET_S = 70.0           # sim seconds one route may spend
# Transit clearance around a tabletop footprint. Deliberately SMALLER than
# `STANCE_CLEAR_M`: a validated stance sits 0.22 m out, so a router that
# inflated by 0.22 could not draw a segment to its own goal. It must stay above
# 0.15, because that is what seals the inter-table gap: the tabletops are
# 0.30 m apart in y, the robot is wider than that, and at clear > 0.15 the two
# inflated boxes OVERLAP, so every candidate segment through the gap reads as a
# collision. M1.5's "never path between the two tables" is then geometry
# rather than a special case.
NAV_CLEAR_M = 0.20
NAV_TURN_DEG = 25.0           # heading error worth a rotator burst
# M3.4. `MAX_FACE_CHANGES` bounds walking around the outside of the tables, but
# `approach` searches a 30 deg heading GRID and labels EVERY plan it returns a
# face change -- while the robot's real heading is never on that grid. So once
# refaced to +30 the robot stands at +26.9 and the sweep keeps answering "go to
# +30", which the budget then refuses. That refusal is what ends three of the
# four remaining pick-side `out_of_reach` episodes on `shipped_table`: |dyaw|
# 3.1, 7.0 and 8.8 deg, each buying 5-11 mm of margin over a same-heading plan
# that already existed, and each aborting with 0-1 of its FOUR corrective legs
# spent. A reface the router would fly without a single rotator burst is not a
# change of approach SIDE, so it is priced as the local correction it is: still
# flown as a route (the docking pattern is what lands a stance), but it does not
# spend the face budget. The threshold is `NAV_TURN_DEG` because that is the
# router's own definition of a heading worth turning for, and the measured gap
# is wide -- over 113 executed refaces the |dyaw| distribution is
# 9,12,12,13,16,21 then nothing until 46. 0.0 disables the rule exactly.
REFACE_NEAR_YAW_DEG = NAV_TURN_DEG
# `walk_to` stops the moment it is inside `tol` and then stands for 1.5 s, so a
# transit leg lands up to `tol` + the settle drift SHORT of its waypoint -- at
# 0.15 that is 0.19-0.20 m, measured over every route on `shipped_table` seed 0.
# A corner waypoint has to absorb that or the NEXT leg starts from inside the
# keep-out region and flies along the tabletop: three routes there jammed the
# robot 0.09 m off the brown table's long edge, logged 23-26 stalls, spent 80 s
# and arrived 0.75-1.32 m and 137-166 deg out. So: a tighter leg tolerance, and
# corner waypoints as far out as a STANCE stands (`STANCE_CLEAR_M` past the
# transit clearance), which is the distance the gait is known to be able to
# stop within.
NAV_LEG_TOL_M = 0.10          # position tolerance on a transit leg
NAV_NODE_PAD_M = 0.22         # extra standoff on the corner waypoints
NAV_STAGE_M = 0.60            # length of the final straight leg into a stance
NAV_MAX_LEGS = 8              # legs one route may fly, replanning included
NAV_HANDOFF_RESET = True      # zero `last_action` when the rotator hands back
# M1.14 found a DEAD ATTRACTOR in the walker: the base frozen to the millimetre
# for 16 s of a 20 s budget while `lin_vel_x` reads 0.397, on 21 of 102 forward
# legs. It armed only the detector (`ep.WALK_STALL_KICKS = 0`) because a kick
# was never needed at two or three legs an episode. A route spends four to
# eight, and the first routed transport measured here lost three consecutive
# legs to one stall (7.6 cm in 14.7 s) and arrived 1.13 m from the place stance.
# So the recovery is armed FOR ROUTE LEGS ONLY -- saved and restored around the
# route, so `base.walk_to_pose` and `corrective_leg` are untouched and both
# baselines still replay. Mode "back" is the one M1.14 recorded a reason for:
# no retreat leg has ever stalled.
NAV_STALL_KICKS = 2
NAV_STALL_MODE = "back"
# The carry frame is a WORLD rotation built at the pick, and the place commands
# it verbatim ("no re-aim, no spin" -- the pinch does not constrain rotation, and
# a palm re-aim spun the held cylinder 2 -> 48 -> 81 deg). That is exactly right
# while every pick happens at PICK_YAW: the body turns PICK_YAW -> PLACE_YAW
# between the two, 40 deg, and holding a world frame across that turn IS the
# 40 deg of body-relative re-aim the place was tuned for.
#
# It stops being right the moment the pick heading is a free variable. Written in
# the body frame the place has to fly `Rz(yaw_pick - yaw_place)` of re-aim, so a
# pick at +150 deg asks for 170 deg of it. Measured, once: pick and transport
# both clean (stance err 2.1 cm), place residual 134 mm standing, an unnecessary
# crouch, in-grip tilt 83 -> 90 deg, `tipped_at_place`.
#
# `carry_yaw_fix` rotates the commanded place frame about world z so that the
# BODY-RELATIVE motion the place flies is the tuned one whatever the pick
# heading. It removes a re-aim rather than adding one. Below
# `CARRY_YAW_TOL_DEG` of deviation it is an exact no-op, so the incumbent tree
# is untouched by construction.
CARRY_YAW_COMP = True
CARRY_YAW_TOL_DEG = 30.0

# Ordered milestones. `phase_idx` is the index of the furthest one reached, so
# partial progress is comparable across episodes (the challenge explicitly
# grades partial progress, and the failure taxonomy is derived from it).
PHASES = ("start", "at_pick", "aligned", "grasped", "lifted", "transported",
          "released", "placed")
MONITOR_EVERY = 10      # physics steps between telemetry samples (0.05 s)
UPRIGHT_TILT_DEG = 10.0  # spec 8.2 success threshold
SEATED_TILT_DEG = 25.0   # in-grip tilt the compliant seat can still right
# The close and the lift, as parameters (M1.9). Every value below is the M1.8
# incumbent, and each was made a parameter only to price a candidate repair for
# the two friction-driven failures M1.9 isolated. ALL FOUR REPAIRS FAILED, which
# is why the mechanism is documented as a limitation rather than fixed:
#
# (a) LOW FRICTION (mu <= 2.0) slides the cylinder deep into the palm during the
#     close, and the cage then rotates it to 37-63 deg. On the widened grid at
#     mu 1.7: 13/34 captured, **0/34 seated**, median in-palm depth 1.2 cm.
#     Slowing the squeeze (`CLOSE_S` 0.8 -> 2.0), caging first (a partial grip of
#     0.5/0.6/0.75 with a settle), and raising or lowering `CLOSE_GRIP` (0.9,
#     1.05, 1.1, 1.15) all score 0-1/34 seated. Forcing each tilt at mu 1.7 is
#     0/8 seated at 15, 20 AND 25 deg, against 5/8, 8/8, 8/8 at mu 3.0, so it is
#     not a geometry choice and no knob in `set_grip`'s reach touches it.
# (b) HIGH FRICTION (mu >= 3.8) sticks the cylinder to the tabletop and the
#     straight-up lift topples it instead of picking it up: at mu 4.0 the close is
#     PERFECT (in-palm depth 3.7-4.4 cm, in-grip tilt 0-13 deg, 4-12 contacts on
#     `middle_0/1` and `thumb_1/2`) and then `lift_rise_m` is -1.7 cm with zero
#     contacts. `LIFT_RATE` 0.008 -> 0.003 leaves it at 0/34; `CLOSE_GRIP` 1.15
#     recovers 2/34 and M1.6 measured 1.1 as ejecting at nominal friction, so it
#     is not worth the risk for two cells.
CLOSE_GRIP = 1.0         # 1.1 ejects it, 0.85 slips more (M1.6)
CLOSE_S = 0.8            # seconds to ramp the grip onto the cylinder
LIFT_RATE = 0.008        # joint ramp rate for the straight-up lift
LIFT_M = 0.15            # how far straight up the lift goes
# M1.11: the lift's own `goto` shape. `rounds=2` measures the palm error after
# the ramp's 0.8 s settle and re-aims -- but the arm is still RINGING then (see
# `ep.RAMP_TAPER`), so the correction is measured at an arbitrary phase of a
# ~1 Hz oscillation and commands a spurious ~1 cm move that re-accelerates the
# arm with the cylinder in the cage. `LIFT_ROUNDS` and `LIFT_SETTLE` price that.
LIFT_ROUNDS = 2
LIFT_SETTLE = 0.1
# M1.11: `ep.GRIP_CAP` (a per-finger torque limit -- see its comment) armed
# after the squeeze. `HOLD_CAP` limits only what happens after it, so the close
# still seats the cylinder at full force and the lift and carry hold it with a
# bounded one. In radians of preload; None leaves the raw position command.
HOLD_CAP = 0.02
# The bounded grip is aimed at a STATIONARY hold, which is what the lift's
# settle is; the carry is motion, and measured end to end a cap that stays on
# through it trades the pick's gain straight back (seed 0: `never_captured`
# 8 -> 3 but `dropped_in_transport` 12 -> 15, success 18 -> 19). The fingers get
# their full position command back where `CARRY_CAP_AT` says.
# WHERE the cap is handed over, which M1.12 measured to matter more than what it
# is handed to. Sampling the cradle move every 10 physics steps on the seed-0
# episodes that drop says the carry loses the cylinder in its FIRST move, not in
# the walk and not at the place: at `[lifted]` the grasp reads 8-10 finger
# contacts, 4 cm of slip and 13 deg of in-grip tilt, and 1.3-1.9 s after
# `close_and_lift` hands the fingers their full position command back the tilt
# runs 13 -> 16 -> 22 -> 40 -> 48 deg and the contacts go to zero, with the palm
# still tracking its command. That is M1.11's stationary-hold instability
# exactly, one stage later and with the bound taken off.
#   "lift"   -- release at the end of `close_and_lift` (M1.11).
#   "cradle" -- release after the cradle move, i.e. keep the lift's bound
#               through the carry's first move and hand the grip over at the
#               1.08 tighten, which is where the tuck's motion starts.
#   "tuck"   -- ... or keep it through the tuck as well.
# M1.13: ... and it may be scheduled, like the cradle tighten,
# because the trade is the same 1/mu one. A `(key, delta_min_m, above, below)`
# tuple reads the close's measured depth delta and hands back `above` when the
# delta is >= the threshold (the HIGH-friction side -- the deltas run -2.66 cm
# at mu 1.5 to -0.91 at mu 4.0, so "less negative" is "more friction") and
# `below` otherwise. M1.12 priced "cradle" UNCONDITIONALLY (seed 0: 17 against
# the incumbent 20) and rejected it; the roll's surviving losses are at
# mu >= 3, which is exactly the half a fixed setting cannot serve.
CARRY_CAP_AT = "lift"
# --- M1.12: the close-delta note: a measured friction observable --------- #
# One fixed `HOLD_CAP` cannot serve a 1.5-4.0 friction draw: a pinch needs
# normal force ~ W/mu, so 0.01 takes mu 1.5 from 0 to 12 seated and mu 2.0 from
# 10 to 0 (M1.11). What breaks the tie is that the CLOSE ITSELF measures the
# friction. The descent delivers the cylinder to 5.3 cm of in-palm depth at
# EVERY friction (median 5.28-5.37 over the 34-cell grid x 7 friction levels),
# and the close then drags it in by an amount that is a clean function of mu:
# measured post-close depth delta, median over the grid,
#
#   mu     1.5    2.0    2.5    3.0    3.5    3.8    4.0
#   delta -4.00  -1.22  -1.38  -1.39  -1.16  -1.07  -0.99   cm
#
# and per cell the mu 1.5 population is -2.19..-5.18 cm against -3.17 cm for the
# most extreme cell at any other friction: a threshold anywhere in [-3.6, -3.2]
# puts 16 of 18 mu 1.5 cells on one side and 0 of 90 higher-friction cells on
# the other. Unlike the PRE-close state, which M1.10 measured to be
# friction-invariant, this is a live per-episode friction estimate, available
# after the close and before the lift -- exactly where the cap is armed.
#
# That delta comes in two flavours, and they are informative about DIFFERENT
# parts of the friction axis, so both are measured and a schedule names the one
# it reads:
#
#   "close"   -- depth after the close's 0.8 s settle, minus the pre-close
#                depth. M1.10's quantity. Separates mu 1.5 from mu >= 2.0 with
#                103/104 accuracy (threshold -3.5 cm: 16 of 18 mu 1.5 cells
#                below it, 0 of 90 higher-friction cells), and mu 2.0 from
#                mu >= 2.5 with only 74/87.
#   "squeeze" -- depth at the END OF THE SQUEEZE RAMP, minus the pre-close
#                depth, i.e. the same delta with the settle's own creep left
#                out. Monotone in mu over the WHOLE range (median -2.66 /
#                -1.85 / -1.46 / -1.12 / -1.02 / -1.00 / -0.91 cm at mu 1.5 ->
#                4.0) and separates mu 2.0 from mu >= 2.5 with 83/90 -- which
#                the "close" delta cannot do at all, because at mu 1.5 the
#                cylinder keeps creeping through the settle and at mu 2.0-2.5 it
#                does not, so the settle scrambles exactly that comparison.
#
# Neither separates the high tail (mu >= 3.8 from mu 2.5-3.5: ~78%).
# --- M1.12: how long the CARRY stands still -------------------------------- #
# M1.11's mechanism: an over-driven pinch loses a near-massless cylinder during
# a STATIONARY hold, monotonically in how long the hold lasts (`LIFT_SETTLE`
# 0.1 / 0.3 / 0.8 -> 9 / 9 / 6 seated at mu 3.8), and the ramp's own 8 mrad
# staircase is what keeps the contact from getting going. Shortening the lift's
# settle was half of M1.11's repair -- and then `dropped_in_transport` went
# 30 -> 50, because the carry is four more `goto`s (cradle, tuck, un-tuck,
# pre-place = 7 aim-and-correct rounds) EACH ending in the same 0.8 s stationary
# settle: 5.6 s of standing still with the cylinder in an unbounded grip.
# `CARRY_SETTLE` is `LIFT_SETTLE` for those moves. 0.8 replays M1.11 exactly.
CARRY_SETTLE = 0.8
CLOSE_SAMPLES = 8        # depth samples taken through the squeeze (M1.10)
# --- M1.10: the squeeze, closed on the measurement it was blind to ----------
# In-palm depth is the quantity M1.8's cage envelope is written in (it holds
# over ~3-4.5 cm, and at <= 2 cm it holds but stops ALIGNING, rotating the
# cylinder to 53-59 deg). Measured over the widened grid x the friction axis,
# the descent delivers the cylinder to 4.3-4.8 cm at EVERY friction, and it is
# the close that moves it: -0.6 cm at mu >= 2.5 and -3.0 cm at mu 1.5, where
# the curling fingers drag it across the tabletop and out the bottom of the
# window. So the fix cannot be a pre-close correction (there is nothing
# friction-dependent to correct yet) and it cannot be a constant grip (M1.9
# measured four of those). It is a GATE ON THE SQUEEZE: stop advancing the grip
# the moment the cylinder leaves the window, which is a no-op wherever the
# incumbent already holds it.
CLOSE_GATE_MIN_GRIP = 0.7  # never freeze below this; an open hand holds nothing
CLOSE_GATE_EVERY = 10      # physics steps between gate checks (0.05 s)


# High-level policy seam (hl/commander.py). `None` = the script's own fixed
# `PICK_STANCE` / `PLACE_STANCE`, bit for bit. Otherwise an object with
# `pick_stance(ctx)` and `place_stance(ctx)`, each returning `(xy, yaw)` in the
# world frame. A commander with `direct = True` also owns the crouch and the
# grasp (`grasp`, `move_instead`), and the replanner is skipped; otherwise the
# script verifies and corrects everything after the stance.
COMMANDER = None


def _command_ctx(runner, geo, stage):
  """What a commander may read: the scene as the script already knows it."""
  data = runner.data
  return {"stage": stage, "geo": geo, "cyl": runner.cyl_pos().copy(),
          "base_xy": data.qpos[:2].copy(), "base_yaw": float(ep.base_yaw(data))}


def _ctx_record(ctx):
  """JSON form of a `_command_ctx`, logged on EVERY run (commander or not), so
  a learner trains on exactly what the commander was shown."""
  g = ctx["geo"]
  return {"cyl": [round(float(v), 4) for v in ctx["cyl"]],
          "base_xy": [round(float(v), 4) for v in ctx["base_xy"]],
          "base_yaw": round(float(ctx["base_yaw"]), 4),
          **{k: [round(float(v), 4) for v in np.atleast_1d(g[k])]
             for k in ("brown_center", "brown_half", "blue_center",
                       "blue_half")},
          "brown_top_z": round(float(g["brown_top_z"]), 4),
          "blue_top_z": round(float(g["blue_top_z"]), 4)}


def _base_pose(data):
  return [round(float(data.qpos[0]), 4), round(float(data.qpos[1]), 4),
          round(float(np.degrees(ep.base_yaw(data))), 2)]


def direct_grasp_frame(runner, search, cyl, tilt, sgn, az_deg):
  """The grasp frame for ONE commanded (tilt, curl side, azimuth), built by the
  search's own `_frames` so the geometry is the script's to the last term --
  but with no ceiling, no ordering and no feasibility test: the commander's
  choice is flown as given."""
  g = globals()
  saved = (g["GRASP_TILTS"], g["GRASP_SIGNS"], g["GRASP_DZ_MAX"])
  try:
    g["GRASP_TILTS"], g["GRASP_SIGNS"], g["GRASP_DZ_MAX"] = (
      (float(tilt),), (float(sgn),), None)
    out = search._frames(runner.data.site_xpos[runner.ik.site_id][:2], cyl,
                         az_deg=float(az_deg))
  finally:
    g["GRASP_TILTS"], g["GRASP_SIGNS"], g["GRASP_DZ_MAX"] = saved
  return out[0][0]


# The episode's wall deadline as MODULE state, the way `ep.STALL_LOG` is.
#
# `Runner._monitor` enforces `sim_limit` and `wall_limit`, but it only runs
# from inside the physics loop. The stance search is pure IK -- it never steps
# the sim -- so neither guard could fire inside it, and a search that went
# pathological ran unbounded. Measured: a `shipped` episode with the brown
# tabletop dropped to 0.613 m held one worker at 89% CPU for over 11 minutes
# against a 600 s wall limit, because multi-start IK on an out-of-reach target
# is slow and `basin` builds up to `APPROACH_MAX_MAPS` maps of them.
#
# `_search_deadline` closes that hole. `run_once` sets it, `_map_at` checks it
# per grid cell, and the episode unwinds as a `timeout` -- which is the honest
# classification for "the search could not find a stance inside the budget".
_SEARCH_DEADLINE = None


def _set_search_deadline(wall_limit):
  global _SEARCH_DEADLINE
  _SEARCH_DEADLINE = None if wall_limit is None else time.time() + wall_limit


def _check_search_deadline():
  if _SEARCH_DEADLINE is not None and time.time() > _SEARCH_DEADLINE:
    raise EpisodeAborted("timeout")


class EpisodeAborted(Exception):
  """Raised from inside the physics loop to end an episode early."""

  def __init__(self, reason):
    super().__init__(reason)
    self.reason = reason


class _NoRenderer:
  """Stand-in for mujoco.Renderer: no GL context, no frames."""

  def __init__(self, *a, **k):
    pass

  def update_scene(self, *a, **k):
    pass

  def render(self):
    return np.zeros((480, 640, 3), np.uint8)


class _NoWriter:
  def __init__(self, *a, **k):
    pass

  def write(self, frame):
    pass

  def release(self):
    pass

  def isOpened(self):
    return False


@contextlib.contextmanager
def headless():
  """Run `ep.Runner.__init__` without a renderer or a video file.

  Video writing dominates the wall clock of a sweep and a per-episode GL
  context is both slow and a crash risk in worker processes, but Runner
  always builds both. Patching the two constructors it looks up is the
  least invasive way to skip them without touching e2e_pick_place.py.
  """
  r, w = mujoco.Renderer, cv2.VideoWriter
  mujoco.Renderer, cv2.VideoWriter = _NoRenderer, _NoWriter
  try:
    yield
  finally:
    mujoco.Renderer, cv2.VideoWriter = r, w


def scene_geometry(model):
  """Actual tabletop geometry read from the model (never hard-coded, so the
  height randomization stays consistent with the place logic)."""
  def bid(n):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

  def gid(n):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

  brown, blue = bid("table"), bid("table_white")
  gb, gw = gid("table_top"), gid("table_white_top")
  return {
    "brown_center": model.body_pos[brown].copy(),
    "brown_half": model.geom_size[gb].copy(),
    "brown_top_z": float(model.body_pos[brown][2] + model.geom_size[gb][2]),
    "blue_center": model.body_pos[blue].copy(),
    "blue_half": model.geom_size[gw].copy(),
    "blue_top_z": float(model.body_pos[blue][2] + model.geom_size[gw][2]),
  }


def brown_table_bounds(model, margin=0.06):
  """(x_lo, x_hi, y_lo, y_hi) of the brown tabletop interior."""
  g = scene_geometry(model)
  c, h = g["brown_center"], g["brown_half"]
  return (float(c[0] - h[0] + margin), float(c[0] + h[0] - margin),
          float(c[1] - h[1] + margin), float(c[1] + h[1] - margin))


def randomize_scene(model, data, spawn_yaw=0.0, cyl_xy=None, cyl_shift=(0., 0.),
                    cyl_mass_scale=1.0, cyl_friction=None,
                    brown_dz=0.0, blue_dz=0.0,
                    brown_dxy=(0., 0.), blue_dxy=(0., 0.),
                    table_friction=None, hand_friction=None,
                    brown_scale=(1., 1.), blue_scale=(1., 1.)):
  """Apply domain randomization to a freshly built sim (spec 8.2).

  Tables are static bodies, so shifting `body_pos` is enough; the legs are
  children and move with the top. Returns the resulting scene facts.

  `brown_dxy` / `blue_dxy` shift a table horizontally. The CYLINDER RIDES WITH
  THE BROWN TABLE: `cyl_xy` is drawn in the tabletop's own frame, so the two
  axes stay orthogonal: "where the table is" and "where on the table the
  cylinder is" are separate questions, and a table shift can never place the
  cylinder off its own top. Nothing downstream is told a table moved; the place
  point, both nav keep-outs and the tabletop bounds all re-read `body_pos`
  through `scene_geometry`, so this axis measures whether that is really true.

  `brown_scale` / `blue_scale` multiply the tabletop's half-size per axis; the
  legs are sibling geoms and are moved to the new corners.
  """
  def bid(n):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

  brown_dxy = np.asarray(brown_dxy, float).reshape(2)
  blue_dxy = np.asarray(blue_dxy, float).reshape(2)
  if brown_dz:
    model.body_pos[bid("table"), 2] += brown_dz
  if brown_dxy.any():
    model.body_pos[bid("table"), :2] += brown_dxy
  if blue_dz:
    model.body_pos[bid("table_white"), 2] += blue_dz
  if blue_dxy.any():
    model.body_pos[bid("table_white"), :2] += blue_dxy

  def scale_table(body, top, scale):
    """Scale a tabletop's footprint; the legs (sibling geoms) follow."""
    sx, sy = float(scale[0]), float(scale[1])
    if sx == 1.0 and sy == 1.0:
      return
    b = bid(body)
    gt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, top)
    model.geom_size[gt, 0] *= sx
    model.geom_size[gt, 1] *= sy
    # Scaling geom_size alone leaves MuJoCo's cached broad-phase bounds
    # stale; with a scale factor > 1 the newly added footprint area has no
    # collision until these are refreshed (shrinking is safe on its own:
    # a stale larger bound only over-triggers narrow phase).
    model.geom_rbound[gt] = float(np.linalg.norm(model.geom_size[gt]))
    model.geom_aabb[gt, :3] = 0.0
    model.geom_aabb[gt, 3:] = model.geom_size[gt]
    for g in range(model.ngeom):
      if model.geom_bodyid[g] == b and g != gt:
        model.geom_pos[g, 0] *= sx
        model.geom_pos[g, 1] *= sy
    # The table is a compound static body (top + 4 legs), so MuJoCo also
    # compiled a per-body midphase BVH (`bvh_aabb`) over those geoms; that
    # tree is baked in at compile time from the *original* sizes/positions
    # and mj_forward never rebuilds it, so an enlarged footprint is still
    # broad-phase-culled even after geom_size/geom_rbound/geom_aabb are
    # refreshed above (verified: a cylinder dropped over the added area
    # falls straight through with only those three refreshed). Disabling
    # midphase makes collision fall back to exhaustive geom-pair testing,
    # which is always correct; only episodes that actually request a
    # footprint scale pay for it; the default (1., 1.) scale never reaches
    # this line.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE

  scale_table("table", "table_top", brown_scale)
  scale_table("table_white", "table_white_top", blue_scale)
  brown_scaled = tuple(brown_scale) != (1.0, 1.0)

  brown_moved = bool(brown_dz) or bool(brown_dxy.any()) or brown_scaled
  geo = scene_geometry(model)

  if spawn_yaw:
    data.qpos[3] = np.cos(spawn_yaw / 2)
    data.qpos[4:6] = 0.0
    data.qpos[6] = np.sin(spawn_yaw / 2)

  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
  adr = model.jnt_qposadr[jid]
  if cyl_xy is not None:
    data.qpos[adr] = float(cyl_xy[0]) + brown_dxy[0]
    data.qpos[adr + 1] = float(cyl_xy[1]) + brown_dxy[1]
  else:
    data.qpos[adr] += cyl_shift[0] + brown_dxy[0]
    data.qpos[adr + 1] += cyl_shift[1] + brown_dxy[1]
  if cyl_xy is not None or brown_moved:
    # Drop it just above the (possibly moved) tabletop instead of the XML's
    # 12 cm free-fall: same rest pose, no bounce lottery.
    data.qpos[adr + 2] = geo["brown_top_z"] + ep.CYL_HALF + 0.01
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    dadr = model.jnt_dofadr[jid]
    data.qvel[dadr:dadr + 6] = 0.0

  cb = bid("red_block")
  if cyl_mass_scale != 1.0:
    model.body_mass[cb] *= cyl_mass_scale
    model.body_inertia[cb] *= cyl_mass_scale
  if cyl_friction is not None:
    for gname in ("red_cylinder", "red_cap_top", "red_cap_bot"):
      g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
      model.geom_friction[g, 0] = float(cyl_friction)

  # M1.10 decoupling instrument. `cyl_friction` sets the CYLINDER's geoms, and
  # MuJoCo combines an auto contact's friction as the element-wise MAX of the
  # two geoms, so one number raises the cylinder<->tabletop pair and the
  # cylinder<->finger pair together, and the sweep cannot tell which of them
  # drives the high-friction lift topple. Raising the table's or the hand's own
  # friction above a moderate cylinder separates them: with the cylinder at 3.0,
  # `table_friction=4.0` gives a 4.0 base and a 3.0 grip, and
  # `hand_friction=4.0` the reverse.
  if table_friction is not None:
    for gname in ("table_top", "table_white_top"):
      g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
      if g >= 0:
        model.geom_friction[g, 0] = float(table_friction)
  if hand_friction is not None:
    for g in range(model.ngeom):
      bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                             int(model.geom_bodyid[g])) or ""
      if "right_hand" in bn:
        model.geom_friction[g, 0] = float(hand_friction)

  mujoco.mj_forward(model, data)
  geo["cyl_start"] = data.xpos[cb].copy().tolist()
  geo["cyl_mass"] = float(model.body_mass[cb])
  geo["cyl_friction"] = float(
    model.geom_friction[mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "red_cylinder"), 0])
  return geo


class HybridRunner(ep.Runner):
  """Runner whose base policy can be swapped walker <-> croucher.

  Setting crouch_h routes the locomotion slot through the croucher at that
  commanded pelvis height; the arm IK override, ramped grip, gravity comp and
  video capture in Runner.step_once all still apply.
  """

  def __init__(self, *args, video="v2_run.mp4", snapshots=None,
               sim_limit=None, wall_limit=None, **kwargs):
    if video is None:
      with headless():
        super().__init__(*args, **kwargs)
      self.recording = False
      self.snapshots = False if snapshots is None else snapshots
    else:
      super().__init__(*args, video_name=video, **kwargs)
      self.snapshots = True if snapshots is None else snapshots
    self.crouch_h = None
    # Keep the REACHER while the croucher drives the legs. `crouch_step`
    # replaces `G1Controller.step` wholesale and ends by writing the arm
    # joints to their defaults, so with this False -- what every run before
    # `G1HL_S2_STAND` did -- gating the croucher STOWS the right arm. The
    # pipeline only ever crouches with the arm already stowed, so False is
    # what it wants and is the default; nothing but a skill that gates the
    # croucher with the hand loaded needs the overlay.
    self.crouch_reach = False
    # --- telemetry / episode guards ---
    self.sim_limit = sim_limit
    self.wall_deadline = None if wall_limit is None else time.time() + wall_limit
    self.slip_trace = []
    self.trace_slip = False
    # M1.10: a telemetry hook that fires from inside the physics loop, so a
    # move's own settle is sampled too. `goto`'s `on_step` is not called during
    # `ramp_to`'s trailing `run(settle)`, and that 0.8 s hole is exactly where
    # the high-friction lift loses the cylinder.
    self.sampler = None
    # slip watchdog for the bounded grip (M1.11)
    self.hold_ref = None
    self.hold_tol = None
    self.hold_released = False
    self.max_tilt = 0.0
    self.min_pelvis_z = float(self.data.qpos[2])
    self.fell = False
    self._fall_hits = 0

  def snap(self, name, lookat=None):
    if getattr(self, "snapshots", True):
      super().snap(name, lookat)

  def step_once(self):
    if self.crouch_h is None:
      super().step_once()
    else:
      h = self.crouch_h
      saved = self.ctrl.step
      if self.crouch_reach:
        self.ctrl.step = lambda: self.ctrl.arm_overlay(base.crouch_step(
          self.ctrl, [h, float(self.data.qpos[2])]))
      else:
        self.ctrl.step = lambda: base.crouch_step(
          self.ctrl, [h, float(self.data.qpos[2])])
      try:
        super().step_once()
      finally:
        self.ctrl.step = saved
    self._monitor()

  def _monitor(self):
    """Sample telemetry and enforce the episode guards. Raises
    EpisodeAborted('robot_fell' | 'timeout') from inside the physics loop, so
    every caller in the pipeline unwinds at once."""
    if self.sim_limit is not None and self.data.time > self.sim_limit:
      raise EpisodeAborted("timeout")
    if self.wall_deadline is not None and time.time() > self.wall_deadline:
      raise EpisodeAborted("timeout")
    if self.state["cs"] % MONITOR_EVERY:
      return
    self.max_tilt = max(self.max_tilt, self.cyl_tilt_deg())
    z = float(self.data.qpos[2])
    self.min_pelvis_z = min(self.min_pelvis_z, z)
    if self.sampler is not None:
      self.sampler()
    # The bounded grip's watchdog: give the fingers their full command back the
    # moment the cylinder starts leaving the palm. Checked on the monitor's own
    # schedule, so `ramp_to`'s trailing settle is covered too.
    if (self.hold_tol is not None and self.grip_cap is not None
        and self.slip() > self.hold_tol):
      self.grip_cap = None
      self.hold_released = True
    if self.trace_slip and self.state["cs"] % (2 * MONITOR_EVERY) == 0:
      self.slip_trace.append((round(float(self.data.time), 2),
                              round(self.slip(), 4)))
    # pelvis upright component: R[2,2] = 1 - 2(qx^2 + qy^2)
    up = 1 - 2 * (self.data.qpos[4] ** 2 + self.data.qpos[5] ** 2)
    if z < 0.30 or up < 0.5:      # crouching keeps z >= ~0.55 and up ~ 1
      self._fall_hits += 1
      if self._fall_hits >= 4:    # 0.2 s sustained, not a transient
        self.fell = True
        raise EpisodeAborted("robot_fell")
    else:
      self._fall_hits = 0

  def set_crouch(self, h_cmd, seconds=3.0):
    """Crouch to a commanded pelvis height (None -> hand back to walker)."""
    self.crouch_h = h_cmd
    self.run(seconds)
    if h_cmd is None:
      self.ctrl.last_action[:] = 0  # clean handoff back to the walker

  def slip(self):
    """Cylinder depth along the finger axis in the palm frame (grasp slip)."""
    p = self.data.site_xpos[self.ik.site_id]
    R = self.data.site_xmat[self.ik.site_id].reshape(3, 3)
    return float((R.T @ (self.cyl_pos() - p))[0])

  def report(self, tag):
    print(f"    [{tag}] slip {self.slip() * 100:4.1f} cm, "
          f"in-grip tilt {self.cyl_tilt_deg():3.0f} deg, "
          f"contacts {self.finger_contacts():2d}, "
          f"pelvis z {float(self.data.qpos[2]):.3f}, "
          f"palm z {float(self.data.site_xpos[self.ik.site_id][2]):.3f}, "
          f"cyl z {float(self.cyl_pos()[2]):.3f}")


def pitch_about(R, axis, deg):
  """Rotate frame R by `deg` about a world axis (Rodrigues)."""
  t = np.radians(deg)
  n = axis / np.linalg.norm(axis)
  K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
  return (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K) @ R


def yaw_rot(rad):
  """Rotation about world z."""
  c, s = np.cos(float(rad)), np.sin(float(rad))
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def carry_yaw_fix(R_c, yaw_pick, yaw_place, T=None):
  """The place's carry frame, corrected for a pick made at another heading.

  The pipeline's place is tuned for one heading pair, `PICK_YAW -> PLACE_STANCE`
  yaw. Whatever the pick heading actually was, this rotates the commanded frame
  about world z so the palm motion the place flies IS that tuned motion in the
  BODY frame. See the note at `CARRY_YAW_COMP`.
  """
  tuned = float(PLACE_STANCE[1]) - float(PICK_YAW)
  d = float(yaw_place) - float(yaw_pick) - tuned
  d = float(np.arctan2(np.sin(d), np.cos(d)))
  # bool(...) around the WHOLE test, not just the flag: a numpy comparison
  # returns `np.bool_`, which `json.dump` refuses -- and `eval/sweep.py`
  # serializes this dict, so the leak killed a whole sweep's artifacts
  # after its summary had already printed.
  on = bool(CARRY_YAW_COMP and abs(d) > np.radians(CARRY_YAW_TOL_DEG))
  if T is not None:
    T["carry_yaw_delta_deg"] = round(float(np.degrees(d)), 1)
    T["carry_yaw_comp"] = on
  if not on:
    return R_c
  print(f"[7 CARRY ] pick heading {np.degrees(yaw_pick):+.0f} deg, place "
        f"{np.degrees(yaw_place):+.0f}: rotating the carry frame "
        f"{np.degrees(d):+.0f} deg about world z so the place flies the tuned "
        f"body-frame re-aim")
  return yaw_rot(d) @ R_c


def horizontal_frame(runner, target_xy, tilt_deg, sep_axis_sign=1.0):
  """Palm frame with fingers aimed at target_xy, tilted tilt_deg below
  horizontal (negative = above)."""
  d = np.asarray(target_xy, float) - runner.data.site_xpos[runner.ik.site_id][:2]
  d = d / np.linalg.norm(d)
  t = np.radians(tilt_deg)
  x = np.array([d[0] * np.cos(t), d[1] * np.cos(t), -np.sin(t)])
  y = sep_axis_sign * np.array([d[1], -d[0], 0.0])
  return ep.R_from_axes(x, y)


# ======================================================================= #
# M1.7: the grasp path, as geometry
# ======================================================================= #
def _palm_up_fwd(v_xz, tilt_deg, zs=1.0):
  """World (up, horizontal-forward) components of a palm-frame (x, z) vector for
  a palm whose finger axis is `tilt_deg` below horizontal.

  `zs = +1` for the curl side whose palm z points up, `-1` for the other one;
  the two are a 180 deg roll apart, so every z component changes sign. Read it
  off the frame with `np.sign(R[2, 2])`; do not pass it by hand.
  """
  t = np.radians(tilt_deg)
  st, ct = np.sin(t), np.cos(t)
  return (-v_xz[0] * st + zs * v_xz[1] * ct,
          v_xz[0] * ct + zs * v_xz[1] * st)


def grasp_geometry(tilt_deg, cyl_z, top_z, zs=1.0):
  """The whole shape of the grasp path, as three lengths.

  Returns `(dz, apex_up, standoff_back)`:
    dz             how far ABOVE the cylinder's centre to put the grasp centre
    apex_up        how far above the grasp pose the apex sits
    standoff_back  how far behind the apex the standoff sits, horizontally

  Every term comes from HAND_TIP / PAIR_TIP and the scene, never from a sweep;
  see the derivation at the top of this file.
  """
  tip_up, tip_fwd = _palm_up_fwd(HAND_TIP[1.0 if zs < 0 else -1.0],
                                 tilt_deg, zs)
  pair_up, _ = _palm_up_fwd(PAIR_TIP, tilt_deg, zs)
  # (a) centre the pinch on the cylinder, (b) keep the hand off the tabletop
  dz = max(-pair_up, (top_z + TABLE_CLEAR - tip_up) - cyl_z)
  # the traverse must carry the deepest part of the hand over the cylinder's top
  apex_up = max(CYL_HALF - dz - tip_up + APEX_MARGIN, APEX_MARGIN)
  # ... and start from behind it, or the rise sweeps the cylinder on the way up
  return float(dz), float(apex_up), float(tip_fwd + CYL_R + STANDOFF_MARGIN)


def grasp_path_targets(cyl, R, tilt_deg, offset, top_z, back_m=None):
  """Palm-site targets for the three poses the grasp flies, plus `dz`.

  standoff -> apex: a horizontal traverse at apex height, ending directly above
  the grasp pose.  apex -> grasp: the vertical descent the cage's open vertical
  corridor is the reason for. DESCENT_DEG < 90 tilts the last segment and pulls
  the apex back by the matching amount, so the shape stays consistent.

  `back_m` overrides `GRASP_BACK_M` for one call. `None` is the constant.
  """
  zs = 1.0 if R[2, 2] >= 0 else -1.0     # which curl side, from the frame
  dz, apex_up, back = grasp_geometry(tilt_deg, float(cyl[2]), top_z, zs)
  back_m = GRASP_BACK_M if back_m is None else back_m
  grasp = np.asarray(cyl, float) + np.array([0, 0, dz]) - R @ offset
  if back_m:
    grasp = grasp - back_m * R[:, 0]
  d = R[:2, 0] / max(float(np.linalg.norm(R[:2, 0])), 1e-9)
  dhat = np.array([d[0], d[1], 0.0])
  apex = grasp + np.array([0.0, 0.0, apex_up])
  if DESCENT_DEG < 89.9:
    apex = apex - (apex_up / np.tan(np.radians(DESCENT_DEG))) * dhat
  standoff = apex - back * dhat + np.array([0.0, 0.0, STANDOFF_UP])
  return standoff, apex, grasp, dz


def _line_over_table(model, p0, p1, top_z, clear, pad=0.0, steps=12):
  """Does the straight palm line `p0 -> p1` pass over the brown tabletop with
  less than `clear` of vertical clearance anywhere along it?

  The footprint is read from the model every call (`brown_table_bounds`), never
  hard-coded: the table can be shifted by `brown_dz`, and pricing geometry
  against a scene the file does not have is the single largest error this
  project has made. `pad` inflates the footprint outward, so a line that skims
  the edge counts as over it.
  """
  x0, x1, y0, y1 = brown_table_bounds(model, margin=-float(pad))
  p0 = np.asarray(p0, float)
  p1 = np.asarray(p1, float)
  ceiling = float(top_z) + float(clear)
  for k in range(steps + 1):
    q = p0 + (k / steps) * (p1 - p0)
    if q[2] < ceiling and x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
      return True
  return False


def pick_azimuth(runner, search, cyl, T):
  """`(verdict, R, tilt, sgn, resid, foul)`: the grasp frame at this stance.

  Wraps the `evaluate` `replan_pick` used to call inline, so the accepted
  candidate and its telemetry are unchanged.

  The azimuth is NOT chosen here. Four pre-commit predictors of "the controller
  will not get there" were tried on the 25 M3.1 capture failures and every one
  failed to separate them from the 251 successes of the two presets:

  | predictor (measured at commit, before any motion) | ncap  | `shipped` succ |
  | `plan_cartesian` truncated                        |  0/25 | 0/35           |
  | plan endpoint residual > 2 mm                     |  0/25 | 0/35           |
  | commanded endpoint penetration > 0                |  0/25 | 0/35           |
  | worst-waypoint penetration > 30 mm                | 24/25 | 21/35 (20 pass)|

  The last row is the trap, and it is also the clue: the straight-line raise
  drags the hand through the tabletop on EVERY episode, 13-48 mm on 35 of 35
  `shipped` episodes that then place the cylinder, so the DEPTH of the drag
  predicts nothing. What matters is whether the drag lasts long enough to walk
  the robot off its stance -- see the note in `fly_grasp_path`.
  """
  az0 = 0.0
  resid, R, tilt, sgn, foul = search.evaluate(search.snapshot(runner.data), cyl,
                                              iters=300, log=True, az_deg=az0)
  T["grasp_candidates"] = list(search.cand_log)
  T["grasp_resid_m"] = round(float(resid), 4)
  T["search_ik_solves"] = search.n_solve
  T["grasp_az_deg"] = az0
  if resid > GRASP_TOL:
    return None, R, tilt, sgn, resid, foul
  return "ok", R, tilt, sgn, resid, foul


def fly_grasp_azimuths(runner, search, cyl, offset, top_z, R_s, tilt, sgn, T,
                       verbose=True):
  """Fly the grasp path and record how far the base walked while it was flown.
  Returns `(R_s, tilt, sgn, G)`.

  `apex_err_m` (the palm error at the apex, after the raise and the traverse)
  separates the two populations completely, over the 300 M3.1 episodes of both
  presets:

  | `apex_err_m`            | `shipped` | `shipped_table` |
  | successes  median / MAX | 1.32 / 1.72 cm | 1.25 / 1.58 cm |
  | non-capture failures    | 1.29 / 1.69 cm | 1.25 / 1.33 cm |
  | `never_captured` median | (none)    | **19.63 cm**    |
  """
  data = runner.data
  T["grasp_az_tried"] = []
  base0 = data.qpos[:2].copy()
  yaw0 = float(ep.base_yaw(data))

  def drift():
    """How far the BASE has walked since the grasp was chosen. The standoff and
    the apex are WORLD points, so a base that walks away carries the whole path
    with it, and the aim-and-correct rounds then chase a world target from a
    body that is still moving."""
    dyaw = float(ep.base_yaw(data)) - yaw0
    return (float(np.linalg.norm(data.qpos[:2] - base0)),
            float(np.degrees(np.arctan2(np.sin(dyaw), np.cos(dyaw)))))

  G = fly_grasp_path(runner, R_s, tilt, offset, top_z, verbose=verbose)
  d_m, d_deg = drift()
  T["pick_base_drift_m"] = round(d_m, 4)
  T["pick_base_drift_deg"] = round(d_deg, 2)
  T["grasp_az_tried"].append({"az": T.get("grasp_az_deg", 0.0),
                              "apex_err_m": G["apex_err_m"],
                              "nudge_m": G["nudge_m"], "tilt": tilt,
                              "drift_m": round(d_m, 4),
                              "drift_deg": round(d_deg, 2)})
  return R_s, tilt, sgn, G


# ======================================================================= #
# M1.5: search for a feasible grasp; if there is none, re-approach the
#       widest feasible BASIN instead of the lowest-residual point.
# ======================================================================= #
def body_offset(data, cyl_xy):
  """Cylinder position in the robot's frame: the only pick geometry the grasp
  actually cares about. A few cm of stance error moves it clean out of the
  graspable envelope, which is why it, not the stance error, is the quantity
  the planner closes the loop on."""
  yaw = ep.base_yaw(data)
  cy, sy = np.cos(yaw), np.sin(yaw)
  v = np.asarray(cyl_xy, float)[:2] - data.qpos[:2]
  return np.array([cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]])


def offset_to_base(cyl_xy, yaw, fwd, lat):
  """Inverse of `body_offset`: the world base xy that would put the cylinder at
  (fwd, lat) in the robot's frame at heading `yaw`."""
  cy, sy = np.cos(yaw), np.sin(yaw)
  return np.asarray(cyl_xy, float)[:2] - np.array(
    [cy * fwd - sy * lat, sy * fwd + cy * lat])


def _grid_margins(ok, dfwd, dlat):
  """Distance from each feasible cell to the nearest infeasible one, in metres.

  This is the number the stance choice is actually made on. Minimizing the IK
  residual is useless here: the residual is EXACTLY ZERO over a ~20 x 36 cm
  slab of offsets (measured), so "the best point" is degenerate, while the
  quantity that decides whether a base controller carrying ~13 cm of error
  arrives somewhere usable is how much room the point has around it.

  The grid is padded with one ring of infeasible cells, so cells on the edge of
  the searched region are rated conservatively; we have not shown that the
  region continues, and at the near edge it demonstrably does not (the table).
  """
  nf, nl = ok.shape
  pad = np.zeros((nf + 2, nl + 2), bool)
  pad[1:-1, 1:-1] = ok
  bad = np.argwhere(~pad).astype(float)
  m = np.full(ok.shape, -1.0)
  idx = np.argwhere(ok)
  for i, j in idx:
    d = np.hypot((bad[:, 0] - (i + 1)) * dfwd, (bad[:, 1] - (j + 1)) * dlat)
    m[i, j] = d.min()
  return m


class GraspSearch:
  """Kinematic feasibility search for the side grasp.

  Pure kinematics throughout: one reusable scratch `MjData`, `mj_kinematics`
  and `ArmIK6.solve` (itself kinematic, on its own scratch), never `mj_step`
  and never a policy. That is what makes a few hundred candidate stances
  affordable inside every episode rather than a training-time luxury.

  Feasibility of a candidate is the worst IK residual over ALL THREE palm poses
  the grasp actually flies: the standoff, the apex above the cylinder and the
  grasp itself (M1.7 `grasp_path_targets`), minimized over the same
  (tilt, curl-side) sweep the executed grasp uses. Scoring fewer poses than the
  controller flies is how M1.5 accepted stances the approach could not be flown
  from; the pose list here and the pose list in `fly_grasp_path` come from the
  same function, so they cannot drift apart.
  """

  # Bodies of the right arm chain. A contact between one of these and anything
  # that is not another one of them (or the cylinder) is a pose the arm cannot
  # physically hold, however clean its IK residual.
  ARM_TOKENS = ("right_shoulder", "right_elbow", "right_wrist", "right_hand")

  def __init__(self, model, ctrl, ik, offset, top_z):
    self.model, self.ik = model, ik
    self.offset, self.top_z = offset, float(top_z)
    self.probe = mujoco.MjData(model)
    self.n_solve = 0
    self.cand_log = []            # M1.9 instrumentation; filled when log=True
    self.map_counts = []          # (pelvis_h, clean, n_feasible) per map built
    self.face_log = []            # M3.1: (heading, cost, margin) per map built
    self.arm_bodies = set()
    for b in range(model.nbody):
      n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
      if any(t in n for t in self.ARM_TOKENS):
        self.arm_bodies.add(b)
    self.cyl_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      "red_block")
    self.table_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in ("table", "table_white")}
    # Finger joints, and the angle the 0.25 pre-curl the approach uses puts
    # them at. The probe must be pre-curled like the real hand or the collision
    # check is answering a question about a different hand: with the fingers
    # fully open the index/middle links reach 2-4 cm below the palm and read as
    # buried in the tabletop at every otherwise-perfect grasp pose.
    self.finger_qadr, self.finger_precurl = [], []
    for act_id, closed_val in ctrl.right_finger_actuators:
      jid = int(model.actuator_trnid[act_id, 0])
      self.finger_qadr.append(int(model.jnt_qposadr[jid]))
      self.finger_precurl.append(GRASP_PRECURL * float(closed_val))

  # ---------------- candidate scoring ----------------
  def snapshot(self, data, cyl_xy=None, yaw=None, fwd=None, lat=None, h=None):
    """The scratch body, posed like `data` but with the hand pre-curled, and
    optionally moved to the candidate base pose that yields offset (fwd, lat).

    Only `qpos[0:7]` changes for a candidate stance: the arm hangs off the
    torso, which hangs off the pelvis's free joint, so a candidate stance is
    exactly a rewrite of the free joint. The waist and leg joints keep their
    measured values; for a crouch that is an approximation (the croucher also
    pitches the waist), which is why a requested crouch is re-verified against
    the real state after it executes rather than trusted from the grid."""
    p = self.probe
    p.qpos[:] = data.qpos
    p.qvel[:] = 0.0
    for adr, v in zip(self.finger_qadr, self.finger_precurl):
      p.qpos[adr] = v
    if cyl_xy is not None:
      p.qpos[0:2] = offset_to_base(cyl_xy, yaw, fwd, lat)
      p.qpos[2] = h
      p.qpos[3] = np.cos(yaw / 2)
      p.qpos[4:6] = 0.0
      p.qpos[6] = np.sin(yaw / 2)
    mujoco.mj_kinematics(self.model, p)
    return p

  def _fouls(self, d, tables, depth=0.002):
    """Does this arm posture drive into the torso (or, if `tables`, a tabletop)?

    The IK is blind to contact, and its residual is a poor predictor of
    achievable palm accuracy on its own: measured, the search accepted grasps
    whose solution buries `right_shoulder_yaw_link` in the torso (executed palm
    error 16.4 cm at a 0.1 mm residual) and one that laid the index finger on
    the brown tabletop (12.5 cm at 0.0 mm). `mj_collision` on the scratch data
    the solve already left forward-kinematic costs 0.02 ms, so the check is
    free next to the solve. Grazes are allowed; only penetration is a veto.

    `tables=False` for the grasp pose itself: it sits 3.7 cm above the tabletop
    by construction and the lower finger legitimately reaches down past the
    cylinder's centre there. The pre-grasp is the pose that must clear the
    table; that is the entire reason the approach is flown from above."""
    mujoco.mj_collision(self.model, d)
    for i in range(d.ncon):
      c = d.contact[i]
      if c.dist > -depth:
        continue
      b1 = int(self.model.geom_bodyid[c.geom1])
      b2 = int(self.model.geom_bodyid[c.geom2])
      if (b1 in self.arm_bodies) == (b2 in self.arm_bodies):
        continue                             # arm-vs-arm, or neither: not ours
      if self.cyl_body in (b1, b2):          # touching the target is the point
        continue
      if not tables and (b1 in self.table_bodies or b2 in self.table_bodies):
        continue
      return True
    return False

  def _frames(self, palm_xy, cyl, az_deg=0.0):
    """(R, tilt, sgn, dz) candidates in the order the executed grasp prefers.

    `az_deg` rotates the palm's in-plane approach direction about world z. The
    cylinder is axisymmetric, so which compass direction the pinch closes along
    is free for the GRASP (every derived length in `grasp_geometry` is a
    function of `tilt` and the curl side, and none of them of the azimuth), but
    it is emphatically not free for the ARM (see the M3.2 note above
    `GRASP_PRELIFT`). Rotating
    `d` by `a` yields exactly `Rz(a) @ R`, because both palm axes are built from
    `d`, so the whole grasp/carry frame chain is rotated by the same world yaw
    and the only downstream term that has to know is the place's carry frame.
    `az_deg = 0.0` reproduces the M3.1 candidate set element for element.

    `dz` is the pinch height `grasp_geometry` will command for this candidate:
    the quantity capture actually tracks (M1.9). With `GRASP_DZ_MAX` the
    candidates above the
    measured capture ceiling are dropped, so a stance that offers only those
    reads as INFEASIBLE and the stance search gets the problem instead.
    """
    d = cyl[:2] - np.asarray(palm_xy, float)
    n = np.linalg.norm(d)
    d = np.array([1.0, 0.0]) if n < 1e-9 else d / n
    if az_deg:
      a = np.radians(float(az_deg))
      ca, sa = np.cos(a), np.sin(a)
      d = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
    out = []
    for tilt in GRASP_TILTS:
      t = np.radians(tilt)
      x = np.array([d[0] * np.cos(t), d[1] * np.cos(t), -np.sin(t)])
      for sgn in GRASP_SIGNS:
        y = np.array([sgn * d[1], -sgn * d[0], 0.0])
        R = ep.R_from_axes(x, y)
        zs = 1.0 if R[2, 2] >= 0 else -1.0
        dz = grasp_geometry(tilt, float(cyl[2]), self.top_z, zs)[0]
        out.append((R, tilt, sgn, float(dz)))
    if GRASP_DZ_MAX is not None:
      # The ceiling is a statement about where the PINCH lands, not about `dz`
      # as a number: `GRASP_BACK_M` moves the palm back along a finger axis that
      # is `tilt` below horizontal, which lifts the pinch by back*sin(tilt):
      # 2.6 mm at 15 deg, 4.2 mm at 25. Charging that to the ceiling keeps M1.9's
      # invariant (the pinch below the cylinder's top face) exactly, instead of
      # letting a 1 cm aim bias smuggle the tallest candidate over it.
      out = [c for c in out
             if c[3] + GRASP_BACK_M * np.sin(np.radians(c[1]))
             <= GRASP_DZ_MAX + 1e-9]
    return out

  def _path_fouls(self, q_a, q_b, tables, steps=PATH_STEPS):
    """Foul check along the JOINT-SPACE segment q_a -> q_b.

    Since M1.7 the controller flies exactly these segments as planned Cartesian
    paths (`CART_SKIP_TAGS` is empty), and the executed deviation from the
    straight line between two consecutive waypoints is ~1 cm, so interpolating
    them is now the RIGHT model rather than a stand-in for an arc; the mismatch
    M1.6 recorded here is gone. It stays a preference, not a veto (see
    `evaluate`). Measured twice, before the paths were explicit: a raise whose arc jammed the
    index finger on the brown tabletop (palm stalled 12.5 cm short, the
    actuators kept pushing and the robot went over), and an approach whose arc
    carried the palm 10 cm ABOVE the cylinder and brought it down on top,
    sweeping it 78 cm off the table. Both had a 0.0 mm residual at both
    endpoints. Checking the endpoints alone is simply the wrong feasibility
    question for a joint-space controller."""
    d = self.ik.scratch
    for k in range(1, steps):
      f = k / steps
      d.qpos[self.ik.qpos_idx] = q_a + f * (q_b - q_a)
      mujoco.mj_kinematics(self.model, d)
      if self._fouls(d, tables=tables):
        return True
    return False

  def _resid(self, d, cyl, R, tilt, iters):
    """Worst IK residual over the three palm poses the grasp flies, plus whether
    any of them (or any segment between them) fouls.

    The poses are solved in the order the controller flies them in reverse
    (grasp, then apex seeded from it, then standoff seeded from that), because a
    DLS residual is a property of the seed (M1.6) and consecutive poses are 7-11
    cm apart, so a seeded solve converges in a few iterations.
    """
    q0 = d.qpos[self.ik.qpos_idx].copy()
    p_s, p_a, p_g, _ = grasp_path_targets(cyl, R, tilt, self.offset, self.top_z)
    # solve_ms, not solve: M1.6 measured that a DLS residual is a property of
    # the SEED and put multi-start in the controller, but left the search on the
    # single-seed solve, which reports 585-676 mm at the close cells (offset
    # fwd 0.26) that the same arm reaches to 0.0 mm from another posture, i.e.
    # the search was calling reachable geometry infeasible. Retries fire only
    # when the first solve misses `GRASP_TOL`, so a feasible cell costs the same.
    q_g, rp_g, rr = self.ik.solve_ms(d, p_g, R, iters=iters, tol=GRASP_TOL)
    # The grasp pose is no longer exempt from the tabletop check: M1.7 commands
    # a height that clears the top by TABLE_CLEAR, so a foul there is a real
    # foul rather than the "3.7 cm above the top by construction" the old
    # exemption assumed (measured: 3-7 cm INSIDE the top at every feasible cell).
    foul = self._fouls(self.ik.scratch, tables=True)
    q_a, rp_a, _ = self.ik.solve_ms(d, p_a, R, iters=iters, seed=q_g,
                                    tol=GRASP_TOL)
    foul = foul or self._fouls(self.ik.scratch, tables=True)
    q_p, rp_s, _ = self.ik.solve_ms(d, p_s, R, iters=iters, seed=q_a,
                                    tol=GRASP_TOL)
    foul = foul or self._fouls(self.ik.scratch, tables=True)
    self.n_solve += 3
    if not foul:
      # The two short segments are modelled by interpolating joints between
      # their endpoints, which is what the controller's own waypoints do over
      # 7-11 cm. The RAISE is different: it is a ~25 cm move with a large
      # orientation change, and interpolating IT in joint space is the wrong
      # model now that it is flown as a straight palm line: measured, at the
      # nominal stance the joint interpolant fouls the torso for EVERY
      # candidate, which left `evaluate` with nothing clean and made it fall
      # back on a genuinely fouling pose (executed standoff error 12.9 cm,
      # cylinder batted). So the raise is checked the way it is flown: at its
      # palm-space midpoint, solved from the same chain the controller uses.
      p_mid, R_mid = ep.cartesian_waypoints(
        d.site_xpos[self.ik.site_id].copy(),
        d.site_xmat[self.ik.site_id].reshape(3, 3).copy(),
        p_s, R, n=2)[0]
      self.ik.solve(d, p_mid, R_mid, iters=iters, seed=q0)
      self.n_solve += 1
      foul = (self._fouls(self.ik.scratch, tables=True)
              or self._path_fouls(q_p, q_a, tables=True)
              or self._path_fouls(q_a, q_g, tables=True))
    return max(rp_g, rp_a, rp_s), rr, foul
  def evaluate(self, d, cyl, iters=300, tol=GRASP_TOL, log=False, az_deg=0.0):
    """Best (resid, R, tilt, sgn, fouls) for the grasp from the pose in `d`.

    Fouling is a PREFERENCE, not a veto. A clean candidate inside `tol` wins and
    ends the sweep immediately; otherwise the cleanest candidate inside `tol`
    wins; only if nothing is inside `tol` does the lowest residual come back.

    Making it a veto was measured and rejected: it removed the fall and pushed
    three more episodes through to `lifted`, but it also turned two genuinely
    reachable geometries into `out_of_reach`, because the foul model is a
    heuristic (it interpolates the ramp linearly, while `ramp_to` moves every
    joint at the same rate, so short-travel joints arrive early and the real
    path bends away from the modelled one). A heuristic should not be allowed
    to manufacture the failure mode this milestone exists to remove; it should
    break ties, which is what it does now.

    The sweep is ordered by preference, not by score, and the executed grasp is
    tuned for the early entries, so an early exit is a feature."""
    palm = d.site_xpos[self.ik.site_id]
    best_any = best_clean = None
    log = self.cand_log if log else None
    if log is not None:
      log.clear()
    for R, tilt, sgn, dz in self._frames(palm[:2], cyl, az_deg):
      m, rr, foul = self._resid(d, cyl, R, tilt, iters)
      cand = (m, R, tilt, sgn, foul)
      if log is not None:
        log.append({"tilt": int(tilt), "sgn": float(sgn),
                    "dz_m": round(float(dz), 4),
                    "resid_m": round(float(m), 4),
                    "rot_resid_deg": round(float(rr), 2), "foul": bool(foul)})
      if best_any is None or m < best_any[0]:
        best_any = cand
      if not foul:
        if best_clean is None or m < best_clean[0]:
          best_clean = cand
        if m <= tol and rr < 5.0:
          if log is not None:
            log[-1]["chosen"] = "early_clean"
          return cand
    if best_any is None:                    # GRASP_DZ_MAX admitted nothing
      return 9.99, np.eye(3), GRASP_TILTS[0], GRASP_SIGNS[0], True
    if best_clean is not None and best_clean[0] <= tol:
      if log is not None:
        for row in log:
          if (row["tilt"], row["sgn"]) == (best_clean[2], best_clean[3]):
            row["chosen"] = "best_clean"
      return best_clean
    if log is not None:
      for row in log:
        if (row["tilt"], row["sgn"]) == (best_any[2], best_any[3]):
          row["chosen"] = "best_any"
    return best_any

  # ---------------- hypothetical stances ----------------
  def stance_feasible(self, data, cyl, yaw, fwd, lat, h, clean,
                      iters=SEARCH_ITERS):
    """Can this candidate stance grasp the cylinder? `clean` also demands that
    the approach path be predicted foul-free."""
    p = self.snapshot(data, cyl[:2], yaw, fwd, lat, h)
    resid, _, _, _, foul = self.evaluate(p, cyl, iters=iters)
    return resid <= GRASP_TOL and not (clean and foul)

  # ---------------- the basin search ----------------
  def basin(self, data, cyl, yaw, blocked, stand_h, heights=None, need=None,
            geo=None):
    """Widest-margin stance for grasping `cyl`, over offsets AND HEADINGS.

    M3.1. The heading is tried in one order and one order only: the robot's
    CURRENT heading first, and if that yields any stance at all it is returned
    unchanged, so every plan the incumbent pipeline made, it still makes, bit
    for bit (`_basin_at_yaw` is M1.14's `basin` verbatim). The heading sweep
    runs only where the incumbent aborted: 28 of the 29 `out_of_reach` episodes
    on `shipped_table` seed 0 are this function returning None on its first
    call, before a single corrective leg is spent.

    `APPROACH_SEARCH = False` restores the M1.14 behaviour exactly.

    `need` is read at CALL time. As a default argument it was bound before
    `_apply_env_overrides` could touch it, the same shape of bug that made
    `MAX_PICK_LEGS` a silent no-op for a dozen milestones (see `replan_pick`).
    """
    need = BASIN_NEED_M if need is None else float(need)
    plan = self._basin_at_yaw(data, cyl, yaw, blocked, stand_h, heights, need)
    if plan is not None:
      plan["yaw"] = float(yaw)
      plan["face_change"] = False
    if not APPROACH_SEARCH:
      return plan
    if plan is not None and plan["margin_m"] >= APPROACH_MIN_MARGIN_M:
      return plan                       # a basin the gait can actually land in
    alt = self.approach(data, cyl, yaw, blocked, stand_h, heights, need, geo)
    if alt is None:
      return plan                       # nothing better anywhere: keep it
    if plan is None or alt["margin_m"] > plan["margin_m"]:
      return alt
    return plan

  def approach(self, data, cyl, yaw, blocked, stand_h, heights, need, geo):
    """Choose an approach HEADING, then the widest basin at that heading.

    Candidate headings are ranked by what they COST (the routed walk to a
    representative stance plus `APPROACH_TURN_M` per 90 deg of turning), not by
    what they score, because the scores are nearly flat (ten of twelve
    tabletop cells offer 10-12.5 cm from some heading) while the walks differ
    by metres. Headings whose offset grid is entirely inside a tabletop
    footprint are dropped for free before any IK runs, and at most
    `APPROACH_MAX_MAPS` feasibility maps are built; the sweep stops early on
    the first heading worth `APPROACH_NEED_M`.
    """
    here = data.qpos[:2].copy()
    ranked = []
    for deg in APPROACH_YAWS_DEG:
      y = float(np.radians(float(deg)))
      free = [offset_to_base(cyl[:2], y, f, l)
              for f in STANCE_FWD for l in STANCE_LAT
              if not blocked(offset_to_base(cyl[:2], y, f, l))]
      if len(free) < APPROACH_MIN_FREE:
        continue
      rep = np.mean(np.asarray(free), axis=0)
      dyaw = abs(np.arctan2(np.sin(y - yaw), np.cos(y - yaw)))
      cost = (nav_len(here, rep, geo) if geo is not None
              else float(np.linalg.norm(rep - here)))
      cost = float(cost + APPROACH_TURN_M * dyaw / (np.pi / 2))
      ranked.append((round(cost, 4), int(deg), y, len(free)))
    ranked.sort()
    self.face_log = []
    best = None
    for cost, deg, y, nfree in ranked[:max(int(APPROACH_MAX_MAPS), 0)]:
      plan = self._basin_at_yaw(data, cyl, y, blocked, stand_h, heights, need)
      m = None if plan is None else plan["margin_m"]
      self.face_log.append({"yaw_deg": deg, "cost_m": cost, "n_free": nfree,
                            "margin_m": m})
      if plan is None:
        continue
      plan["yaw"] = y
      plan["face_change"] = True
      plan["route_cost_m"] = cost
      if best is None or plan["margin_m"] > best["margin_m"]:
        best = plan
      if plan["margin_m"] >= APPROACH_NEED_M:
        break
    return best

  def _basin_at_yaw(self, data, cyl, yaw, blocked, stand_h,
                    heights=None, need=BASIN_NEED_M):
    """Widest-margin (fwd, lat, pelvis height) stance for grasping `cyl`.

    Crouch heights are a LAST RESORT, tried only when standing has no feasible
    cell at all, not merely when standing misses `need`. Measured reasons, both
    against the design's expectation that crouch is a first-class pick DoF:

      * the croucher drifts the base while it settles, by up to 13 cm of
        cylinder-in-body offset (0.389,-0.188 -> 0.369,-0.321 in one episode),
        which invalidates the very offset the crouch was planned for;
      * the raised-arm approach exists to clear the tabletop, and it stops
        clearing it when the pelvis drops: at h=0.65 the palm settled 2 cm
        above the brown top with `table<->right_hand_index_0_link` in contact,
        versus 12.6 cm of clearance standing.

    Crouch stays load-bearing for the PLACE (the blue table at 0.633 genuinely
    is out of standing reach); for the PICK the brown table at 0.733 is already
    in the good band, and the ~4 cm of extra forward reach the kinematic map
    credits it with is not worth either cost.

    Pelvis height is the OUTER loop and path-cleanliness the inner one, in
    that order for a measured reason: insisting on a clean approach path admits
    only 1-14 standing cells, and if cleanliness ranked above height that
    scarcity alone tipped the search into a crouch, whose base drift then broke
    the very offset it had planned (both remaining `out_of_reach` episodes, once
    each). So: try standing, prefer a clean path there, accept a fouling one
    rather than leave standing, and only drop the pelvis if standing has
    nothing at all.

    Returns a dict, or None if nothing anywhere is feasible AT THIS HEADING,
    which is where `basin` takes over and sweeps the heading (M3.1).
    """
    dfwd = float(STANCE_FWD[1] - STANCE_FWD[0])
    dlat = float(STANCE_LAT[1] - STANCE_LAT[0])
    heights = [stand_h] + list(PICK_CROUCH_HEIGHTS if heights is None
                               else heights)
    fallback = None
    passes = (True, False)
    for hi, h in enumerate(heights):
      found = []
      for clean in passes:
        cand = self._map_at(data, cyl, yaw, blocked, h, clean, dfwd, dlat)
        if cand is None:
          continue
        cand["crouch_h"] = None if hi == 0 else float(h)
        found.append(cand)
        if fallback is None:
          fallback = cand
        if hi == 0:
          # M1.7 REGRESSION, FIXED HERE. Three-pose feasibility narrowed the
          # clean standing basin to 1-4 cells on `spec`, and this returned the
          # first standing map with ANY cells, so it preferred a 1-cell clean
          # point over a wide path-fouling one. The gait carries ~13 cm of error,
          # so a 1-cell basin is a stance it will miss; the episode spent its
          # corrective legs and ended `out_of_reach` (6 -> 8 on `spec` seed 0).
          #
          # The repair is deliberately NARROW: only a clean map that is not a
          # basin at all falls through. Falling through whenever the clean map
          # merely misses `need` was also measured, and it buys `tipped_at_place`
          # 5/30 -> 2/30 on `reachable` by converting three episodes that reached
          # the place into `never_captured` (14 -> 18): a path-fouling stance is
          # a worse pick, which is the whole reason cleanliness is a preference.
          # That is not a gain, it is the failure moving upstream.
          if clean and cand["margin_m"] < BASIN_CLEAN_FLOOR_M:
            continue        # not a basin; see what the relaxed map offers
          return cand       # standing works, and its basin is usable
        if cand["margin_m"] >= need:
          return cand
        break               # this height has cells; no need for the relaxed pass
      if found:
        if hi == 0:
          return max(found, key=lambda c: c["margin_m"])
        break
    return fallback

  def _map_at(self, data, cyl, yaw, blocked, h, clean, dfwd, dlat):
    """Feasibility map over the offset grid at one pelvis height; widest cell."""
    ok = np.zeros((len(STANCE_FWD), len(STANCE_LAT)), bool)
    for j, lat in enumerate(STANCE_LAT):
      seen = False
      for i, fwd in enumerate(STANCE_FWD):
        _check_search_deadline()
        if blocked(offset_to_base(cyl[:2], yaw, fwd, lat)):
          continue
        if self.stance_feasible(data, cyl, yaw, fwd, lat, h, clean):
          ok[i, j] = seen = True
        elif seen:
          break   # measured: past the near edge the residual only grows
    self.map_counts.append((round(float(h), 3), bool(clean), int(ok.sum())))
    if not ok.any():
      return None
    m = _grid_margins(ok, dfwd, dlat)
    i, j = np.unravel_index(np.argmax(m), m.shape)
    return {"fwd": float(STANCE_FWD[i]), "lat": float(STANCE_LAT[j]),
            "margin_m": round(float(m[i, j]), 4),
            "n_feasible": int(ok.sum()), "clean": bool(clean)}


def _table_blocker(geo, clear=STANCE_CLEAR_M):
  """`blocked(xy)` -> the base cannot stand there without fouling a table.

  Both tabletops, inflated by `clear`. Without this the basin search happily
  proposes standing on the brown table for a cylinder near its far edge."""
  boxes = [(geo["brown_center"][:2], geo["brown_half"][:2]),
           (geo["blue_center"][:2], geo["blue_half"][:2])]

  def blocked(xy):
    for c, hh in boxes:
      if (abs(xy[0] - c[0]) < hh[0] + clear
          and abs(xy[1] - c[1]) < hh[1] + clear):
        return True
    return False
  return blocked


# ======================================================================= #
# M3.1: routing around the outside of the two tables
# ======================================================================= #
def _nav_boxes(geo, clear=None):
  """Both tabletop footprints as (centre_xy, half_xy), inflated by `clear`."""
  clear = NAV_CLEAR_M if clear is None else float(clear)
  return [(np.asarray(geo["brown_center"][:2], float),
           np.asarray(geo["brown_half"][:2], float) + clear),
          (np.asarray(geo["blue_center"][:2], float),
           np.asarray(geo["blue_half"][:2], float) + clear)]


def _in_box(xy, box):
  c, hh = box
  return bool(np.all(np.abs(np.asarray(xy, float)[:2] - c) < hh))


def _seg_hits_box(a, b, box, eps=1e-4):
  """Does the segment a->b pass through the box's INTERIOR? (slab clipping)

  Grazing along a face is allowed; the goal of a route leg is frequently a
  stance that sits exactly on the inflated boundary.
  """
  c, hh = box
  lo, hi = c - hh, c + hh
  a = np.asarray(a, float)[:2]
  d = np.asarray(b, float)[:2] - a
  t0, t1 = 0.0, 1.0
  for k in range(2):
    if abs(d[k]) < 1e-12:
      if a[k] <= lo[k] or a[k] >= hi[k]:
        return False                 # parallel to this slab and outside it
      continue
    ta, tb = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
    if ta > tb:
      ta, tb = tb, ta
    t0, t1 = max(t0, ta), min(t1, tb)
    if t0 >= t1 - eps:
      return False
  return True


def _nav_nodes(boxes, pad=None):
  pad = NAV_NODE_PAD_M if pad is None else float(pad)
  """Corner waypoints: each inflated box's corners, pushed `pad` further out
  and dropped if they fall inside the OTHER box (the two overlap in the gap)."""
  nodes = []
  for c, hh in boxes:
    for sx in (-1.0, 1.0):
      for sy in (-1.0, 1.0):
        p = c + np.array([sx * (hh[0] + pad), sy * (hh[1] + pad)])
        if not any(_in_box(p, b) for b in boxes):
          nodes.append(p)
  return nodes


def nav_path(start_xy, goal_xy, geo, clear=None):
  """Waypoints (goal last, start excluded) whose straight legs stay outside
  both inflated tabletop footprints.

  A visibility graph over the boxes' corners plus Dijkstra: ten nodes, so it
  costs nothing and it is exact for two convex obstacles. It returns `[goal]`
  whenever the straight segment is already clear, which is why every route the
  incumbent pipeline flies is left untouched.
  """
  boxes = _nav_boxes(geo, clear)
  a = np.asarray(start_xy, float)[:2].copy()
  b = np.asarray(goal_xy, float)[:2].copy()
  # A point inside an inflated box ignores THAT box: a spawn, a settle or a
  # validated stance (which stands `STANCE_CLEAR_M` out, not `NAV_CLEAR_M`) can
  # legitimately be inside it, and otherwise nothing would be reachable.
  inside = [{i for i, box in enumerate(boxes) if _in_box(p, box)} for p in (a, b)]
  pts = [a] + _nav_nodes(boxes) + [b]
  skip = [inside[0]] + [set()] * (len(pts) - 2) + [inside[1]]

  def clearway(i, j):
    ign = skip[i] | skip[j]
    return not any(_seg_hits_box(pts[i], pts[j], box)
                   for k, box in enumerate(boxes) if k not in ign)

  n = len(pts)
  if clearway(0, n - 1):
    return [b]
  INF = float("inf")
  dist = [INF] * n
  prev = [-1] * n
  dist[0] = 0.0
  done = [False] * n
  for _ in range(n):
    u = min((d, i) for i, d in enumerate(dist) if not done[i])[1]
    if dist[u] == INF:
      break
    done[u] = True
    if u == n - 1:
      break
    for v in range(n):
      if done[v] or not clearway(u, v):
        continue
      d = dist[u] + float(np.linalg.norm(pts[v] - pts[u]))
      if d < dist[v]:
        dist[v], prev[v] = d, u
  if dist[n - 1] == INF:
    return [b]                       # no route: fly it straight and let the
  out = []                           # gait report the error honestly
  k = n - 1
  while k > 0:
    out.append(pts[k])
    k = prev[k]
  return out[::-1]


def nav_len(start_xy, goal_xy, geo, clear=None):
  """Length of `nav_path`, for ranking candidate approach headings."""
  p = np.asarray(start_xy, float)[:2]
  total = 0.0
  for q in nav_path(start_xy, goal_xy, geo, clear):
    total += float(np.linalg.norm(q - p))
    p = q
  return total


def route_to_pose(runner, goal_xy, goal_yaw, geo, tag="route",
                  budget_s=None, stage_m=None):
  """Walk to (goal_xy, goal_yaw) around the outside of both tables.

  A sequence of STRAIGHT legs with rotator bursts between them, never an arc
  and never a strafe. That shape is forced by two measurements: the walker
  crabs (0.30 m in 21.5 s for a leg with a lateral component, against 0.5 m in
  1.3-2.4 s straight ahead), and it is dead for in-place yaw, so the rotator is
  the only tool for a turn and survives only as a burst (it falls on velocity
  commands and drifts 8-35 cm per spin).

  The route is RE-PLANNED from the pose actually reached before every leg, not
  flown as a fixed waypoint list. That is not a refinement: a leg lands up to
  0.20 m short of its waypoint and the rotator then drifts another 8-35 cm, so
  a fixed list has the next leg starting from somewhere the plan never
  considered: measured, three routes cut the brown table's north-west corner
  and jammed against its long edge for 80 s.

  The last leg is the docking pattern `corrective_leg` and `walk_to_pose`
  already use: stage `NAV_STAGE_M` back along the FINAL heading, spend the
  rotation there where drift is harmless, then one straight leg in.
  """
  data = runner.data
  goal_xy = np.asarray(goal_xy, float)[:2]
  stage_m = NAV_STAGE_M if stage_m is None else float(stage_m)
  budget_s = NAV_BUDGET_S if budget_s is None else float(budget_s)
  t0 = float(data.time)
  log = []
  saved_kicks, saved_mode = ep.WALK_STALL_KICKS, ep.WALK_STALL_MODE
  ep.WALK_STALL_KICKS, ep.WALK_STALL_MODE = NAV_STALL_KICKS, NAV_STALL_MODE
  n_stalls0 = len(ep.STALL_LOG)
  try:
    _route_legs(runner, goal_xy, goal_yaw, geo, log, t0, budget_s, stage_m, tag)
  finally:
    ep.WALK_STALL_KICKS, ep.WALK_STALL_MODE = saved_kicks, saved_mode
  err = float(np.linalg.norm(goal_xy - data.qpos[:2]))
  yaw_e = float(np.degrees(abs(base.yaw_err_to(data, goal_yaw))))
  print(f"  [{tag}] {len(log)} leg(s), "
        f"{sum(1 for r in log if r.get('turned'))} turn(s), "
        f"{len(ep.STALL_LOG) - n_stalls0} stall(s), "
        f"{float(data.time) - t0:.1f} s sim -> pos err {err * 100:.1f} cm, "
        f"yaw err {yaw_e:.1f} deg", flush=True)
  return err, yaw_e, log


def _route_leg(runner, wp, walk_yaw, tol, t0, tag, k, kind):
  """One straight leg: turn onto the heading, then walk it."""
  data = runner.data
  here = data.qpos[:2].copy()
  row = {"leg": k, "kind": kind,
         "goal": [round(float(v), 3) for v in wp],
         "yaw_deg": round(float(np.degrees(walk_yaw)), 1),
         "dist_m": round(float(np.linalg.norm(np.asarray(wp) - here)), 3),
         "turned": False}
  if abs(base.yaw_err_to(data, walk_yaw)) > np.radians(NAV_TURN_DEG):
    row["turned"] = True
    row["turn_deg"] = round(float(np.degrees(base.yaw_err_to(data, walk_yaw))), 1)
    row["turn_err_deg"] = round(base.rotate_to(runner, walk_yaw), 1)
    # Hand the locomotion slot back the way `set_crouch` does. `last_action`
    # is a term of the walker's own observation, and it comes back holding
    # the ROTATOR's last output; on the croucher handoff that mattered enough
    # to be commented as "what makes the transition safe", and a route spends
    # three or four bursts where the incumbent pipeline spent one. Measured:
    # without it the first routed reface fell the robot (pelvis 0.567 m,
    # `up` past 60 deg) 25 s in; with it the same route lands 7.4 cm out.
    if NAV_HANDOFF_RESET:
      runner.ctrl.last_action[:] = 0.0
  d = np.asarray(wp) - data.qpos[:2]
  dist = float(np.linalg.norm(d))
  # A leg is only as long as it is; the walker holds ~0.4 m/s and `walk_to`
  # adds a 1.5 s settle, so the timeout tracks the distance instead of
  # spending 20 s on every 30 cm hop. 0.15 m/s, not 0.4: a leg that spends
  # its first two seconds in a stall still has to be able to finish.
  timeout = float(np.clip(6.0 + dist / 0.15, 8.0, 25.0))
  row["err_m"] = round(ep.walk_to(runner, wp, goal_yaw=walk_yaw,
                                  timeout=timeout, tol=tol), 3)
  row["sim_s"] = round(float(data.time) - t0, 1)
  row["at"] = [round(float(v), 3) for v in data.qpos[:2]]
  row["moved_m"] = round(float(np.linalg.norm(data.qpos[:2] - here)), 3)
  row["pelvis_z"] = round(float(data.qpos[2]), 3)
  print(f"    [{tag} {k}] {row}", flush=True)
  return row


def _route_legs(runner, goal_xy, goal_yaw, geo, log, t0, budget_s, stage_m, tag):
  """Fly the route, replanning from the measured pose before every leg.

  Split out of `route_to_pose` so the stall-kick arming there is restored on
  every exit, including an `EpisodeAborted` from the monitor.
  """
  data = runner.data
  head = np.array([np.cos(goal_yaw), np.sin(goal_yaw)])
  stage = goal_xy - stage_m * head
  # A staging point under a table cannot be stood on; dock from wherever the
  # router can legitimately stand instead.
  boxes = _nav_boxes(geo)
  target = goal_xy if any(_in_box(stage, b) for b in boxes) else stage
  dead = 0
  for k in range(max(int(NAV_MAX_LEGS), 1)):
    if float(data.time) - t0 > budget_s:
      log.append({"leg": k, "skipped": "budget"})
      break
    here = data.qpos[:2].copy()
    # Stop transiting once the remainder is inside the walker's own deadband:
    # `walk_to` floors out at 7-9 cm and adds a 1.5 s settle, so a 15 cm hop
    # costs 7 s of sim to buy 10 cm, and the dock leg re-aims from wherever we
    # are anyway. Sim time is the budget that turns `out_of_reach` into
    # `timeout`, so it is not free.
    if float(np.linalg.norm(target - here)) <= 2.0 * NAV_LEG_TOL_M:
      break
    path = nav_path(here, target, geo)
    wp = path[0]
    d = wp - here
    if float(np.linalg.norm(d)) < 0.05:
      break
    walk_yaw = float(np.arctan2(d[1], d[0]))
    row = _route_leg(runner, wp, walk_yaw, NAV_LEG_TOL_M, t0, tag, k, "transit")
    log.append(row)
    # Two legs in a row that go nowhere is the dead attractor with its
    # recovery already spent; more of the same only converts `out_of_reach`
    # into `timeout`.
    dead = dead + 1 if row["moved_m"] < 0.05 else 0
    if dead >= 2:
      row["gave_up"] = "no progress"
      break
  # dock: spend the rotation where drift is harmless, then one straight leg in
  if float(data.time) - t0 <= budget_s:
    log.append(_route_leg(runner, goal_xy, float(goal_yaw), 0.07, t0, tag,
                          len(log), "dock"))


def _pose_route_clear(runner, goal_xy, goal_yaw, geo):
  """Would `base.walk_to_pose` reach this pose without walking into a table?

  It replicates `walk_to_pose`'s own branch choice rather than guessing, so
  that "the straight route is clear" means exactly "the legs walk_to_pose would
  fly are clear", and when it is true the incumbent call is made unchanged.
  """
  data = runner.data
  goal_xy = np.asarray(goal_xy, float)[:2]
  boxes = _nav_boxes(geo)
  here = data.qpos[:2].copy()
  dyaw = abs(np.degrees(base.yaw_err_to(data, goal_yaw)))
  dist = float(np.linalg.norm(goal_xy - here))
  if dyaw < 30.0 and dist > 0.5:
    segs = [(here, goal_xy)]
  else:
    heading = np.array([np.cos(goal_yaw), np.sin(goal_yaw)])
    stage = goal_xy - 0.60 * heading
    segs = [(here, stage), (stage, goal_xy)]
  ign = {i for i, b in enumerate(boxes) if _in_box(here, b)}
  ign |= {i for i, b in enumerate(boxes) if _in_box(goal_xy, b)}
  for a, b in segs:
    for i, box in enumerate(boxes):
      if i not in ign and _seg_hits_box(a, b, box):
        return False
  return True


def walk_to_pose_around(runner, goal_xy, goal_yaw, geo, tag="route"):
  """`base.walk_to_pose`, but routed around the tables when it has to be.

  When the straight route is clear (every leg the incumbent pipeline flies,
  including the whole `shipped` preset), this IS `base.walk_to_pose`, called with
  the same arguments. Only a pose on the far side of a table takes the router.
  """
  if not APPROACH_SEARCH or _pose_route_clear(runner, goal_xy, goal_yaw, geo):
    return base.walk_to_pose(runner, goal_xy, goal_yaw)
  err, yaw_e, _ = route_to_pose(runner, goal_xy, goal_yaw, geo, tag=tag)
  return err, yaw_e


def corrective_leg(runner, base_xy, yaw):
  """Walk back and re-approach: the only correction the gait can execute.

  A 5-10 cm nudge is physically unavailable: the walker deadbands below
  ~0.25-0.3 m/s, so short hops step in place and move nothing. So the
  correction is spent as a retreat along the approach heading followed by one
  straight leg back in, the same docking pattern the place-pose retry already
  had to use. The rotator is deliberately not involved: it drifts 8-35 cm per
  burst and this is a translation fix, not a heading one.

  The staging point is placed **on the approach line** (`base_xy - L*head`),
  not simply behind the current pose, and `L` is stretched so the retreat is
  still a real leg. That detail is load-bearing: staging off the line leaves a
  lateral component in the final leg, and a leg that is 15 deg off the heading
  makes the walker crab: measured 0.30 m of progress in 21.5 s, versus 0.5 m
  in 1.3-2.4 s for the same command straight ahead. The lateral part of the
  correction is spent on the retreat instead, where accuracy does not matter.
  """
  head = np.array([np.cos(yaw), np.sin(yaw)])
  here = runner.data.qpos[:2].copy()
  base_xy = np.asarray(base_xy, float)
  ahead = float((here - base_xy) @ head)      # >0: already past the stance
  L = max(RETREAT_M, MIN_RETREAT_M - ahead)   # keep BOTH legs long enough
  stage = base_xy - L * head
  t0 = float(runner.data.time)
  e_out = ep.walk_to(runner, stage, goal_yaw=yaw, timeout=20.0)
  t1 = float(runner.data.time)
  at_stage = runner.data.qpos[:2].copy()
  e_in = ep.walk_to(runner, base_xy, goal_yaw=yaw, timeout=20.0)
  detail = {
    "L": round(float(L), 4), "ahead": round(ahead, 4),
    "retreat_err_m": round(float(e_out), 4),
    "retreat_s": round(t1 - t0, 2),
    "retreat_along": round(float((at_stage - stage) @ head), 4),
    "return_err_m": round(float(e_in), 4),
    "return_s": round(float(runner.data.time) - t1, 2),
    "return_len_m": round(float(np.linalg.norm(base_xy - at_stage)), 4),
  }
  return float(e_in), detail


def replan_pick(runner, search, geo, T, max_legs=None):
  """Get into a stance the side grasp is actually feasible from.

  Called only when the grasp the pipeline used to assume is not available. Each
  round re-measures the cylinder from the pose the robot really reached, so
  nothing here is committed before it is verified; the budget is `max_legs`
  corrective legs, after which the episode ends as a genuine `out_of_reach`
  rather than flailing.

  Two kinds of correction, and `basin` decides which. A stance at the CURRENT
  heading is a corrective leg (retreat along the approach line, one straight leg
  back in). A stance at another heading is a face change: M3.1 lets the search
  choose the approach SIDE, which is what "a cylinder 68 cm out needs a
  different approach side" was waiting for, and getting there is a route around
  the outside of both tables. A face change spends one iteration of this loop
  but is counted separately in `T["face_changes"]`, so `pick_legs` stays
  comparable with M1.14's numbers.
  """
  # Read the constant at CALL time, not at `def` time. A default argument is
  # bound when the module is defined, which is before `_apply_env_overrides`
  # runs -- so `G1_SET='e2.MAX_PICK_LEGS=4'` silently did nothing (measured: a
  # 4-leg sweep that spent at most 2 legs and reproduced the baseline exactly).
  max_legs = MAX_PICK_LEGS if max_legs is None else max_legs
  data = runner.data
  blocked = _table_blocker(geo)
  stand_h = float(data.qpos[2])
  T["pick_legs"] = 0
  T["pick_crouch_h"] = None
  T["pick_leg_log"] = []
  T["pick_offsets"] = []
  T.setdefault("face_changes", 0)
  T.setdefault("face_log", [])
  T.setdefault("approach_yaw_deg", None)
  T.setdefault("route_log", [])
  if REFACE_NEAR_YAW_DEG:
    T.setdefault("near_refaces", 0)
  faces = n_legs = n_near = n_reface = 0
  for leg in range(max_legs + 1):
    cyl = runner.cyl_pos()
    off = body_offset(data, cyl)
    T["pick_offsets"].append([round(float(v), 4) for v in off])
    # M3.2: `pick_azimuth` wraps the `evaluate` this used to call inline. It
    # returns the same candidate whenever the controller can fly its path, so
    # the accepted grasp and the telemetry are unchanged on every episode whose
    # plan is flyable: 149 of 150 on `shipped`.
    verdict, R, tilt, sgn, resid, foul = pick_azimuth(runner, search, cyl, T)
    print(f"  [replan {leg}] offset {np.round(off, 3).tolist()} pelvis z "
          f"{float(data.qpos[2]):.3f} best resid {resid * 1000:5.1f} mm "
          f"(tilt {tilt}, sgn {sgn:+.0f}, "
          f"path {'fouls' if foul else 'clear'}, az "
          f"{T.get('grasp_az_deg', 0.0):+.0f} deg) -> "
          f"{'FEASIBLE' if verdict else 'infeasible'}")
    if verdict:
      T["cyl_in_body_grasp"] = [round(float(v), 4) for v in off]
      T["map_counts"] = search.map_counts
      return R, tilt, sgn
    if leg == max_legs:
      break
    yaw = ep.base_yaw(data)
    plan = search.basin(data, cyl, yaw, blocked, stand_h, geo=geo)
    if plan is None:
      T["face_log"] = list(getattr(search, "face_log", []))
      print(f"  [replan] no feasible stance at any searched heading "
            f"({len(T['face_log'])} map(s) built)")
      break
    if plan.get("face_change"):
      T["face_log"] = list(getattr(search, "face_log", []))
      # M3.1. The plan is on another side of the table, so the correction is a
      # ROUTE, not a leg: the two tabletops are 0.30 m apart and the robot is
      # wider than the gap, so it has to go around the outside.
      #
      # M3.4. Unless it is not on another side at all; see
      # `REFACE_NEAR_YAW_DEG`. `approach` quantizes the heading to 30 deg, so
      # the plan it returns after a reface is routinely the heading the robot
      # already holds, and charging that to the face budget aborted the episode
      # with its corrective legs unspent.
      dyaw = float(np.degrees(abs(np.arctan2(
        np.sin(float(plan["yaw"]) - yaw), np.cos(float(plan["yaw"]) - yaw)))))
      near = bool(REFACE_NEAR_YAW_DEG) and dyaw <= float(REFACE_NEAR_YAW_DEG)
      if not near and faces >= MAX_FACE_CHANGES:
        print(f"  [replan] face-change budget spent ({faces}); aborting a "
              f"reface to {np.degrees(plan['yaw']):+.0f} deg "
              f"(|dyaw| {dyaw:.1f} deg, margin "
              f"{plan['margin_m'] * 100:.1f} cm)")
        break
      if float(data.time) > FACE_CHANGE_DEADLINE_S:
        print(f"  [replan] sim t={float(data.time):.0f}s past the "
              f"{FACE_CHANGE_DEADLINE_S:.0f}s face-change deadline; a route "
              f"here would only convert `out_of_reach` into `timeout`")
        break
      tgt = offset_to_base(cyl[:2], plan["yaw"], plan["fwd"], plan["lat"])
      print(f"  [replan] {'RE-DOCK' if near else 'APPROACH SIDE'}: heading "
            f"{np.degrees(plan['yaw']):+.0f} deg (from "
            f"{np.degrees(yaw):+.0f}), offset "
            f"({plan['fwd']:+.3f},{plan['lat']:+.3f}) margin "
            f"{plan['margin_m'] * 100:.1f} cm over {plan['n_feasible']} cells, "
            f"stance {np.round(tgt, 3).tolist()}, route cost "
            f"{plan.get('route_cost_m', 0.0):.2f} m "
            f"({search.n_solve} IK solves)")
      if runner.crouch_h is not None:
        runner.set_crouch(None, seconds=2.0)
        T["pick_crouch_h"] = None
      e, ye, rlog = route_to_pose(runner, tgt, plan["yaw"], geo, tag="reface")
      # `face_changes` keeps counting every reface FLOWN, so the field means
      # what it always meant; only the budget counter distinguishes them.
      n_reface += 1
      if near:
        n_near += 1
      else:
        faces += 1
      T["face_changes"] = n_reface
      T["approach_yaw_deg"] = round(float(np.degrees(plan["yaw"])), 1)
      row = {"kind": "redock" if near else "reface", "err_m": round(e, 4),
             "yaw_err_deg": round(ye, 2), "margin_m": plan["margin_m"],
             "yaw_deg": T["approach_yaw_deg"], "legs": rlog}
      # Kept behind the knob so `REFACE_NEAR_YAW_DEG = 0.0` replays the
      # incumbent record KEY FOR KEY, not merely field for field.
      if REFACE_NEAR_YAW_DEG:
        T["near_refaces"] = n_near
        row["dyaw_deg"] = round(dyaw, 1)
      T["route_log"].append(row)
      off_after = body_offset(data, runner.cyl_pos())
      print(f"  [replan] refaced: offset now "
            f"{np.round(off_after, 3).tolist()} vs plan "
            f"({plan['fwd']:+.3f},{plan['lat']:+.3f})")
      continue
    aim_fwd, aim_lat = plan["fwd"], plan["lat"]
    tgt = offset_to_base(cyl[:2], yaw, aim_fwd, aim_lat)
    print(f"  [replan] basin: offset ({plan['fwd']:+.3f},{plan['lat']:+.3f}) "
          f"margin {plan['margin_m'] * 100:.1f} cm over {plan['n_feasible']} "
          f"feasible cells ({'clean' if plan['clean'] else 'path-fouling'}"
          f"), crouch {plan['crouch_h']}, stance "
          f"{np.round(tgt, 3).tolist()} ({search.n_solve} IK solves)")
    T["basin_margin_m"] = plan["margin_m"]
    T["basin_offset"] = [plan["fwd"], plan["lat"]]
    if runner.crouch_h is not None:
      runner.set_crouch(None, seconds=2.0)   # croucher velocity cmds are DEAD
      T["pick_crouch_h"] = None
    head = np.array([np.cos(yaw), np.sin(yaw)])
    e, detail = corrective_leg(runner, tgt, yaw)
    # A face change spends an iteration of this loop, so count the LEGS, not
    # the iterations: `pick_legs` has to stay comparable with M1.14's numbers.
    n_legs += 1
    T["pick_legs"] = n_legs
    off_after = body_offset(data, runner.cyl_pos())
    yaw_after = ep.base_yaw(data)
    # Signed stance error, in the only frame the grasp cares about. `d_fwd > 0`
    # means the cylinder ended up FURTHER FORWARD in the body frame than the
    # basin asked for, i.e. the robot stopped SHORT. `along` is the same error
    # measured against the commanded base point (negative = short), and the two
    # agree except through the yaw the leg also changed.
    T["pick_leg_log"].append({
      "leg": n_legs,
      "plan": [round(float(plan["fwd"]), 4), round(float(plan["lat"]), 4)],
      "aim_fwd": round(float(aim_fwd), 4),
      "aim_lat": round(float(aim_lat), 4),
      "bias_m": 0.0,
      "lat_bias_m": 0.0,
      "off_before": [round(float(v), 4) for v in off],
      "off_after": [round(float(v), 4) for v in off_after],
      "d_fwd": round(float(off_after[0] - plan["fwd"]), 4),
      "d_lat": round(float(off_after[1] - plan["lat"]), 4),
      "along": round(float((runner.data.qpos[:2] - tgt) @ head), 4),
      "cross": round(float((runner.data.qpos[:2] - tgt)
                           @ np.array([-head[1], head[0]])), 4),
      "err_m": round(float(e), 4),
      "yaw_err_deg": round(float(np.degrees(np.arctan2(
        np.sin(yaw_after - yaw), np.cos(yaw_after - yaw)))), 2),
      "margin_m": plan["margin_m"],
      "n_feasible": plan["n_feasible"],
      "crouch_h": plan["crouch_h"],
      **detail,
    })
    print(f"  [replan] leg {n_legs}: stance err {e * 100:.1f} cm "
          f"(along {T['pick_leg_log'][-1]['along'] * 100:+.1f}, cross "
          f"{T['pick_leg_log'][-1]['cross'] * 100:+.1f}), offset now "
          f"{np.round(off_after, 3).tolist()} vs plan "
          f"({plan['fwd']:+.3f},{plan['lat']:+.3f}) -> d_fwd "
          f"{T['pick_leg_log'][-1]['d_fwd'] * 100:+.1f} cm")
    if plan["crouch_h"] is not None:
      runner.set_crouch(plan["crouch_h"] + CROUCH_BIAS, seconds=3.0)
      T["pick_crouch_h"] = plan["crouch_h"]
      print(f"  [replan] crouched to {plan['crouch_h']:.2f} -> pelvis z "
            f"{float(data.qpos[2]):.3f}")
  T["search_ik_solves"] = search.n_solve
  T["map_counts"] = search.map_counts
  T["cyl_in_body_final"] = [round(float(v), 4)
                            for v in body_offset(data, runner.cyl_pos())]
  if runner.crouch_h is not None:
    runner.set_crouch(None, seconds=2.0)   # never end an episode crouched
  return None


def _table_contact_monitor(runner):
  """`stop()` -> the hand has reached the tabletop.

  The commanded grasp pose clears the top by TABLE_CLEAR, so this should not
  fire; it is the guard for the case where it does (a 2 cm table-height
  randomization, a stale cylinder measurement, an IK residual that lands the
  palm low). M1.5 measured what happens without it: the actuators keep pushing a
  jammed finger and the robot goes over.
  """
  model = runner.model
  tables = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("table", "table_white")}
  hand = set()
  for b in range(model.nbody):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if "right_hand" in n or "right_wrist" in n:
      hand.add(b)
  data = runner.data
  hit = {"n": 0}

  def stop():
    for i in range(data.ncon):
      c = data.contact[i]
      b1 = int(model.geom_bodyid[c.geom1])
      b2 = int(model.geom_bodyid[c.geom2])
      if (b1 in hand and b2 in tables) or (b2 in hand and b1 in tables):
        hit["n"] += 1
        return hit["n"] >= 2        # 2 steps, not a single-step transient
    hit["n"] = 0
    return False
  return stop


def pinch_geometry(runner, offset=None):
  """Where the cylinder sits relative to the pinch, right now.

  M1.10 instrumentation. `depth_m` is the SAME quantity M1.8 called in-palm
  depth and `HybridRunner.slip` returns (the cylinder centre's component along
  the palm's finger axis), so these numbers are directly comparable with the
  M1.8 cage envelope (holds over ~3-4.5 cm) and M1.9's post-close depths. What
  is new is (a) reading it BEFORE the close, where the M1.9 telemetry only had
  it after, and (b) the lateral offset of the cylinder from the pinch CENTRE,
  which is the moment arm the straight-up lift acts through.
  """
  data, ik = runner.data, runner.ik
  p = data.site_xpos[ik.site_id].copy()
  R = data.site_xmat[ik.site_id].reshape(3, 3).copy()
  c = runner.cyl_pos()
  v = R.T @ (c - p)                       # cylinder centre, palm frame
  out = {"depth_m": round(float(v[0]), 4)}
  if offset is not None:
    d = v - np.asarray(offset, float)     # cylinder relative to the pinch
    out["pinch_dx_m"] = round(float(d[0]), 4)
    out["pinch_dy_m"] = round(float(d[1]), 4)
    out["pinch_dz_m"] = round(float(d[2]), 4)
    out["pinch_lat_m"] = round(float(np.hypot(d[1], d[2])), 4)
    pinch_w = p + R @ np.asarray(offset, float)
    out["arm_xy_m"] = round(float(np.hypot(*(pinch_w - c)[:2])), 4)
    out["pinch_up_m"] = round(float(pinch_w[2] - c[2]), 4)
  return out


def fly_grasp_path(runner, R_s, tilt, offset, top_z, verbose=True,
                   back_m=None):
  """Fly the explicit pre-capture path: rise to a standoff behind and above the
  cylinder, traverse horizontally to the apex directly above the grasp pose,
  then descend vertically onto it. Returns a measurement dict.

  This is the M1.7 replacement for M1.6's two skipped moves. It is the SAME
  geometry the search scores (`grasp_path_targets`), so a stance the search
  called feasible is a stance this can fly, and every segment is a planned
  Cartesian path, so the foul model applies to all of it.
  """
  data, ik = runner.data, runner.ik

  def tgt():
    return grasp_path_targets(runner.cyl_pos(), R_s, tilt, offset, top_z,
                              back_m=back_m)

  standoff, apex, grasp, dz = tgt()
  out = {"tilt": tilt, "grasp_dz_m": round(float(dz), 4),
         "apex_up_m": round(float(apex[2] - grasp[2]), 4),
         "standoff_back_m": round(float(np.linalg.norm(
           (apex - standoff)[:2])), 4)}
  # pre-curl BEFORE raising. 0.25 is now a geometric requirement, not a taste:
  # at 0.4 the curled fingers close the vertical corridor over the cylinder's
  # axis (measured: a vertical descent would then need a 7.7 cm apex INSIDE the
  # cage), and fully open fingertips graze the tabletop.
  runner.set_grip(GRASP_PRECURL, seconds=0.4)
  cyl0 = runner.cyl_pos()

  # M3.2. LIFT CLEAR OF THE TABLETOP BEFORE TRAVERSING TO THE STANDOFF.
  #
  # The raise is a straight palm line from wherever the arm hangs (palm z 0.665,
  # BELOW the 0.733 tabletop) to a standoff at z ~0.90. When the standoff is
  # over the table that line crosses the tabletop, and the fingers reach 2-4 cm
  # below the palm, so the hand drags. Measured on the M3.1 capture failures,
  # second by second: 3.2 cm of base drift at t=7 s, then `table <->
  # right_hand_index_0_link` with 0.374 rad of joint tracking error, base
  # velocity spiking to 0.27 m/s, and 7.4 -> 23.1 -> 39.4 cm of drift in two
  # seconds. **The hand pushes on the table, the reaction force perturbs the
  # standing controller, and the robot STEPS AWAY**, and the standoff and apex
  # are world points, so they are then out of reach and the aim-and-correct
  # rounds reconfigure the arm onto a branch 29 cm clear of where it was going.
  #
  # Base drift over the whole approach, which is the quantity that separates:
  #
  #   | preset          | outcome          | n  | min  | median | max     |
  #   | both            | success          | 32 | 0.7  | 1.7    | 3.3 cm  |
  #   | `shipped_table` | `never_captured` | 25 | 2.1  | 20.1   | 41.2 cm |
  #
  # and it predicts the apex error almost exactly, episode by episode.
  #
  # So go UP first, at the current palm xy and with the current palm
  # orientation, and only then translate. The gate is geometric, not a
  # threshold: the pre-lift fires only when the straight line to the standoff
  # would actually pass over the brown tabletop with less than
  # `GRASP_PRELIFT_CLEAR_M` of clearance. At the nominal stance the cylinder
  # sits 5 cm from the near edge and the whole approach happens OUTSIDE the
  # table footprint, so the test is false and the path is M3.1's exactly.
  out["prelift"] = None
  if GRASP_PRELIFT and _line_over_table(
      runner.model, data.site_xpos[ik.site_id].copy(), standoff, top_z,
      GRASP_PRELIFT_CLEAR_M, pad=GRASP_PRELIFT_PAD_M):
    palm_p, palm_R = runner.palm_pose()
    z_up = max(float(standoff[2]), float(top_z) + GRASP_PRELIFT_CLEAR_M)
    p_up = np.array([palm_p[0], palm_p[1], z_up])
    runner.move_tag = "pick_prelift"
    e_up = float(runner.goto(p_up, palm_R, rounds=1, rate=0.008))
    # What the move was FOR, not how far it missed by. The pre-lift exists to
    # put the hand over the tabletop, and the pre-curled fingers hang 2-4 cm
    # below the palm, so this is the number that says whether it worked.
    clear = float(data.site_xpos[ik.site_id][2]) - float(top_z)
    out["prelift"] = {"z_from": round(float(palm_p[2]), 4),
                      "z_to": round(z_up, 4), "err_m": round(e_up, 4),
                      "clear_m": round(clear, 4)}
    if verbose:
      print(f"[3 PRELIFT] the line to the standoff crosses the tabletop; "
            f"palm z {palm_p[2]:.3f} -> {z_up:.3f} first "
            f"(err {e_up * 100:.1f} cm, cleared {clear * 100:.1f} cm)")
    # M3.4. Only if the clearance the pre-lift is for was not reached.
    if GRASP_PRELIFT_MIN_M is not None:
      out["prelift"]["retries"] = 0
      for _ in range(int(GRASP_PRELIFT_RETRIES)):
        if clear >= float(GRASP_PRELIFT_MIN_M):
          break
        runner.move_tag = "pick_prelift"
        e_up = float(runner.goto(p_up, palm_R, rounds=1, rate=0.008))
        gained = float(data.site_xpos[ik.site_id][2]) - float(top_z) - clear
        clear += gained
        out["prelift"]["retries"] += 1
        out["prelift"]["err_m"] = round(e_up, 4)
        out["prelift"]["clear_m"] = round(clear, 4)
        if verbose:
          print(f"[3 PRELIFT] under the {GRASP_PRELIFT_MIN_M * 100:.0f} cm "
                f"floor; round {out['prelift']['retries'] + 1} -> palm z "
                f"{data.site_xpos[ik.site_id][2]:.3f} "
                f"(err {e_up * 100:.1f} cm, cleared {clear * 100:.1f} cm, "
                f"gained {gained * 100:+.1f} cm)")
        # An arm at its reach limit does not get there by being asked again;
        # bound the cost on the episodes the rounds cannot help.
        if gained < float(GRASP_PRELIFT_GAIN_M):
          break

  # rounds=2, not 1: the one-shot palm error is 4.85 cm (the pelvis drifts ~2 cm
  # as the arm's mass extends, after the IK was solved against the old base
  # pose), and a standoff reached 3.7 cm low is what turned the traverse into a
  # climb through the cylinder in the first place.
  runner.move_tag = "pick_raise"
  out["standoff_err_m"] = round(float(runner.goto(
    standoff, R_s, rounds=APEX_ROUNDS, rate=0.008)), 4)
  if verbose:
    print(f"[3 RAISE ] standoff err {out['standoff_err_m'] * 100:.1f} cm, "
          f"palm z {data.site_xpos[ik.site_id][2]:.3f} (tabletop {top_z:.3f}), "
          f"arm-table contacts {sorted(runner.contact_pairs('right_'))}")

  runner.move_tag = "pick_apex"
  out["apex_err_m"] = round(float(runner.goto_track(
    lambda: tgt()[1], R_s, rounds=APEX_ROUNDS, max_shift=0.06, rate=0.006)), 4)
  out["nudge_pre_m"] = round(float(np.linalg.norm(runner.cyl_pos() - cyl0)), 4)
  if verbose:
    print(f"[3 APEX  ] apex err {out['apex_err_m'] * 100:.1f} cm "
          f"({out['apex_up_m'] * 100:.1f} cm over the grasp pose), cylinder "
          f"nudged {out['nudge_pre_m'] * 100:.1f} cm")

  runner.move_tag = "pick_descend"
  before = runner.cyl_pos()
  out["approach_err_m"] = round(float(runner.goto_track(
    lambda: tgt()[2], R_s, rounds=2, max_shift=0.03, rate=0.004,
    on_step=_table_contact_monitor(runner))), 4)
  out["nudge_m"] = round(float(np.linalg.norm(runner.cyl_pos() - cyl0)), 4)
  out["nudge_descend_m"] = round(float(np.linalg.norm(
    runner.cyl_pos() - before)), 4)
  # where the pinch ended up, vertically, against the cylinder it has to hold
  palm_p = data.site_xpos[ik.site_id].copy()
  palm_R = data.site_xmat[ik.site_id].reshape(3, 3).copy()
  centre = palm_p + palm_R @ offset
  out["grasp_dz_err_m"] = round(float(centre[2] - (tgt()[2] + palm_R @ offset)[2]), 4)
  out["grasp_centre_up_m"] = round(float(centre[2] - runner.cyl_pos()[2]), 4)
  out["pinch_over_top_m"] = round(
    float(centre[2] - (runner.cyl_pos()[2] + CYL_HALF)), 4)
  # M1.10: the in-palm depth the close STARTS from. The M1.9 telemetry only
  # had the depth after the close, which cannot distinguish "the descent put it
  # there" from "the close slid it there".
  pre = pinch_geometry(runner, offset)
  out["preclose_depth_m"] = pre["depth_m"]
  out["preclose_pinch_lat_m"] = pre["pinch_lat_m"]
  out["preclose_arm_xy_m"] = pre["arm_xy_m"]
  out["preclose_pinch_dx_m"] = pre["pinch_dx_m"]
  out["preclose_contacts"] = int(runner.finger_contacts())
  out["table_contact"] = sorted(c for c in runner.contact_pairs("right_hand")
                                if "table" in c)
  out["hand_geom"] = hand_geometry_report(runner, runner.cyl_pos())
  if verbose:
    print(f"[4 DESCEN] palm err {out['approach_err_m'] * 100:.1f} cm, cylinder "
          f"nudged {out['nudge_m'] * 100:.1f} cm, pinch "
          f"{out['grasp_centre_up_m'] * 100:+.1f} cm over the cylinder centre "
          f"({out['pinch_over_top_m'] * 100:+.1f} cm over its TOP), links on "
          f"the body {out['hand_geom']['links_on_body']}"
          f"{', TABLE ' + str(out['table_contact']) if out['table_contact'] else ''}")
  return out


HAND_LINKS = ("right_hand_palm_link", "right_hand_index_0_link",
              "right_hand_index_1_link", "right_hand_middle_0_link",
              "right_hand_middle_1_link", "right_hand_thumb_0_link",
              "right_hand_thumb_1_link", "right_hand_thumb_2_link")


def hand_geometry_report(runner, cyl):
  """Where the hand's mesh actually is, relative to the cylinder it must hold.

  M1.9 instrumentation. The pinch-height telemetry the pipeline already had is
  a COMMAND (`grasp_dz_m`) and a palm-frame reconstruction
  (`grasp_centre_up_m`); neither says whether a finger is over the cylinder's
  body or over its rim. This reads the real mesh vertices of every hand link at
  the pose the arm is in and reports, per link, the vertical span relative to
  the cylinder's centre and its top face, plus how close the link comes to the
  cylinder's axis. A link whose whole span is above `top_rel = 0` is over the
  rim and can only close on air.
  """
  model, data = runner.model, runner.data
  c = np.asarray(cyl, float)
  out = {}
  lo_all, near_all = 1e9, 1e9
  for name in HAND_LINKS:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
      continue
    zs, rs = [], []
    for g in range(model.ngeom):
      if int(model.geom_bodyid[g]) != bid:
        continue
      m = int(model.geom_dataid[g])
      if m < 0:
        continue
      a = int(model.mesh_vertadr[m])
      v = np.asarray(model.mesh_vert[a:a + int(model.mesh_vertnum[m])], float)
      w = data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T
      zs.append(w[:, 2])
      rs.append(np.hypot(w[:, 0] - c[0], w[:, 1] - c[1]))
    if not zs:
      continue
    z = np.concatenate(zs)
    r = np.concatenate(rs)
    key = name.replace("right_hand_", "").replace("_link", "")
    out[key] = {"z_lo_rel_centre": round(float(z.min() - c[2]), 4),
                "z_hi_rel_centre": round(float(z.max() - c[2]), 4),
                "z_lo_rel_top": round(float(z.min() - (c[2] + CYL_HALF)), 4),
                "axis_dist_min": round(float(r.min()), 4)}
    lo_all = min(lo_all, float(z.min() - c[2]))
    near_all = min(near_all, float(r.min()))
  out["hand_z_lo_rel_centre"] = round(lo_all, 4)
  out["hand_axis_dist_min"] = round(near_all, 4)
  # How many links have any mesh vertex inside the cylinder's radius AND below
  # its top face, i.e. actually alongside the body rather than over the rim.
  out["links_on_body"] = sorted(
    k for k, v in out.items()
    if isinstance(v, dict) and v["axis_dist_min"] < CYL_R + 0.004
    and v["z_lo_rel_top"] < 0.0)
  return out


def finger_overdrive(runner):
  """`cmd - q` per finger, signed toward closure: the M1.11 blocked deflection.

  The position command each finger is being driven to, minus where the finger
  actually is. In free air it is the actuator's steady-state following error;
  against a blocked finger it is the whole drive torque divided by kp (see
  `GRIP_CAP`, which is a bound on exactly this quantity).
  """
  d = runner.data
  return [float((d.ctrl[a] - d.qpos[q]) * (1.0 if closed >= 0.0 else -1.0))
          for (a, closed), q in zip(runner.ctrl.right_finger_actuators,
                                    runner._fq)]


def squeeze(runner, target, seconds, samples=CLOSE_SAMPLES, offset=None,
            depth_min=None, tilt_max=None):
  """`Runner.set_grip(target, seconds)`, step for step, with the in-palm depth
  sampled through the ramp.

  M1.10 instrumentation. The loop is `set_grip`'s own loop (same step count,
  same linear interpolation of both grip channels) because the alternative
  (calling `set_grip` on sub-targets) rounds `int(seconds / dt)` per sub-ramp
  and comes out 392 steps instead of 399, which this pipeline is chaotic enough
  to notice: it flipped a grid cell from a clean lift to a 90 deg topple.
  """
  n = int(seconds / runner.model.opt.timestep)
  a0_t, a0_f = runner.grip_alpha, runner.grip_alpha_f
  marks = {min(n, max(1, round(n * k / samples))) for k in range(1, samples + 1)}
  trace = []
  stop = {"frac": None, "why": None}
  frac = 0.0
  for i in range(n):
    if stop["frac"] is None:
      frac = (i + 1) / n
    runner.grip_alpha = a0_t + (target - a0_t) * frac
    runner.grip_alpha_f = a0_f + (target - a0_f) * frac
    runner.step_once()
    # The gate. Checked on the SAME schedule regardless of whether it can fire,
    # and it only ever stops the grip advancing; the step count, and so the
    # timing of everything downstream, is identical either way.
    if (stop["frac"] is None and (depth_min is not None or tilt_max is not None)
        and (i + 1) % CLOSE_GATE_EVERY == 0
        and frac * target >= CLOSE_GATE_MIN_GRIP):
      d = pinch_geometry(runner)["depth_m"]
      t = float(runner.cyl_tilt_deg())
      why = ("depth" if depth_min is not None and d < depth_min else
             "tilt" if tilt_max is not None and t > tilt_max else None)
      if why is not None:
        stop.update({"frac": frac, "why": why, "at_depth_m": round(d, 4),
                     "at_tilt_deg": round(t, 1)})
    if (i + 1) in marks:
      g = pinch_geometry(runner, offset)
      trace.append({"t": round(float(runner.data.time), 2),
                    "grip": round(float(runner.grip_alpha_f), 3),
                    "depth_m": g["depth_m"],
                    "contacts": int(runner.finger_contacts()),
                    "tilt_deg": round(float(runner.cyl_tilt_deg()), 1)})
  return trace, stop


def close_and_lift(runner, R_s, cyl_rest, grip=None, lift=None, verbose=True,
                   offset=None, lift_R=None):
  """Close the hand on the cylinder and lift it. Returns a measurement dict.

  `lift_R` (M1.13, `CRADLE_AT = "lift"`) is the palm orientation the LIFT ramp
  is flown to, when the cradle roll is folded into it instead of run as its own
  move. `None` keeps the grasp orientation, which is the incumbent.
  """
  grip = CLOSE_GRIP if grip is None else grip
  lift = LIFT_M if lift is None else lift
  runner.trace_slip = True           # in-palm slip trace starts at capture
  # M1.9: the close is where a LOW-FRICTION cylinder is lost. Measured over the
  # 30 `reachable` episodes, in-palm depth after the close against the drawn
  # cylinder friction: mu <= 2.2 gives 1.0-2.2 cm (10 of 11 episodes) and mu >=
  # 2.3 gives 3.5-4.3 cm (16 of 16). At <= 2 cm the cage stops aligning and the
  # cylinder rotates to 37-63 deg, which is what then arrives at the blue table
  # lying over.
  # M1.12: the in-palm depth the close starts from, for the friction estimate
  # below. Read-only (`pinch_geometry` steps no physics), so it cannot change
  # the episode.
  depth_pre = float(pinch_geometry(runner)["depth_m"])
  tr, stop = squeeze(runner, grip, CLOSE_S, offset=offset)
  out = {"close_trace": tr, "close_gate": stop,
         "close_grip_final": round(float(runner.grip_alpha_f), 3)}
  depth_squeeze = float(pinch_geometry(runner)["depth_m"])
  runner.run(0.8)
  # M1.12: how far the close dragged the cylinder along the finger axis. This is
  # the friction estimate of the M1.12 close-delta note -- see it for the
  # measured delta-vs-mu table and what the observable can and cannot
  # separate. Measured HERE, before the cap is armed, so the cap cannot feed
  # back into its own input.
  deltas, hold_cap, rung = publish_close_deltas(
    runner, depth_pre, depth_squeeze,
    float(pinch_geometry(runner)["depth_m"]))
  out["close_depth_pre_m"] = round(depth_pre, 4)
  out["close_depth_delta_m"] = round(deltas["close"], 4)
  out["squeeze_depth_delta_m"] = round(deltas["squeeze"], 4)
  out["hold_cap_used"] = hold_cap
  out["hold_cap_rung"] = rung
  if verbose:
    runner.report("closed")
  if hold_cap is not None:
    runner.ramp_grip_cap(hold_cap, seconds=0.0)
  out.update({"close_slip_m": round(float(runner.slip()), 4),
         "close_tilt_deg": round(float(runner.cyl_tilt_deg()), 1),
         "n_finger_contacts": int(runner.finger_contacts()),
         # which links the close actually landed on (M1.9): a grasp whose pinch
         # is above the cylinder's rim closes with 0 of these, and one that has
         # only `index` shows the upper finger touching and nothing opposing it.
         "close_contacts": sorted(
           c.replace("right_hand_", "").replace("_link", "")
           for c in runner.contact_pairs("right_hand") if "red_block" in c)})
  palm_now = runner.data.site_xpos[runner.ik.site_id].copy()
  # M1.10: the moment arm the straight-up lift acts through, and whether the
  # cylinder's base is still pinned to the tabletop when it starts. The
  # high-friction failure is `lift_rise_m` NEGATIVE with zero contacts, i.e. the
  # cylinder rotating about a stuck base instead of translating.
  pre = pinch_geometry(runner, offset)
  out["prelift_depth_m"] = pre["depth_m"]
  out["prelift_pinch_lat_m"] = pre.get("pinch_lat_m")
  out["prelift_arm_xy_m"] = pre.get("arm_xy_m")
  # `table` is the BROWN pick table; `table_white` is the blue destination.
  out["prelift_table_contact"] = bool(
    runner.pair_contact("red_block", "table"))
  cyl_pre = runner.cyl_pos().copy()
  runner.move_tag = "lift"
  # Sample the topple as it happens. `on_step` is polled every physics step and
  # only aborts the ramp when it returns True, so a recorder can ride along.
  lt = []

  phase = ["ramp"]

  def _watch():
    if True:
      lt.append({"ph": phase[0], "palm_dz_m": round(float(
        runner.data.site_xpos[runner.ik.site_id][2] - palm_now[2]), 4),
        "cyl_dz_m": round(float(runner.cyl_pos()[2] - cyl_pre[2]), 4),
        "tilt_deg": round(float(runner.cyl_tilt_deg()), 1),
        "contacts": int(runner.finger_contacts()),
        "on_table": bool(runner.pair_contact("red_block", "table"))})
    return False
  runner.sampler = _watch
  runner.goto(palm_now + np.array([0, 0, lift]),
              R_s if lift_R is None else lift_R, rounds=LIFT_ROUNDS,
              rate=LIFT_RATE, settle=LIFT_SETTLE)
  # The settle is sampled too, and it is NOT a formality: M1.9 read the
  # high-friction failure as "the straight-up lift topples the cylinder", but
  # the ramp gets it to 15 cm with contacts on it and the grip loses it HERE.
  phase[0] = "settle"
  runner.run(1.0)
  runner.sampler = None
  out["lift_trace"] = lt
  # where in the lift the cylinder went over, in cm of palm rise
  tip = next((r for r in lt if r["tilt_deg"] > 45.0), None)
  out["tip_at_palm_dz_m"] = None if tip is None else tip["palm_dz_m"]
  out["tip_at_cyl_dz_m"] = None if tip is None else tip["cyl_dz_m"]
  palm_end = runner.data.site_xpos[runner.ik.site_id].copy()
  out["lift_palm_dxy_m"] = round(
    float(np.hypot(*(palm_end - palm_now)[:2])), 4)
  out["lift_palm_dz_m"] = round(float(palm_end[2] - palm_now[2]), 4)
  out["lift_cyl_dxy_m"] = round(
    float(np.hypot(*(runner.cyl_pos() - cyl_pre)[:2])), 4)
  out["lift_table_contact"] = bool(
    runner.pair_contact("red_block", "table"))
  post = pinch_geometry(runner, offset)
  out["lift_depth_m"] = post["depth_m"]
  out["lift_pinch_lat_m"] = post.get("pinch_lat_m")
  rise = float(runner.cyl_pos()[2] - cyl_rest[2])
  out["lift_rise_m"] = round(rise, 4)
  out["n_finger_contacts_lifted"] = int(runner.finger_contacts())
  out["captured"] = bool(rise > 0.10 and runner.finger_contacts() > 0)
  # M1.9: `captured` is blind to the thing that decides the rest of the episode.
  # The cage aligns the cylinder only while it sits AT THE PINCH; at <= ~2 cm of
  # in-palm depth it holds but rotates to 37-63 deg (M1.8's envelope), and every
  # such episode arrives at the blue table lying over. So record the depth and
  # the in-grip tilt, and score `seated` = captured AND still upright in the
  # cage. `SEATED_TILT_DEG` is M1.8's measured seating threshold: the compliant
  # press rights 11 of 15 cells that touch down at <= 25 deg and 1 of 6 at
  # 30-61 deg.
  out["lift_slip_m"] = round(float(runner.slip()), 4)
  out["lift_tilt_deg"] = round(float(runner.cyl_tilt_deg()), 1)
  out["seated"] = bool(out["captured"]
                       and out["lift_tilt_deg"] <= SEATED_TILT_DEG)
  out["hold_released"] = bool(getattr(runner, "hold_released", False))
  out["hold_cap"] = HOLD_CAP
  # The lift is over; the carry gets its own (or no) bound. Done AFTER every
  # measurement above, so the grid's `seated` is untouched by this line.
  if hold_cap is not None and carry_cap_at(runner) == "lift":
    release_carry_cap(runner)
  return out


def hold_cap_for(deltas):
  """`(cap, rung)` for a close whose measured depth deltas are `deltas`."""
  return HOLD_CAP, None


def publish_close_deltas(runner, depth_pre, depth_squeeze, depth_close):
  """Record the close's friction observable and pick the hold cap off it.

  `close_and_lift` both TOOK these three in-palm depth samples and ARMED the
  cap from them; the two are separable and only the second belongs to whoever
  flies the close. Any driver that can produce the same three samples --
  `pinch_geometry` before the squeeze, at the end of it, and after the 0.8 s
  settle -- publishes them here and gets the same `runner.close_deltas` that
  `CRADLE_TIGHTEN_FIRST` and `CARRY_CAP_AT` read downstream.

  Read-only with respect to the physics: it steps nothing and arms nothing.
  """
  deltas = {"squeeze": depth_squeeze - depth_pre,
            "close": depth_close - depth_pre}
  runner.close_deltas = deltas
  cap, rung = hold_cap_for(deltas)
  return deltas, cap, rung


def cradle_frame(R_s, tilt):
  """`(R_c, R_hold, roll_deg)`: the carry orientation and what is flown to it.

  The cradle roll is a rotation of the palm about the HELD CYLINDER'S axis
  (palm z), by `-(tilt + CRADLE_PITCH)`. It is computed in three places now
  (`run_once`, and the learned grasp's own end-of-episode probe), and M1.13
  measured the angle as a sharp optimum rather than a plateau, so it is
  computed in ONE place.
  """
  roll_deg = -(tilt + CRADLE_PITCH) * float(CRADLE_ROLL_SCALE)
  R_c = pitch_about(R_s, R_s[:, 2], roll_deg)
  return R_c, R_c, roll_deg


def cradle_tighten_first(runner):
  """Does the carry grip go up BEFORE the cradle roll on this episode?

  `CRADLE_TIGHTEN_FIRST` is either a bool or a `(key, delta_max_m)` pair read
  against the close's measured depth deltas -- the same per-episode friction
  observable the M1.12 close-delta note describes.
  """
  spec = CRADLE_TIGHTEN_FIRST
  if not isinstance(spec, (tuple, list)):
    return bool(spec)
  key, thr = spec[0], float(spec[1])
  deltas = getattr(runner, "close_deltas", None)
  if not deltas:
    return False
  return bool(deltas[key] <= thr)


def carry_cap_at(runner):
  """Which carry move hands the fingers their full command back, this episode.

  A plain string is that move. A `(key, delta_min_m, above, below)` tuple picks
  per episode off the close's measured depth delta -- see `CARRY_CAP_AT`.
  """
  spec = CARRY_CAP_AT
  if not isinstance(spec, (tuple, list)):
    return spec
  key, thr, above, below = spec[0], float(spec[1]), spec[2], spec[3]
  deltas = getattr(runner, "close_deltas", None)
  if not deltas:
    return below
  return above if deltas[key] >= thr else below


def release_carry_cap(runner):
  """Hand the fingers their full position command back."""
  if runner.grip_cap is None:
    return
  runner.hold_tol = None
  runner.ramp_grip_cap(None, seconds=0.0)


def set_cylinder(model, data, xy, top_z, drop=0.01):
  """Stand the cylinder upright at `xy`, `drop` above the tabletop, at rest."""
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
  adr = int(model.jnt_qposadr[jid])
  data.qpos[adr:adr + 2] = np.asarray(xy, float)
  data.qpos[adr + 2] = float(top_z) + ep.CYL_HALF + drop
  data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
  dadr = int(model.jnt_dofadr[jid])
  data.qvel[dadr:dadr + 6] = 0.0
  mujoco.mj_forward(model, data)


def stow_arm(runner, up=0.15):
  """Lift the palm `up` metres, orientation held, so the cylinder can be dropped
  into place underneath it.

  A static-stance harness has to place the cylinder AFTER the robot has settled,
  and at the near offsets the RESTING hand is already there: the palm sits at
  offset (0.28, -0.22) at z 0.794, which is the cylinder's own height. A cylinder
  spawned inside the fingers is launched off the table before the grasp is even
  chosen (measured: 2 of 25 cells ended with the cylinder on the floor and a
  "grasp" planned 78 cm above it).
  """
  p, R = runner.palm_pose()
  runner.move_tag = "stow"
  runner.goto(p + np.array([0.0, 0.0, up]), R, rounds=1, rate=0.01)


def pick_only(runner, geo, place_cyl=None, force=None):
  """Settle, choose a grasp from the pose the robot is standing in, fly the
  explicit path, close and lift: no walking, no replanning, no carry.

  This is what the static pick grid (on the `dev` branch) measures: the grasp alone, over a grid
  of cylinder-in-body offsets, from a static stance. It shares `fly_grasp_path`
  and `close_and_lift` with the pipeline, so the grid cannot measure a different
  controller from the one `run_once` flies.
  """
  data, ik = runner.data, runner.ik
  T = {}
  runner.run(2.0)
  # `grasp_center_offset` closes the hand in free air to find the pinch centre.
  # The pipeline does that at the spawn, a metre from the table; a static-stance
  # harness starts already at the table, so the cylinder is parked out of reach
  # and dropped into place afterwards; otherwise the measurement itself bats it
  # (measured: tilt 19 -> 62 deg before the grasp had even been chosen).
  offset = runner.grasp_center_offset()
  if place_cyl is not None:
    stow_arm(runner)
    place_cyl()
    runner.run(1.0)
  cyl_rest = runner.cyl_pos()
  T["cyl_rest_tilt_deg"] = round(runner.cyl_tilt_deg(), 2)
  search = GraspSearch(runner.model, runner.ctrl, ik, offset,
                       geo["brown_top_z"])
  cyl = runner.cyl_pos()
  T["offset_realized"] = [round(float(v), 4)
                          for v in body_offset(data, cyl)]
  resid, R_s, tilt, sgn, foul = search.evaluate(search.snapshot(data), cyl,
                                                iters=300, log=True)
  T["candidates"] = list(search.cand_log)
  if force is not None:
    # M1.9: replay ONE named (tilt, sgn) instead of the search's choice, so the
    # tilt axis can be measured at fixed geometry rather than inferred from
    # whichever tilt the search happened to pick per cell.
    ftilt, fsgn = force
    got = [(R, t, sg, dz) for R, t, sg, dz
           in search._frames(data.site_xpos[ik.site_id][:2], cyl)
           if t == ftilt and sg == fsgn]
    if not got:
      T.update({"abort": "forced_tilt_unavailable", "captured": False,
                "tilt": ftilt, "sgn": fsgn})
      return T
    R_s, tilt, sgn = got[0][0], got[0][1], got[0][2]
    resid = next((c["resid_m"] for c in search.cand_log
                  if c["tilt"] == ftilt and c["sgn"] == fsgn), 0.0)
    foul = next((c["foul"] for c in search.cand_log
                 if c["tilt"] == ftilt and c["sgn"] == fsgn), False)
    T["forced"] = [int(ftilt), float(fsgn)]
  T["resid_m"] = round(float(resid), 4)
  T["foul"] = bool(foul)
  T["sgn"] = float(sgn)
  if resid > GRASP_TOL and force is None:
    T["abort"] = "no_grasp_ik"
    T["captured"] = False
    T["tilt"] = tilt
    return T
  T.update(fly_grasp_path(runner, R_s, tilt, offset, geo["brown_top_z"]))
  if T["nudge_m"] > 0.05:
    T["abort"] = "batted_away"
  T.update(close_and_lift(runner, R_s, cyl_rest, offset=offset))
  cyl_f = runner.cyl_pos()
  T["cyl_final"] = [round(float(v), 4) for v in cyl_f]
  T["off_table"] = bool(cyl_f[2] < geo["brown_top_z"] - 0.05)
  T["moves"] = runner.move_log
  devs = [(m["tag"], m["dev0_m"], m["dev0_up_m"]) for m in runner.move_log]
  ap = [d for d in devs if d[0] == "pick_descend"]
  T["dev_approach_m"] = ap[0][1] if ap else None
  T["dev_up_m"] = ap[0][2] if ap else None
  T["sim_time_s"] = round(float(data.time), 2)
  T["ctrl_ik_solves"] = int(runner.ik_solves)
  T["search_ik_solves"] = int(search.n_solve)
  return T



# ======================================================================= #
# M1.8: the place, as a stage of its own
# ======================================================================= #
def on_blue(geo, p):
  """Cylinder resting inside the blue tabletop footprint, at seat height."""
  c, h, top = geo["blue_center"], geo["blue_half"], geo["blue_top_z"]
  return bool(abs(p[0] - c[0]) < h[0] - 0.02
              and abs(p[1] - c[1]) < h[1] - 0.02
              and abs(p[2] - (top + ep.CYL_HALF)) < 0.02)


def verify_settled(runner, geo, seconds=2.0, drift_tol=0.01, speed_tol=0.05):
  """Spec 8.2 success gate: upright AND at rest on the blue table for 2 s."""
  model, data = runner.model, runner.data
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
  dadr = model.jnt_dofadr[jid]
  p0 = runner.cyl_pos()
  drift = speed = 0.0
  tilt = runner.cyl_tilt_deg()
  for i in range(int(seconds / model.opt.timestep)):
    runner.step_once()
    if i % MONITOR_EVERY == 0:
      drift = max(drift, float(np.linalg.norm(runner.cyl_pos() - p0)))
      speed = max(speed, float(np.linalg.norm(data.qvel[dadr:dadr + 3])))
      tilt = max(tilt, runner.cyl_tilt_deg())
  p = runner.cyl_pos()
  return {"drift_m": round(drift, 4), "speed_ms": round(speed, 4),
          "tilt_deg": round(tilt, 2), "on_blue": on_blue(geo, p),
          "at_rest": bool(drift < drift_tol and speed < speed_tol),
          "upright": bool(tilt <= UPRIGHT_TILT_DEG)}


# ---------------------------------------------------------------- #
# M3.5: how close the carried cylinder actually comes to the blue top
# ---------------------------------------------------------------- #
def body_lowest_z(model, data, bid):
  """Exact lowest world z over a body's collision primitives.

  An analytic support function in -z per geom, so it is right at any
  orientation. This matters: `CYL_HALF` (3.7 cm) is the drop from the centre
  only when the cylinder is upright, and the cradle carries it at 20-30 deg,
  where the rim and the reinforcing cap's corner reach 4.09-4.11 cm below the
  centre (the drop peaks at 25 deg). Using `CYL_HALF` overstates the
  clearance by 4 mm against a hazard §3.10 filed at 5 mm.
  """
  lo = None
  for g in range(model.ngeom):
    if int(model.geom_bodyid[g]) != bid:
      continue
    n = data.geom_xmat[g].reshape(3, 3)[2]   # world-z parts of the local axes
    s = model.geom_size[g]
    t = int(model.geom_type[g])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER or t == mujoco.mjtGeom.mjGEOM_CAPSULE:
      d = s[1] * abs(n[2]) + s[0] * float(np.hypot(n[0], n[1]))
      if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        d += s[0] * abs(n[2])
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
      d = float(np.abs(n) @ s[:3])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
      d = float(s[0])
    else:                                    # meshes/planes: not on this body
      continue
    z = float(data.geom_xpos[g][2]) - d
    lo = z if lo is None else min(lo, z)
  return lo


def _body_touches_geom(model, data, bid, gid):
  """Is any geom of body `bid` in contact with geom `gid` right now?"""
  for i in range(data.ncon):
    c = data.contact[i]
    if ((int(model.geom_bodyid[c.geom1]) == bid and int(c.geom2) == gid)
        or (int(model.geom_bodyid[c.geom2]) == bid and int(c.geom1) == gid)):
      return True
  return False


class CarryClearance:
  """Per-move minimum clearance between the carried cylinder and the blue top.

  Installed as `runner.sampler` for the span from the un-tuck to the pre-place,
  so it samples on the monitor's own schedule and therefore covers each move's
  trailing settle (the hole M1.10 found in `goto`'s `on_step`). Chains any
  sampler already installed rather than replacing it.
  """

  def __init__(self, runner, geo):
    self.runner, self.data = runner, runner.data
    self.model = runner.model
    self.bid = runner.cyl_bid
    self.gid = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM,
                                 "table_white_top")
    self.top = float(geo["blue_top_z"])
    self.c = np.asarray(geo["blue_center"], float)
    self.h = np.asarray(geo["blue_half"], float)
    self.per_move = {}
    self.prev = None
    self.first_hit_held = None

  def over_blue(self, p):
    pad = BLUE_FOOTPRINT_PAD_M
    return bool(abs(p[0] - self.c[0]) <= self.h[0] + pad
                and abs(p[1] - self.c[1]) <= self.h[1] + pad)

  def __call__(self):
    if self.prev is not None:
      self.prev()
    if not CARRY_CLEAR_TRACE:
      return
    lo = body_lowest_z(self.model, self.data, self.bid)
    if lo is None:
      return
    clear = lo - self.top
    p = self.runner.cyl_pos()
    tag = self.runner.move_tag or "?"
    r = self.per_move.setdefault(tag, {
      "n": 0, "n_over": 0, "hits": 0, "hits_held": 0, "min_m": None,
      "min_over_m": None, "min_held_m": None, "t_min_over": None,
      "tilt_at_min_over": None, "contacts_at_min_over": None})
    r["n"] += 1
    if r["min_m"] is None or clear < r["min_m"]:
      r["min_m"] = clear
    if self.over_blue(p):
      # `held` separates a scrape from an already-lost cylinder: after the
      # fingers lose it, it LIES on the table and every sample reads a
      # clearance of ~0 with a contact, which would fake the mechanism.
      held = int(self.runner.finger_contacts()) > 0
      r["n_over"] += 1
      if r["min_over_m"] is None or clear < r["min_over_m"]:
        r["min_over_m"] = clear
        r["t_min_over"] = float(self.data.time)
        r["tilt_at_min_over"] = self.runner.cyl_tilt_deg()
        r["contacts_at_min_over"] = int(self.runner.finger_contacts())
      if held and (r["min_held_m"] is None or clear < r["min_held_m"]):
        r["min_held_m"] = clear
      if _body_touches_geom(self.model, self.data, self.bid, self.gid):
        r["hits"] += 1
        if held:
          r["hits_held"] += 1
          if self.first_hit_held is None:
            self.first_hit_held = (tag, float(self.data.time), clear)

  def install(self):
    self.prev = self.runner.sampler
    self.runner.sampler = self
    self.runner.carry_clear = self   # so an aborted episode still publishes
    return self

  def remove(self):
    self.runner.sampler = self.prev
    self.prev = None
    return self

  def publish(self, T):
    """Round the trace into the episode record. Flat scalars first, because
    the whole point is a per-episode separator the analysis can sort on."""
    out, best, best_tag = {}, None, None
    for tag, r in self.per_move.items():
      out[tag] = {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in r.items() if v is not None}
      if r["min_over_m"] is not None and (best is None or r["min_over_m"] < best):
        best, best_tag = r["min_over_m"], tag
    T["carry_clear"] = out
    T["carry_clear_min_m"] = None if best is None else round(best, 4)
    T["carry_clear_min_move"] = best_tag
    T["carry_clear_hits"] = sum(r["hits"] for r in self.per_move.values())
    T["carry_clear_hits_held"] = sum(r["hits_held"]
                                     for r in self.per_move.values())
    held = [r["min_held_m"] for r in self.per_move.values()
            if r["min_held_m"] is not None]
    T["carry_clear_min_held_m"] = None if not held else round(min(held), 4)
    T["carry_first_hit_held"] = (
      None if self.first_hit_held is None
      else [self.first_hit_held[0], round(self.first_hit_held[1], 2),
            round(self.first_hit_held[2], 4)])
    u = self.per_move.get("untuck")
    T["untuck_clear_min_m"] = (
      None if u is None or u["min_over_m"] is None else round(u["min_over_m"], 4))
    T["untuck_hits_held"] = 0 if u is None else u["hits_held"]


def place_object(runner, geo, R_c, T, phase=None):
  """Seat the carried cylinder upright on the blue tabletop and let go.

  Everything from the tucked carry to an open hand clear of the object:
  un-tuck, search the tabletop for a reachable point, crouch if standing
  cannot reach it, translate above it, descend to contact, seat, release in
  stages, retract. Extracted from `run_once` so the static place grid (`dev` branch)
  measures THIS code rather than a copy of it, the same relationship
  `pick_only` has with `fly_grasp_path`.
  """
  data, ik = runner.data, runner.ik
  blue_top = geo["blue_top_z"]
  stages = T.setdefault("place_stages", [])

  def stage(name):
    """Read-only snapshot of the held/placed cylinder. Telemetry only."""
    stages.append({
      "stage": name, "t": round(float(data.time), 2),
      "tilt_deg": round(runner.cyl_tilt_deg(), 2),
      "contacts": int(runner.finger_contacts()),
      "slip_m": round(runner.slip(), 4),
      "cyl_z": round(float(runner.cyl_pos()[2]), 4),
      "palm_z": round(float(data.site_xpos[ik.site_id][2]), 4),
      "grip": round(float(runner.grip_alpha), 3),
      "grip_f": round(float(runner.grip_alpha_f), 3),
    })
    return stages[-1]

  # ================= 4. PLACE (crouch only if IK says so) ==============
  def measure_hold():
    """Cylinder pose in the palm frame; re-measure after any slip."""
    p = data.site_xpos[ik.site_id].copy()
    R = data.site_xmat[ik.site_id].reshape(3, 3).copy()
    return R.T @ (runner.cyl_pos() - p)

  def palm_z_now():
    return float(data.site_xpos[ik.site_id][2])

  def best_place_point(hold, R_hold, label):
    """The task only asks for the blue tabletop, so pick the point on it the
    arm can actually reach from wherever the walk left us, instead of forcing
    one fixed target. Both the pre-place and place poses must solve.

    Orientation is held FIXED at the carry frame: the 3-finger pinch sits
    near the cylinder's CoM and does not constrain its rotation, so every
    palm re-aim spins the object inside the cage (measured 2 -> 48 -> 81
    deg). The cradle already holds it upright, so we keep that and only
    translate."""
    c, h = geo["blue_center"], geo["blue_half"]
    best = None
    for px in np.arange(c[0] - h[0] + 0.10, c[0] + h[0] - 0.05, 0.05):
      for py in np.arange(c[1] - h[1] + 0.10, c[1] + h[1] - 0.05, 0.05):
        tgt = np.array([px, py, blue_top + ep.CYL_HALF + PLACE_UP_M])
        palm_p = tgt - R_hold @ hold
        # solve_ms, not solve. M1.6 established that a DLS residual is a
        # property of the SEED and put multi-start in the controller; M1.7 found
        # the pick's feasibility search had been left behind; this search was the
        # third place. It matters here because its residual is what decides
        # whether to CROUCH, and the crouch is the most damaging move in the
        # place: measured, 4 of the 5 crouches over the 30-episode sweep were
        # triggered by a 30-52 mm standing residual at a point the same arm
        # reaches to 0 mm from another posture, and all 5 crouched episodes
        # tipped while all 4 standing successes stood.
        slv = ik.solve_ms if PLACE_IK_MS else ik.solve
        _, rp, rr = slv(data, palm_p, R_hold, tol=PLACE_CROUCH_RESID) \
            if PLACE_IK_MS else slv(data, palm_p, R_hold)
        hi = palm_p + np.array([0, 0, 0.06])
        _, rp2, _ = slv(data, hi, R_hold, tol=PLACE_CROUCH_RESID) \
            if PLACE_IK_MS else slv(data, hi, R_hold)
        score = rp + rp2 + 0.004 * rr
        if best is None or score < best[0]:
          best = (score, tgt, rp, rr, rp2)
    _, tgt, rp, rr, rp2 = best
    # The seated residual at the CHOSEN point is telemetry on every run (one
    # extra kinematic solve on the IK's own scratch, so it cannot perturb the
    # episode); see the note at `PLACE_UP_M` for what it detects.
    seat_palm = tgt - R_hold @ hold - np.array([0, 0, PLACE_UP_M])
    _, rp_seat, _ = slv(data, seat_palm, R_hold, tol=PLACE_CROUCH_RESID) \
        if PLACE_IK_MS else slv(data, seat_palm, R_hold)
    print(f"[8 SEARCH] {label}: best tabletop point "
          f"({tgt[0]:+.2f},{tgt[1]:+.2f}) residual {rp * 1000:.0f} mm / "
          f"{rr:.0f} deg (pre-pose {rp2 * 1000:.0f} mm, seated "
          f"{rp_seat * 1000:.0f} mm)")
    T.setdefault("place_seat_resid_m", []).append(round(float(rp_seat), 4))
    return tgt, max(rp, rp2)

  stage("carry")
  hold = measure_hold()
  print(f"[7 HOLD  ] cylinder in palm frame {np.round(hold, 3).tolist()}")
  clear_trace = CarryClearance(runner, geo).install()

  # 4a. un-tuck along the cradle orientation FIRST (one big move that both
  # extends the arm and reorients rotates the cylinder inside the cage:
  # measured in-grip tilt 18 -> 79 deg; split into small slow steps instead)
  reach_out = ep.pelvis_to_world(data, list(UNTUCK_OFFSET))
  hang = float(-(R_c @ hold)[2])        # how far below the palm the cradle hangs
  # §3.10's quantity, recorded so its arithmetic can be checked against the
  # flown minimum the tracer measures: the reach-out point's clearance for a
  # cylinder treated as upright and hanging `hang` below the commanded palm.
  T["untuck_hang_m"] = round(hang, 4)
  T["untuck_pred_clear_m"] = round(
    float(reach_out[2] - (blue_top + ep.CYL_HALF + hang)), 4)
  # M3.5: the drop is palm-to-LOWEST-VERTEX, measured,
  # not `CYL_HALF + hang`. `CYL_HALF` is the drop from the centre only when the
  # cylinder is upright; the cradle carries it at 20-30 deg, where the barrel
  # rim and the reinforcing cap corner reach 4.09-4.11 cm below the centre. The
  # old arithmetic therefore understated the drop by ~4 mm against a hazard it
  # measured at 5 mm, which is most of the number. Measuring it needs no new
  # constant and is tilt-aware by construction.
  drop = float(palm_z_now() - body_lowest_z(runner.model, data, runner.cyl_bid))
  T["untuck_drop_m"] = round(drop, 4)
  T["untuck_raise_m"] = 0.0
  runner.move_tag = "untuck"
  runner.goto(reach_out, R_c, rounds=2, rate=0.004, settle=CARRY_SETTLE)
  runner.report("un-tucked")
  stage("untucked")

  # 4b. choose the place point; crouch only if standing can't reach it
  R_p = R_c  # carry orientation held all the way down: no re-aim, no spin
  place_target, resid = best_place_point(hold, R_p, "standing")
  crouched = False
  T["place_force_crouch"] = False
  if resid > PLACE_CROUCH_RESID:
    target_h = PLACE_CROUCH_H
    runner.set_crouch(target_h + CROUCH_BIAS, seconds=3.0)
    crouched = True
    hold = measure_hold()  # the crouch can shift the grip
    print(f"[8 CROUCH] commanded {target_h:.2f} -> pelvis z "
          f"{float(data.qpos[2]):.3f} (standing residual was "
          f"{resid * 1000:.0f} mm)")
    place_target, resid = best_place_point(hold, R_p, "crouched")
    runner.report("crouched")
    stage("crouched")
  else:
    print("[8 CROUCH] not needed: reachable standing")
  T["crouched"] = crouched
  T["place_target"] = [round(float(v), 4) for v in place_target]
  T["place_resid_m"] = round(float(resid), 4)

  def place_palm(R):
    return place_target - R @ hold

  # 4c. translate to the pre-place point above the tabletop (orientation
  # unchanged, so the cylinder keeps the upright pose the cradle gave it)
  runner.move_tag = "place_pre"
  e = runner.goto(place_palm(R_p) + np.array([0, 0, 0.06]), R_p,
                  rounds=2, rate=0.002, settle=CARRY_SETTLE)
  print(f"[9 PRE   ] palm err {e * 100:.1f} cm")
  runner.report("pre-place")
  stage("pre_place")
  # The trace ends here: the descent's whole job is to take the clearance to
  # zero, so sampling it past the pre-place would only measure the design.
  clear_trace.remove().publish(T)
  cc = T["carry_clear"]
  print("[9 CLEAR ] trace off" if not CARRY_CLEAR_TRACE else
        "[9 CLEAR ] cylinder-over-blue min clearance "
        + ", ".join(f"{k} {v['min_over_m'] * 100:+.1f} cm"
                    + (f" ({v['hits_held']}/{v['hits']} contact samples held)"
                       if v["hits"] else "")
                    for k, v in cc.items() if "min_over_m" in v)
        + (f" | worst {T['carry_clear_min_m'] * 100:+.1f} cm at "
           f"{T['carry_clear_min_move']}"
           if T["carry_clear_min_m"] is not None else " | never over the blue top"))
  runner.snap("V5_pre_place", lookat=runner.cyl_pos())

  runner.move_tag = "place_descent"
  touched = runner.touchdown(R_p, max_drop=0.16, seat=SEAT_S, rate=0.002,
                             seat_grip=SEAT_GRIP)
  T["touched"] = bool(touched)
  stage("touchdown")
  # Stop tracing here: once the fingers open the cylinder is no longer in
  # the palm, so "depth along the finger axis" stops meaning slip.
  runner.trace_slip = False
  print(f"[10 TOUCH] table contact={touched}, tilt "
        f"{runner.cyl_tilt_deg():.0f} deg")

  # Release in small slow stages: the cylinder is standing but the hand is
  # still around it, so a fast finger sweep knocks it over (measured: seated
  # at 7 deg, ended at 90 deg). Open only enough to let go, retract, then
  # open fully in free air.
  if RELEASE_STAGED:
    runner.set_grip(0.25, seconds=1.0, fingers=False)  # thumb eases off first
    runner.run(0.5)
    runner.report("thumb open")
    stage("thumb_open")
    runner.set_grip(0.35, seconds=1.0, thumb=False)    # finger pair eases off
    runner.run(0.5)
    runner.report("fingers open")
    stage("fingers_open")
  else:
    runner.set_grip(RELEASE_GRIP, seconds=1.0)         # both, symmetrically
    runner.run(0.5)
    runner.report("thumb open")
    stage("thumb_open")
    runner.run(0.5)
    runner.report("fingers open")
    stage("fingers_open")
  # retract straight up; pulling back along the fingers would sweep the
  # just-placed cylinder off the table
  palm_now = data.site_xpos[ik.site_id].copy()
  runner.move_tag = "retract"
  runner.goto(palm_now + np.array([0, 0, 0.14]), R_p, rounds=1, rate=0.003)
  runner.report("retracted")
  stage("retracted")
  runner.set_grip(0.0, seconds=0.6)                  # fully open, clear of it
  runner.run(1.0)
  if phase is not None:
    phase("released")
  if crouched:  # stand back up, verify the object stays put
    runner.set_crouch(None, seconds=2.5)


def run_once(spawn=(-1.2, 0.15), cyl_shift=(0.0, 0.0), video="v2_run.mp4",
             spawn_yaw=0.0, cyl_xy=None, cyl_mass_scale=1.0, cyl_friction=None,
             brown_dz=0.0, blue_dz=0.0, brown_dxy=(0.0, 0.0),
             blue_dxy=(0.0, 0.0), brown_scale=(1.0, 1.0),
             blue_scale=(1.0, 1.0), sim_limit=300.0, wall_limit=900.0,
             snapshots=None):
  """One full pick & place attempt under the given randomization.

  Returns a telemetry dict: `phase`/`phase_idx` (furthest milestone reached;
  partial progress is real progress), `success` (spec 8.2: upright <= 10 deg,
  at rest on the blue table for 2 s), placement error, max tilt, the in-palm
  slip trace, timings, and the legacy `ok` = {pick, transport, place} flags.
  `video=None` runs headless (no renderer, no frames); use it for sweeps.
  """
  t_wall = time.time()
  model, data, ctrl = ep.build_sim(spawn=spawn)
  geo = randomize_scene(model, data, spawn_yaw=spawn_yaw, cyl_xy=cyl_xy,
                        cyl_shift=cyl_shift, cyl_mass_scale=cyl_mass_scale,
                        cyl_friction=cyl_friction, brown_dz=brown_dz,
                        blue_dz=blue_dz, brown_dxy=brown_dxy,
                        blue_dxy=blue_dxy, brown_scale=brown_scale,
                        blue_scale=blue_scale)
  blue_top = geo["blue_top_z"]          # NOT ep.BLUE_TOP_Z: heights randomize
  ik = ep.ArmIK6(model, ctrl)
  runner = HybridRunner(model, data, ctrl, ik, video=video,
                        snapshots=snapshots, sim_limit=sim_limit,
                        wall_limit=wall_limit)
  ep.STALL_LOG.clear()          # per-episode; the log is module state
  _set_search_deadline(wall_limit)   # the guard `_monitor` cannot reach
  ok = {}
  T = {
    "phase": "start", "phase_idx": 0, "success": False, "ok": ok,
    "abort": None, "error": None,
    "scene": {"brown_top_z": round(geo["brown_top_z"], 4),
              "blue_top_z": round(blue_top, 4),
              "brown_center": [round(float(v), 4)
                               for v in geo["brown_center"][:2]],
              "blue_center": [round(float(v), 4)
                              for v in geo["blue_center"][:2]],
              "cyl_start": [round(v, 4) for v in geo["cyl_start"]],
              "cyl_mass_kg": round(geo["cyl_mass"], 5),
              "cyl_friction": round(geo["cyl_friction"], 3)},
    "place_target": None, "place_resid_m": None, "crouched": False,
    "touched": False, "lift_rise_m": None, "grasp_tilt_deg": None,
    "pick_pos_err_m": None, "pick_yaw_err_deg": None, "cyl_in_body": None,
    "settle": None,
    # M1.5 replanning telemetry
    "pick_legs": 0, "pick_crouch_h": None, "grasp_resid_m": None,
    "cyl_in_body_grasp": None, "basin_margin_m": None, "basin_offset": None,
    "search_ik_solves": 0, "map_counts": [], "foul_veto": False,
    # M3.1 approach-side telemetry
    "approach_search": bool(APPROACH_SEARCH), "face_changes": 0,
    "approach_yaw_deg": None, "face_log": [], "route_log": [],
    "pick_heading_deg": None, "carry_yaw_comp": False,
    "carry_yaw_delta_deg": None,
    # M3.2 approach-azimuth telemetry
    "grasp_az_deg": 0.0, "grasp_fly_gate": None, "grasp_az_tried": [],
    "grasp_az_retry": False, "grasp_az_tries": 0,
    "grasp_apex_tol_m": GRASP_APEX_TOL_M,
    "pick_base_drift_m": None, "pick_base_drift_deg": None,
    "grasp_prelift": bool(GRASP_PRELIFT),
  }

  def phase(name):
    i = PHASES.index(name)
    if i > T["phase_idx"]:
      T["phase"], T["phase_idx"] = name, i

  def episode():
    runner.run(2.0)
    cyl_rest = runner.cyl_pos()
    offset = runner.grasp_center_offset()
    search = GraspSearch(model, ctrl, ik, offset, geo["brown_top_z"])
    print(f"grasp center (palm frame): {np.round(offset, 3).tolist()}")

    # ================= 1. APPROACH (hybrid pose control) =================
    def cyl_in_body():
      """Cylinder position in the robot's frame: what the grasp actually
      cares about (a few cm of stance error moves it out of the graspable
      envelope)."""
      return body_offset(data, runner.cyl_pos())

    ctx = _command_ctx(runner, geo, "pick")
    T["ctx_pick"] = _ctx_record(ctx)
    if COMMANDER is None:
      pe, ye = base.walk_to_pose(runner, *PICK_STANCE)
    else:
      gxy, gyaw = COMMANDER.pick_stance(ctx)
      T["cmd_pick"] = [round(float(gxy[0]), 4), round(float(gxy[1]), 4),
                       round(float(np.degrees(gyaw)), 2)]
      pe, ye = walk_to_pose_around(runner, gxy, gyaw, geo, tag="cmd_pick")
    T["pick_pos_err_m"] = round(pe, 4)
    T["pick_yaw_err_deg"] = round(ye, 2)
    T["cyl_in_body"] = [round(v, 4) for v in cyl_in_body()]
    phase("at_pick")
    print(f"[1 POSE  ] pick stance: pos err {pe * 100:.1f} cm, yaw err "
          f"{ye:.1f} deg, cylinder-in-body "
          f"{np.round(cyl_in_body(), 3).tolist()}")
    runner.snap("V1_stance")

    # ================= 2. SIDE GRASP (standoff-apex-descend) =============
    # The grasp is SEARCHED from the pose actually reached, not assumed from the
    # pose we aimed at. M1.6 kept `side_orientation` (the legacy acceptance, IK
    # on the grasp pose alone) in front of the search so the tuned nominal run
    # took a bit-identical path; M1.7 drops it, because it scores one of the
    # three poses the grasp now flies and would hand back grasps whose apex the
    # arm cannot reach. `replan_pick` evaluates the current pose first, so an
    # adequate stance still costs one `evaluate` and no corrective leg.
    if COMMANDER is not None and getattr(COMMANDER, "direct", False):
      # DIRECT pick: the commander owns the stance, the crouch and the grasp.
      # No feasibility search, no corrective leg, no reface, no IK-residual
      # abort -- whatever it chose is flown from wherever the walk landed.
      # The commander may also decline to grasp from where the walk landed and
      # command a new stance instead (`grasp_or_move`): ITS correction, from
      # its own models, not the replanner's. The LAST stance and grasp are the
      # ones logged as `cmd_pick` / `cmd_grasp`, with their own contexts.
      T["direct_moves"] = []
      while True:
        ch = getattr(COMMANDER, "pick_crouch_h", None)
        T["cmd_pick_crouch_h"] = None if ch is None else round(float(ch), 3)
        if ch is not None:
          runner.set_crouch(float(ch) + CROUCH_BIAS, seconds=3.0)
          T["pick_crouch_h"] = round(float(ch), 3)
        gctx = _command_ctx(runner, geo, "grasp")
        gctx["pelvis_z"] = float(data.qpos[2])
        T["ctx_grasp"] = _ctx_record(gctx)
        T["ctx_grasp"]["pelvis_z"] = round(float(data.qpos[2]), 4)
        move = getattr(COMMANDER, "move_instead", None)
        nxt = move(gctx, len(T["direct_moves"])) if move else None
        if nxt is None:
          break
        mctx = _command_ctx(runner, geo, "pick")
        T["ctx_pick"] = _ctx_record(mctx)
        gxy, gyaw = COMMANDER.pick_stance(mctx)
        T["cmd_pick"] = [round(float(gxy[0]), 4), round(float(gxy[1]), 4),
                         round(float(np.degrees(gyaw)), 2)]
        if runner.crouch_h is not None:
          runner.set_crouch(None, seconds=2.0)
        pe, ye = walk_to_pose_around(runner, gxy, gyaw, geo, tag="cmd_move")
        T["direct_moves"].append({"to": T["cmd_pick"], "err_m": round(pe, 4),
                                  "best_p": round(float(nxt), 4),
                                  "t": round(float(data.time), 1)})
        T["pick_pos_err_m"] = round(pe, 4)
      g_tilt, g_sgn, g_az = COMMANDER.grasp(gctx)
      T["cmd_grasp"] = [round(float(g_tilt), 2), float(g_sgn),
                        round(float(g_az), 2)]
      T["grasp_az_deg"] = round(float(g_az), 2)
      pick = (direct_grasp_frame(runner, search, runner.cyl_pos(), g_tilt,
                                 g_sgn, g_az), float(g_tilt), float(g_sgn))
    else:
      pick = replan_pick(runner, search, geo, T)
    if pick is None:
      T["abort"] = "no_grasp_ik"   # -> out_of_reach, and honestly so
      print(f"ABORT: no reachable side grasp after {T.get('pick_legs', 0)} "
            f"corrective leg(s); this cylinder needs a different approach "
            f"side (out of scope for M1.5)")
      return
    R_s, tilt, sgn = pick
    T["base_at_grasp"] = _base_pose(data)
    T["grasp_tilt_deg"] = tilt
    T["cyl_in_body_grasp"] = [round(float(v), 4) for v in cyl_in_body()]
    cyl = runner.cyl_pos()          # the replan may have moved the base
    print(f"[2 ORIENT] side grasp tilt {tilt} deg down, curl side {sgn:+.0f}")

    # M3.1: the heading the carry frame is built at, read BEFORE the approach.
    # After it, the base may have walked (M3.2 measured up to 41 cm), and it is
    # the heading the frame was CHOSEN at that the place has to correct for.
    yaw_pick = float(ep.base_yaw(data))
    T["pick_heading_deg"] = round(float(np.degrees(yaw_pick)), 1)
    R_s, tilt, sgn, G = fly_grasp_azimuths(
      runner, search, cyl, offset, geo["brown_top_z"], R_s, tilt, sgn, T)
    # The carry orientation, computed here because `CRADLE_AT = "lift"` needs
    # it before the lift ramp is flown.
    R_c, R_hold, roll_deg = cradle_frame(R_s, tilt)
    # M3.2: and the palm's approach AZIMUTH is a second world yaw baked into the
    # same frame. `R_s -> Rz(az) R_s` exactly, and `cradle_frame` rotates about
    # an axis it takes FROM `R_s`, so `R_c -> Rz(az) R_c` exactly too, which
    # means the place has to fly `az` of extra body-relative re-aim on top of
    # the heading's. M3.1's fix corrects for the heading alone; the complete
    # quantity is the heading PLUS the azimuth, and folding it in here is an
    # exact no-op at `az = 0`, i.e. on every episode the gate does not fire on.
    yaw_pick += np.radians(float(T.get("grasp_az_deg") or 0.0))
    T["cradle_roll_deg"] = round(float(roll_deg), 2)
    T["grasp_tilt_deg"] = tilt
    T["grasp_path"] = G
    T["approach_nudge_m"] = G["nudge_m"]
    runner.snap("V2_ready", lookat=runner.cyl_pos())
    if G["nudge_m"] > 0.05:
      T["abort"] = "batted_away"   # -> never_captured
      print("ABORT: batted the cylinder away during the approach")
      return
    phase("aligned")

    L = close_and_lift(runner, R_s, cyl_rest, offset=offset,
                       lift_R=(R_hold if CRADLE_AT == "lift" else None))
    T["regrips"] = []
    # M1.10: the close's own measurements reach the episode record, so a sweep
    # failure can be attributed to the mechanism that caused it instead of to
    # the phase the taxonomy noticed it in. The low-friction mis-seat rotates the
    # cylinder AT THE CLOSE and is scored `dropped_in_transport` because the
    # touchdown detector never fires; `eval/sweep.py`'s taxonomy is the measuring
    # instrument and is deliberately left alone, so the correction lives here.
    T["close"] = {k: v for k, v in L.items()
                  if k not in ("close_trace", "lift_trace")}
    T["lift_rise_m"] = L["lift_rise_m"]
    if L["n_finger_contacts"] > 0:
      phase("grasped")
    rise = L["lift_rise_m"]
    ok["pick"] = L["captured"]
    print(f"[5 LIFT  ] rise {rise * 100:.1f} cm -> "
          f"{'PASS' if ok['pick'] else 'FAIL'}")
    runner.report("lifted")
    runner.snap("V3_lift", lookat=runner.cyl_pos())
    if not ok["pick"]:
      print("\nE2E v2: FAIL (pick)")
      return
    phase("lifted")

    if runner.crouch_h is not None:
      # Stand back up before transport: the croucher's velocity commands are
      # dead (it was trained stationary), so nothing walks while it holds the
      # locomotion slot. `set_crouch(None)` zeroes `last_action` on handoff,
      # which is what makes the walker<->croucher transition safe.
      runner.set_crouch(None, seconds=2.5)
      print(f"[5 STAND ] back up from the pick crouch -> pelvis z "
            f"{float(data.qpos[2]):.3f}, contacts {runner.finger_contacts()}")
      runner.report("stood up")

    # ================= 3. CRADLE CARRY + hybrid transport ================
    # Fingers pitched well ABOVE horizontal: the mouth faces up so gravity both
    # seats the cylinder in the cage and pushes it back along -palm_x (INTO the
    # palm), which is exactly the direction slip runs. 15 deg still let a long
    # loaded translation creep the grip 4.1 -> 8.2 cm; 40 deg roughly doubles
    # the inward component (sin 40 = 0.64 vs 0.26).
    # 40 deg froze slip but let the cylinder rotate until it lay along the
    # PINCH axis (tilt 95 deg = held by its end caps); 20 deg is the
    # compromise, with grip force doing the anti-rotation work instead of
    # geometry.
    # M1.12: the roll is where the carry loses the cylinder (15 of the 20
    # `dropped_in_transport` episodes on `reachable` seed 0 go between
    # `[lifted]` and `[cradled]`, the other 5 at the release).
    tighten_first = cradle_tighten_first(runner)
    T["cradle_tighten_first"] = bool(tighten_first)
    if tighten_first:
      runner.set_grip(CRADLE_GRIP, seconds=0.6)
    runner.move_tag = "cradle"
    n_steps = max(int(CRADLE_STEPS), 1) if tighten_first else 1
    if abs(roll_deg) < 1e-9 or CRADLE_AT != "own":
      n_steps = 0                 # nothing left to roll, or someone else flies it
    for k in range(n_steps):
      R_k = (R_hold if k == n_steps - 1 else
             pitch_about(R_s, R_s[:, 2], roll_deg * (k + 1) / n_steps))
      target_p = data.site_xpos[ik.site_id].copy()
      runner.goto(target_p, R_k, rounds=1, rate=CRADLE_RATE,
                  settle=(CARRY_SETTLE if k == n_steps - 1 else 0.0))
    if carry_cap_at(runner) == "cradle":
      release_carry_cap(runner)
    # Tighten now that the object is captured. 1.1 EJECTS a free cylinder while
    # closing, but mid-carry the extra normal force is what resists rotation
    # about the pinch axis.
    if not tighten_first:
      runner.set_grip(CRADLE_GRIP, seconds=0.6)
    runner.report("cradled")

    # Tuck the hand in near the body for transport. An extended arm sits far
    # from the yaw axis, so a rotator burst's tangential acceleration rips the
    # cylinder out of the cage (measured slip 3.5 -> 17.5 cm). Yaw rotation
    # preserves "mouth up", so the cradle stays valid while tucked.
    # 18 cm above the pelvis, not 12: the cradle holds the cylinder 6.5 cm BELOW
    # the palm (the pinch grips it above its mid-height, so it hangs), and the
    # pelvis sags up to 2.5 cm under the loaded walk, so pelvis + 12 cm arrived
    # at the place stance with the cylinder 1.8-4.0 cm over its rest height,
    # 5 cm from scraping the tabletop and either side of the 3 cm the
    # `transported` phase asks for, on a grip that was demonstrably intact
    # (9 finger contacts, in-palm slip frozen at 3.6 cm). This is a carry
    # constant, not a grasp one; it is raised here only because M1.7's grasp
    # pose is 2 cm higher on the cylinder and that was enough to expose it.
    tuck = ep.pelvis_to_world(data, [0.20, -0.16, 0.18])
    runner.move_tag = "tuck"
    e = runner.goto(tuck, R_hold, rounds=2, rate=0.006, settle=CARRY_SETTLE)
    if carry_cap_at(runner) == "tuck":
      release_carry_cap(runner)
    print(f"[6 TUCK  ] palm err {e * 100:.1f} cm, self-contacts "
          f"{sorted(c for c in runner.contact_pairs('right_hand') if 'red' not in c)}")
    runner.report("tucked")

    # M3.1: the pick may have happened on the far side of the brown table, in
    # which case the straight line to the place stance goes THROUGH it. When it
    # does not (every episode the incumbent pipeline ever flew), this is
    # `base.walk_to_pose(runner, *PLACE_STANCE)` unchanged.
    ctx = _command_ctx(runner, geo, "place")
    T["ctx_place"] = _ctx_record(ctx)
    if COMMANDER is None:
      pxy, pyaw = PLACE_STANCE
    else:
      pxy, pyaw = COMMANDER.place_stance(ctx)
      T["cmd_place"] = [round(float(pxy[0]), 4), round(float(pxy[1]), 4),
                        round(float(np.degrees(pyaw)), 2)]
    pe, ye = walk_to_pose_around(runner, pxy, pyaw, geo, tag="transport")
    T["base_at_place"] = _base_pose(data)
    rise_now = float(runner.cyl_pos()[2] - cyl_rest[2])
    held = runner.finger_contacts() > 0 and rise_now > 0.03
    T["transport_rise_m"] = round(rise_now, 4)
    T["transport_contacts"] = int(runner.finger_contacts())
    ok["transport"] = bool(held)
    T["place_pos_err_m"] = round(pe, 4)
    T["place_yaw_err_deg"] = round(ye, 2)
    print(f"[6 TRANS ] place stance: pos err {pe * 100:.1f} cm, "
          f"yaw err {ye:.1f} deg, held={bool(held)} (cylinder "
          f"{rise_now * 100:+.1f} cm over its rest height, contacts "
          f"{runner.finger_contacts()})")
    runner.report("arrived")
    runner.snap("V4_at_blue", lookat=runner.cyl_pos())
    if not held:
      print("\nE2E v2: FAIL (dropped in transport)")
      return
    phase("transported")

    # ================= 4. PLACE (crouch only if IK says so) ==============
    # `place_object` is module level so the static place grid (`dev`) can fly
    # exactly this code from a static place stance.
    place_object(runner, geo,
                 carry_yaw_fix(R_c, yaw_pick, float(ep.base_yaw(data)), T),
                 T, phase=phase)


    # ================= 5. VERIFY (upright + at rest for 2 s) =============
    settle = verify_settled(runner, geo, 2.0)
    T["settle"] = settle
    ok["place"] = bool(settle["on_blue"] and settle["tilt_deg"] < 25)
    T["success"] = bool(settle["on_blue"] and settle["upright"]
                        and settle["at_rest"])
    if T["success"]:
      phase("placed")
    cyl_f = runner.cyl_pos()
    print(f"[11 DONE ] cylinder {np.round(cyl_f, 3).tolist()}, tilt "
          f"{settle['tilt_deg']:.0f} deg, "
          f"{np.linalg.norm(cyl_f[:2] - np.asarray(T['place_target'])[:2]) * 100:.1f} cm "
          f"from chosen point, drift {settle['drift_m'] * 100:.1f} cm, "
          f"robot z {float(data.qpos[2]):.3f} -> "
          f"{'PASS' if ok['place'] else 'FAIL'}")
    runner.snap("V6_placed", lookat=cyl_f)

  try:
    episode()
  except EpisodeAborted as exc:
    T["abort"] = exc.reason
    print(f"ABORT: {exc.reason} (sim t={data.time:.1f}s, "
          f"pelvis z={float(data.qpos[2]):.3f})")
  except Exception:                       # a crashed episode is a failure,
    T["error"] = traceback.format_exc()   # not a dead sweep
    print(T["error"])

  # ---------------- final telemetry ----------------
  # An episode that aborts inside the carry still measured its clearance up to
  # the abort, and those are exactly the episodes the trace exists for.
  cc = getattr(runner, "carry_clear", None)
  if cc is not None and "carry_clear" not in T:
    cc.remove().publish(T)
  cyl_f = runner.cyl_pos()
  T["walk_stalls"] = list(ep.STALL_LOG)
  T["fell"] = bool(runner.fell)
  T["min_pelvis_z"] = round(float(runner.min_pelvis_z), 3)
  T["cyl_final"] = [round(float(v), 4) for v in cyl_f]
  T["final_tilt_deg"] = round(runner.cyl_tilt_deg(), 2)
  T["max_tilt_deg"] = round(float(runner.max_tilt), 2)
  T["on_blue_table"] = on_blue(geo, cyl_f)
  T["slip_trace"] = runner.slip_trace
  if runner.slip_trace:
    mags = [abs(v) for _, v in runner.slip_trace]
    T["max_slip_m"] = round(max(mags), 4)
    T["final_slip_m"] = runner.slip_trace[-1][1]
  else:
    T["max_slip_m"] = T["final_slip_m"] = None
  if T["place_target"] is not None:
    T["place_err_m"] = round(float(np.linalg.norm(
      cyl_f[:2] - np.asarray(T["place_target"][:2]))), 4)
  else:
    T["place_err_m"] = None
  T["place_err_nominal_m"] = round(float(np.linalg.norm(
    cyl_f[:2] - PLACE_XY)), 4)
  T["sim_time_s"] = round(float(data.time), 2)
  T["wall_time_s"] = round(time.time() - t_wall, 2)
  # M1.6: straightness of every executed palm move (see Runner._log_move)
  T["moves"] = runner.move_log
  devs = [m["dev0_m"] for m in runner.move_log]
  T["max_path_dev_m"] = round(max(devs), 4) if devs else None
  T["cart_path"] = bool(ep.CART_PATH)
  T["ctrl_ik_solves"] = int(runner.ik_solves)
  runner.close_video()

  print(f"\nE2E v2 (hybrid pose + side grasp + crouch): "
        f"{'SUCCESS' if T['success'] else 'FAIL'}  phase={T['phase']}  {ok}")
  return T


# ======================================================================= #
# M1.12: constant overrides from the environment
# ======================================================================= #
# `eval/sweep.py` has no `--set` (and its randomization and taxonomy are off
# limits), so measuring a variant end to end otherwise means editing a default
# and remembering to put it back. `G1_SET` applies exactly the overrides
# `pick_grid.py --set` applies, in the same `mod.ATTR=json` syntax, SEMICOLON
# separated (a JSON list has commas of its own), at IMPORT time -- the sweep pool is spawned, so a parent-process
# assignment would not survive into the workers, but an environment variable
# does. Unset, it is a no-op.
#
#   G1_SET='e2.CARRY_SETTLE=0.1;e2.HOLD_CAP=0.01' \
#     .venv/bin/python eval/sweep.py -n 50 -j 8 --preset reachable --seed 0
def _apply_env_overrides():
  import json
  import os
  spec = os.environ.get("G1_SET", "").strip()
  if not spec:
    return {}
  mods = {"ep": ep, "e2": sys.modules[__name__]}
  applied = {}
  for item in spec.split(";"):
    if not item.strip():
      continue
    name, _, val = item.partition("=")
    mod, _, attr = name.strip().partition(".")
    if mod not in mods or not hasattr(mods[mod], attr):
      raise SystemExit(f"G1_SET: no such constant {name.strip()!r}")
    v = json.loads(val)
    # a schedule is a tuple of rungs; JSON can only give lists
    if isinstance(v, list):
      v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
    setattr(mods[mod], attr, v)
    applied[name.strip()] = v
  return applied


G1_SET_APPLIED = _apply_env_overrides()
if G1_SET_APPLIED:
  print(f"[G1_SET] {G1_SET_APPLIED}")


def main():
  res = run_once()
  sys.exit(0 if res["success"] else 1)


if __name__ == "__main__":
  main()
