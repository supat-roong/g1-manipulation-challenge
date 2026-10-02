# G1 Pick & Place: approach, findings, and what the measurements changed

**Status: 94.4% end-to-end** on the task as shipped (425/450, Wilson 95% CI [91.9, 96.2]), and **88.9%** with the cylinder placed anywhere on the brown tabletop (400/450, CI [85.6, 91.5]), up from 0% at the start, across twenty-four measured milestones.

Those are nine-seed figures. **On the three seeds the project was developed against they read 97.3% and 90.7%**, from the same pipeline, bit-for-bit. Developing and verifying on one seed set absorbed its luck, and the honest numbers are the nine-seed ones. That correction is itself a result, and §4 lists it with the others.

The robot walks to the brown table, picks up the red cylinder, carries it, and places it upright on the blue table, under randomized robot spawn (±15 cm, ±15°) and randomized cylinder position. Success requires the cylinder **on the blue table, tilt ≤ 10°, and still at rest after a 2-second settle**.

The brief asks how the problem was decomposed and iterated on rather than for a score, so this document leads with **what the measurements falsified** (most of it my own reasoning) and treats the success rate as one number among many.

---

## 1. The approach

A four-layer decomposition, deliberately putting learning last:

| Layer | What it does | How |
|---|---|---|
| **High level** | Choose pick/place stance, crouch, grasp geometry | Search over a kinematic feasibility map |
| **Base** | Walk / rotate / crouch to a commanded pose | Provided ONNX policies, frozen |
| **Arm** | Move the palm to a commanded 6-D pose | Damped-least-squares IK + Cartesian waypoint paths |
| **Hand** | Close on the cylinder and hold it | Scripted schedule, and an RL policy trained against it |

The reasoning: the provided policies are a fixed asset, the arm is solved kinematics, and the genuinely hard part, contact, is small and late. Training a policy first would have meant optimizing against a pipeline whose failures were not yet understood. That turned out to be right for an unexpected reason: **every early failure had a cause that measurement could find and geometry could fix, and none of them was where I first guessed.**

## 2. How I worked

**Build the instrument before changing the behaviour.** The end-to-end sweep scores whole episodes, so a grasp change reaches it through the walk, the replanner and the carry; ten episodes cannot separate *"the grasp got better"* from *"the walk landed somewhere else"*. So each layer got a harness that isolates it: a static grasp grid that spawns the robot at the pick stance and places the cylinder to realize a requested offset, and a place grid that spawns it already holding the cylinder at a chosen in-grip pose.

**Keep an ablation ladder and never remove a rung.** Every milestone is a row with the same measurement, so a regression is visible immediately and a claimed improvement must survive comparison with what it replaced.

**Assume the pipeline is chaotic, because it is.** Respawning the cylinder at its rest height instead of the scene's 12 cm free-fall (an *identical resting position*) flips the nominal run from success to failure. **No single-scene result from this pipeline means anything.**

**Ship a behaviour-preservation switch.** Every milestone's change defaults off and must replay the previous milestone's numbers seed-for-seed. That check caught real bugs three times, including a config override that had silently never applied.

## 3. The four things I did not expect

### 3.1 The working grasp was exploiting a bug

The grasp had been hand-tuned until it captured, and documented as a "side grasp approached from above on a descending diagonal". It was nothing of the kind.

The arm's ramp clipped **per-joint** deltas, so joints with small deltas finished early and the palm traced an uncontrolled arc, **bowing 4–14 cm above** the straight line and descending *onto* the cylinder. Flying the same waypoints straight dropped capture from **7/25 to 2/25**. The tuned constants encoded an artefact of a rate limiter.

Making the shape explicit (standoff behind and above, horizontal traverse, vertical descent) took capture to **19/25**, because the geometry could then be *derived* from mesh vertices: the finger cage cannot be entered from the front at any tilt (index and middle surfaces sit inside the cylinder's radius), the vertical corridor over the axis is empty, and the apex the accidental arc had been reaching is computable to 1 mm.

### 3.2 The table was setting the grasp height

The grasp used a zero vertical offset, resting on a documented claim that the pose "sits 3.7 cm above the tabletop by construction". It does not: at **every** feasible cell the pre-curled hand is 3–7 cm *inside* the tabletop. The arm was being physically stalled by the table, and that stall was setting the height. Two load-bearing constants, in consecutive milestones, turned out to be physical accidents. A third followed later: a corrective-leg lateral error that lands on the tuned nominal *by coincidence*, so correcting it costs successes.

### 3.3 Motion was stabilising the grasp

At high friction the cylinder was being lost about a second **after** the arm stopped moving: palm displacement 0.5 mm, pelvis 0.1 mm, command converged. Not a jolt: an 8.9 g cylinder held by a pinch with **10–20× its weight of tangential capacity**, with contact torque growing 30× in half a second under a stationary hand. During the ramp, the 8 mrad command staircase had been *dithering* the contact.

So outcomes track **how long the object is held still**: shorter settles are better, the opposite of the intuition the pipeline was built on. The fix was a per-finger bound on each finger's command relative to its own *measured* angle, which no single grip scalar can express.

### 3.4 The instruments lied, three times

- A grasp grid spanning a quarter of the real offset range read **76%** where the honest number was **35%**.
- Its `captured` metric was blind to the mis-seat that decides the rest of the episode. At one friction it read 13/34 captured and **0/34 seated**.
- Worst: **`seated` turned out to be anti-correlated with end-to-end success** (Spearman −0.8 across four policies, against +1.0 for a post-carry `handed_over` criterion). A policy with the second-best grid score ever measured placed **dead last** end to end.

## 4. What I got wrong, and how I found out

Every row is a claim I made or inherited, and the measurement that killed it. The brief asks about iteration; this table *is* the iteration.

| Claim | Reality | Found by |
|---|---|---|
| Base reaches the stance to 1.7 cm | True only from the nominal spawn; off-nominal, 5.8–17.6 cm | Randomized sweep |
| Arm reaches 0.25 cm after correction | 0.57 cm mean / 0.91 max over the real workspace | Workspace-wide sampling |
| Removing the sim-only gravity cheat is an accuracy win | Algebraically **identical**: a legitimacy refactor, zero dynamics change | Doing the algebra afterwards |
| Both fingers land only within ±0.8 cm of centre | Grips with **lower finger and thumb**; real tolerance ~4 cm | Mesh-vertex measurement |
| The nominal run was a success | A **lucky drop**: `contacts 0` at pre-place; it had already left the hand | Reading telemetry I had already printed |
| Capture is governed by grasp **tilt** (0/10 above 25°) | **A confound.** The axis is pinch *height*; 15° was merely the only tilt ever reached, and my proposed fix would have discarded the best candidate | Controlled replay at fixed geometry |
| The friction tails are one mechanism, fixable by feedback on insertion depth | The pre-close state is **friction-invariant**, so there is nothing to measure there. Two different failures | Instrumenting depth across friction |
| The carry loses it in a settle | **Zero** losses in the settles it shortens; 30 of 50 are mid-ramp in the cradle roll | Per-drop census |
| The cradle roll is the problem; remove it | It is **load-bearing**: removing it costs 24 successes as losses move downstream and multiply | Actually disabling it |
| `out_of_reach` is a systematic undershoot; bias it out | The bias is real but the cause is a **dead attractor** in the walker, frozen for 16 s while commanded 0.397 m/s. The bias fix works exactly as designed and **costs 5 successes** | Signed leg telemetry |
| **The cylinder's friction and mass are part of the task's randomization** | **They are not.** `scene.xml` fixes `friction="3 0.1 0.01"` and `density="100"`, unchanged since the fork. The ranges were **mine**, invented with the eval harness, and μ = 3.0 is the band the pipeline handles *best*, so 76% of test episodes came from friction the task never presents. At shipped physics the same tree scores **90.7%**, not 43.3% | The user asking why μ was changeable at all |
| The three-seed figure is the success rate | **It absorbed the seeds' luck.** Every milestone was developed *and* verified on seeds 0–2; at nine seeds the same pipeline reads **94.4%**, not 97.3%, and seeds 3–8 alone give 93.0% | Confirming one milestone at n = 450 |
| The un-tuck clears the blue tabletop by 0.5 cm | **That figure was never measured.** Recorded live it is **negative on 289 of 289 un-tucks** (median −3.23 cm); what keeps the cylinder up is the arm's *down-reach limit*, +4.15 cm above the commanded pose. And 56% of un-tucks scrape the table while holding, yet **458 of those 471 succeed** | Instrumenting the clearance instead of trusting a note |

**The last row is the most important thing in this document.** I built a randomization the task never asked for, never labelled it as a stress test of my own choosing, and then spent five milestones (M1.9–M1.13) attacking "the friction tails" it created, concluding along the way that the project's target was probably unreachable because of them. The work those milestones produced was real physics and it lifted the shipped-physics path too, which is why that path reads 90.7% rather than the 20% it started at. But the framing was wrong for hours of effort, and the fix was to read a file I should have read on day one. **A randomization is a choice, and it has to be labelled as one.**

Two others deserve emphasis. The **lucky drop** is the least comfortable: the evidence was in output I had generated and displayed, and I read past it.

The **tilt confound** is the methodological lesson. I found a correlation so clean it looked like a mechanism (one variable, ten episodes, zero exceptions), and it was an artefact of which candidates the search ever reached. When a variable correlates perfectly with success, look for the quantity it happens to move, and run the controlled replay before believing it. That error would have cost real capability.

## 5. Results

**At the physics `scene.xml` ships with**, 150 episodes per row, three seeds:

| preset | what varies | end-to-end |
|---|---|---|
| `shipped` | robot spawn ±15 cm/±15°, cylinder within 10 cm of nominal | **146/150 = 97.3%** |
| `shipped_table` | same, cylinder **anywhere on the tabletop** | **134/150 = 89.3%** |

Success by cylinder distance from nominal on the harder preset, which is the clearest single measure of what the last two milestones bought:

| distance | before approach-side search | now |
|---|---|---|
| 0–10 cm | 9/10 | 11/12 |
| 10–20 cm | 14/19 | 23/26 |
| 20–30 cm | 5/23 | 28/32 |
| **30–90 cm** | **0/48** | **72/80** |

Remaining failures across both presets: 7 unreachable stances, 5 tipped at the place, 5 never captured, 3 lost in the carry. M3.3 showed that six of the twelve previously counted as "lost in the carry" were never carry losses at all: the place descent stopped 1.4–3.4 cm above the tabletop and the fingers opened on a cylinder in mid-air.

### The ladder

Measured on the harder self-chosen preset (friction 1.5–4.0, mass 0.6–1.6×) for the middle of the project, because that is what those milestones were developed against (see §4 for why that was a mistake).

| Milestone | What it changed | Success |
|---|---|---|
| Baseline | (none) | 0% |
| M1.5 | Feasible *basin* instead of a feasible point | `out_of_reach` 7/10 → 0 |
| M1.6 | Cartesian palm paths; multi-start IK | not measured |
| M1.7 | Derived standoff→traverse→descent grasp | 3% |
| M1.8 | Compliant release | 23% |
| M1.9–M1.14 | Pinch-height ceiling, per-finger torque bound, friction-keyed carry grip, leg budget | 43.3% |
| M2–M2.3 | RL per-finger grasp policy; friction-gated arm switch | 52.0% (p = 0.0066) |
| **M3.1** | **Approach side as a search dimension** | `shipped_table` 28% → 76.7% |
| **M3.2** | **Pre-lift clear of the tabletop before traversing** | `shipped_table` → **89.3%**, p = 3.1e-4 |
| **M3.3** | **Grasp-tilt preference, measured instead of assumed** | `shipped` → **97.3%**, p = 9.8e-4 |

### The RL result, honestly

The policy's thesis (that the friction tails need per-finger force control no grip scalar can express) held: median per-finger closure spans **0.90** between tightest and slackest finger, the low-friction bucket that had been zero since the grid existed went to 12/34, and the spec's capture acceptance (≥90% at ±10 cm) is met at **92%**. A friction-gated switch between the scripted and learned grasps is worth +10 episodes at **p = 0.0066** over 300 episodes.

But **on the task as shipped the RL is not needed**: friction is fixed at μ = 3.0, which the scripted grasp handles well, and the 90% figures above are the scripted pipeline. The policy's value was against a distribution we invented. Total RL cost was ~9 M steps in about three hours on CPU.

### The hierarchical result, and the measurement that closed it

The second learning branch put an RL policy *above* the provided policies rather than beside them: a 21-D action at 10 Hz (a gate selecting one of three body experts, a walk command, a crouch height, a 6-D reach setpoint and seven finger targets) over the frozen walker, rotator, croucher and reacher. The thesis was that the provided policies are a fixed asset and the learning belongs in the *composition*.

**It reached 1.5% (3/200) against the scripted pipeline's 94.4%, and it is now closed with a reason rather than a plateau.**

Thirty-two hypotheses, ~45 evaluation sweeps. Every channel the layer owns was tried and each one trades one failure for another: the crouch schedule buys a clean grasp (pre-liftoff tip 22.8° → 3.3°) and costs the carry (s3 entries 48 → 2); slewing the reach fixes the knock (35 → 21) and triples falls (8 → 36); holding the grip fixes the tip (35 → 16) and costs the drop (21 → 30). Five separate attempts on the finger channel destroyed the grasp. Raising the decision rate to 50 Hz (implemented, retrained, evaluated) scored 0/200.

The invariant underneath all of it: **successes = conversion × s3 entries**, and interventions move entries (97 → 123 pooled, replicated on two seeds) while conversion never moves. It sat at ~6% in every arm.

**Why, measured directly.** Three candidate fixes were specced to a stated premise and each was falsified *before* it was built:

| design | premise | the measurement |
|---|---|---|
| grip-force floor | force decays through the carry | force at loss **0.0235** exceeds force at lift **0.0193**; it decays *more* in successes |
| orientation servo | the reacher fails to track a good ask | the **ask** is 38.4° off the cylinder against 28.5° achieved, so tracking error is *compensating* |
| command rewrite | a better ask gives a better grasp | `in_palm` floors at **21–25°** on all three rotational axes; authority 0.17 |

The last one is the answer. `in_palm`, the angle between the palm axis and the cylinder's, is **29°** where the scripted pipeline's IK achieves **2.74°**, and perturbing the commanded orientation by ±40° on roll, pitch and yaw never brings it below **21.2°**. The decisive cell is yaw +30°, which produced the *best commanded alignment of any condition* (33.4° against the baseline's 39.5°) and a **worse** achieved grasp (28.9° against 26.3°). Improving what you ask for does not improve what you get.

I first read that as the frozen reacher's manifold excluding the geometry, and **that was wrong**. Swapping the arm to the scripted pipeline's own IK (same action space, only the executor changes) leaves `in_palm` at **26.0°** against the reacher's 26.3, while `knocked` more than doubles (38 → 92) because IK drives straight to the commanded endpoint and sweeps the cylinder off the table. An executor that reaches 2.74° in the pipeline reaches 26° here.

That reading was wrong too, and the next measurement says why. The pipeline's own preferred grasp frame is **20° off the cylinder axis**, not 2.74°. The 2.74 is measured at the *end of s2*, after the close and lift: the pipeline's own record has `in_palm` going **25.1 → 2.74 across the lift**, when its grip cap collapses blocked deflection to 0.02. So at the moment of grasp the pipeline sits at ~25°, indistinguishable from the HL's 26–29°.

**The grasps match. The lifts are opposite**: the pipeline's cylinder settles onto the palm axis by 9×, the HL's rolls off it by 2× (26 → 48.7). Arming the pipeline's own grip cap on the HL reproduced its load collapse almost exactly (0.11 → 0.02) and the cylinder *still* rolled off, because the pipeline's lift is compliant **and straight up from a stationary base**.

Which led to the finding that overturned this project's most load-bearing claim. Throughout, I had recorded that *the walker gaits at 0.106 m/s with every velocity component zeroed, and only the croucher stands still*, the reason the body channel was declared closed. Holding a true zero command and measuring the decay:

| sustained zero command | base speed |
|---|---|
| 0–1 s | 0.0166 m/s |
| 3–6 s | 0.0004 |
| 6–30 s | **0.0002** |

**The frozen walker settles to a dead stand in 3–6 seconds**, at the pipeline's own 0.0003 regime, and unlike the croucher it leaves the arm completely free. The 0.106 figure was the 0–1 s transient; every window this project ever zeroed WALK over was too short to see the settle. The body channel was open the whole time, and the pipeline's stillness during the squeeze is not a different capability; it is the same walker, having stood long enough.

And the HL still cannot spend it. Three packagings (croucher schedule, dwell, dwell with the reach frozen) each delivered the still base (45 of 91 episodes settling at **0.00309 m/s**), and each lost more downstream than it gained, because the dwell has to happen *before* the hand reaches the cylinder and this policy approaches and contacts in one motion. Waiting at contact with an open hand ejects the cylinder (`ejected` 18 → 66).

So the requirement is not a channel a skill can write. It is a **behaviour**: separating approach from contact in time. Conversion is pinned at 6% because every intervention adjusted a grasp that was never the discriminating variable: force, handover tilt and `in_palm` are all **identical in successes and failures**.

**What this cost and what it bought.** The branch did not improve the task. It produced a falsification with a mechanism, three bugs in my own instrumentation found by disbelieving clean-looking nulls (a skill that wrote a forward creep while claiming to stop the base; two missing imports that would have read as ordinary failures), and one methodological result I would keep: *a correlation strong enough to explain 39% of the data can still have the causal arrow backwards.* Both of the conversion correlates I found, dwell over the seat (16× spread) and proximity to it (10×), were symptoms, and building each one made the number worse.

The premise is falsifiable and it is false for this policy, though not for any of the three reasons I recorded along the way, each of which the next measurement overturned. What remains is not a better channel or a better executor: it is a policy that stops before it touches. The scripted pipeline gets its 94.4% by separating approach from contact in time, and that separation is the one thing a skill layered over a policy cannot impose.

### The command-level HL, and the direct HL

The third learning branch kept the scripted pipeline as the actuator and put the policy where the script makes *decisions*: it issues commands through one seam (`ik.pipeline.COMMANDER`) and never moves a joint. All of it is measured on `hl_full`, the HL's wide randomization (spawn ±30 cm / ±45°, both tables moved ±15 cm, tops 0.60–0.80 m and 0.50–0.72 m, footprints scaled ±25%), where the script scores **90/200 (45%)**, not the 94.4% of the shipped task. Seed 3 is held out from every fit.

**Commanding the script's choices adds nothing.** A model choosing the script's pick and place stances, crouch depths and retry budget, falling back to the script unless it was clearly better, scored **91/200**, but **91 vs 93** against the script given the same retry budget. Its own stance choices lost in all three evaluations they were tried in. The script's replanner corrects any stance it is handed, so the stance carries almost no signal, and the model's skill came from recognising hard scenes rather than from knowing which command helps.

**The direct HL owns the pick.** The replanner is skipped: no feasibility search, no corrective legs, no IK-residual abort and no script default. The HL commands a stance relative to the cylinder and a crouch, then, from where the walk actually landed, a grasp (tilt, curl side, approach azimuth), or it declines and commands a new stance (at most twice). Two small MLP ensembles score P(success) for the stance and for the grasp; trained on 2,230 episodes (1,000 uniform, then six rounds of Thompson-sampled on-policy collection: 18.5% → 31.5%).

| seed 3, same 200 scenes | script | direct HL |
|---|---|---|
| success | 90 | **83** (p = 0.51) |
| lifted | 147 | 139 |
| success once lifted | 61% | 60% |
| median wall time per episode | 121 s | 71 s |

What mattered:

- **Label success, not the lift.** Lift-trained grasps lifted 44% but carried badly (45% of lifts converted): tilt and azimuth set the in-grip pose the carry inherits.
- **A free argmax exploits a weak model.** The first unconstrained policy scored 24%; candidates drawn from the exploration distribution and from stances that lifted before fixed it.
- **A calibrated model can correct itself.** Best grasp p < 0.20 succeeded 2 times in 40 held out; re-standing there succeeded 9 of 18.

Of the 61 scenes the direct HL fails to pick, 47 were picked by some other controller version; **14 were never picked by anything, all with the brown top at 0.601–0.633 m**, the lowest of the randomization. The remaining gap is landing accuracy: the walk stops a median 17 cm from the commanded stance in failed picks, 10 cm in successful ones, and the script's corrective legs recover from that where one shot does not.

## 6. What's next

1. **Recover what the shared prefix costs.** The 400 ms probe needs finger contact to read friction, and it destroys the policy's advantage exactly where the policy is strongest: at μ 1.5–2.0 the switch scores the incumbent's 11 where the policy alone scores 16.
2. **Give the terminal reward the depth window back.** A rebalanced run produced the best capture ever measured (13/150 never-captured) and 91 drops, because the dense shaping had been silently carrying a constraint the terminal reward stopped encoding.
3. **μ ≥ 3.5 turned out to be a MASS band, and it is the ceiling.** M2.3 split it on the sweep's other draw: `mu >= 3.5` with the cylinder *lighter* than nominal is 8/24 = 33%, and heavier than nominal is **0/44 over 300 episodes**: 14.7% of the distribution that no arm has ever won. Thirteen scripted settings at 0/15 on the grid, an arm trained on nothing but those episodes at 0/26, a deeper pinch aim at 0/26, and the reward rebalance all score zero. It is not a weight limit (the same mass lifts 12/12 on a slippery grip) but a *cage-versus-pinch* limit: at high friction the fingers cannot slide the cylinder in, so it is held by a fingertip pinch that the lift then squeezes out; 27 of the 44 leave the hand past 7 cm of in-palm depth while the palm rises. **Everything else working caps at 85.3%** of *our* distribution: `scene.xml` ships the cylinder at μ 3.0 and the design mass, so the friction and mass ranges that create the cell are in our own presets, and at the shipped physics the incumbent is already at 88%.

## 7. Honest limitations

- **50% is not a solved task.** The pipeline is not robust; it is *diagnosable*, which is a different thing.
- **The headline gain is now established, and that took episodes rather than ideas**: +20/300 at p = 0.0066, with seeds 3/4/5 fully held out and the threshold never refitted. At 150 episodes it was p = 0.087 and honestly reported as suggestive.
- **90% is not reachable without the mass cell.** 14.7% of the distribution (high friction *and* heavier than the design cylinder) has never produced a success on any of eight arms, so a perfect everything-else caps at 85.3%. Any plan for 90% has to say what it does about that cell first, and has to name which distribution it means, because the cell is created entirely by randomization ranges we chose ourselves.
- **Every cheap instrument in the project was blind to it.** The static grid never once ran a mass other than 1.0 in 375 recorded runs; the validation bank reports 79% of the cell's captures surviving the cradle roll where the sweep scores none of them; and the close's own friction observable, which the switch is built on, cannot see mass at all (|rho| ≤ 0.08). That is the honest reason a bucket sat unmoved for six milestones.
- **The `spec` preset is much harder and partly confounded**: it scatters the cylinder across the whole 80 cm tabletop, and a cylinder 68 cm from nominal needs a different approach *side*. Most of its aborts are "no feasible stance anywhere", which no stance fix reaches.
- **The RL policy is trained on descent → close → lift only.** Curriculum stages for holding under base motion and for the place were specified and never trained.
- **Cameras are unused.** The brief supplies head and wrist cameras; this solution is state-based. Vision was scoped as a later distillation step and not reached.

## 8. Where things live

| Path | What |
|---|---|
| `ik/pipeline.py` | The pure IK pipeline end to end (`run_once`), and the commander seam |
| `ik/sim.py` | Simulation runner, arm IK, Cartesian paths, the walk |
| `ik/base.py` | Base-pose primitives over the walker, rotator and croucher |
| `hl/commander.py` | The direct HL: stance, crouch, grasp and re-stance decisions |
| `hl/train.py` | The per-stage success models; `hl/models/direct.pt` is the trained HL |
| `eval/sweep.py` | Randomized episode harness and failure taxonomy, for both |
| `tests/` | Episode-for-episode regression tests |

The full history (every measurement, the RL hand policy, the 10 Hz HL, the probe scripts, the instruments and every recorded run) is on the `dev` branch.

Run one episode: `python -m ik.pipeline`. Sweep the IK pipeline: `python eval/sweep.py -n 200 --seed 3 --preset hl_full -j 8`. The same sweep with the direct HL: prefix it with `G1HL_COMMANDER=direct:hl/models/direct.pt`.
