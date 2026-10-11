# Push-Recovery Training and Fall Detection for RL Bipeds

Context: 8-DOF biped, 43 cm, 1.75 kg, CoM height about 0.25 m, 3.4 Nm hobby servos, BNO055 IMU, PPO in MuJoCo at 50 Hz. It currently falls in about 50% of episodes under domain randomization plus random 0.3 m/s velocity pushes every 2-4 s.

Method note: PDF text could not be extracted for several primary sources (Kalyanakrishnan & Goswami IJHR 2011, Yang et al. Humanoids 2018 PDF, ROMELA DARwIn-OP paper). Their numbers below come from ar5iv/HTML versions, the abstract, or the related Honda patent, as marked. Calculations labelled "computed" are my own arithmetic from the cited formulas.

---

## 1. Push-recovery training: curricula, magnitudes, rewards

### Takeaway
Published RL push-recovery work samples pushes randomly every 3-5 s. The pushes are either short impulses (20 ms) or longer forces (0.1-0.5 s), sometimes with an adaptive curriculum that keeps magnitudes near the edge of what the policy can currently survive. Ramping the magnitude naively over the episode, or starting with very large pushes, has been reported to hurt learning. Normalized by size, a 0.3 m/s push on a robot with h = 0.25 m is about as hard as the largest push Valkyrie rejected *without stepping* in Yang et al. So the current setup requires reliable stepping, which partly explains the 50% fall rate.

### Cited Findings
**Push schedules and magnitudes**
- Yang et al. (Valkyrie, 137 kg, CoM 1.1 m): horizontal pushes on the pelvis with impulses from about 26.5 to 106 Ns (53×0.5 to 53×2 Ns), random direction and magnitude, **5 s between pushes**. Policy at 25 Hz, PD at 500 Hz. — [Yang et al., Learning Whole-body Motor Skills for Humanoids (ar5iv)](https://ar5iv.labs.arxiv.org/html/2002.02991)
- Yang et al. results: maximum impulse rejected without stepping (sagittal/lateral) was TRPO 240/78 Ns, DDPG 75/160 Ns, PPO 192/36 Ns, against 53/78 Ns for a capture-point baseline. Normalized impulse was 0.57 Ns/kg (sagittal, no step), 0.56 Ns/kg (lateral) and **1.73 Ns/kg with stepping**. The algorithm affected lateral performance a lot (PPO was weakest laterally at 36 Ns). — [Yang et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2002.02991)
- Atalante exoskeleton (Duburcq et al.): pelvis pushes of up to **800 N peak, 400 ms bell-shaped profile**, direction sampled from a sphere, **every 3 s with 2 s jitter**. Ramping force from 0 to 800 N over the episode "led to premature convergence to a suboptimal policy". Policy at 25 Hz, PD at 100 Hz, 100M steps (about 6 h on 40 workers). Maximum recoverable sagittal impulse was about 190 Ns, against about 240 Ns for Yang et al. — [Duburcq et al., Reactive Stepping ... Atalante (ar5iv 2203.01148)](https://ar5iv.labs.arxiv.org/html/2203.01148)
- Digit (van Marum et al., IROS 2024), "Single Contact RL": each 50 Hz frame has a **1% chance** of a push (on average one every 2 s), **200-800 N for one 20 ms step**. The improved "Single Contact++" used **20-200 N for 200-500 ms in all 3 axes**. The benchmark used 79-214 N for 200-500 ms in ±X/±Y; 214 N is about half the robot's weight. Single Contact++ passed **258 N for 500 ms in all directions**. — [van Marum et al., Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking](https://arxiv.org/html/2404.19173v2)
- Kumar (Georgia Tech thesis), COMAN model (31 kg, 0.9 m, 35 Nm torque cap): pushes on the torso for **0.2 s**. Postural (ankle/hip) controller pushes 1-51 N in all directions; stepping controller pushes 41-81 N, forward only (−π/4 to π/4). — [Kumar, Learning Control Policies for Fall Prevention and Safety in Bipedal Locomotion (ar5iv 2201.01361)](https://ar5iv.labs.arxiv.org/html/2201.01361)
- **Adaptive "Region of Attraction" curriculum** (Kumar): the RoA is the set of push magnitudes per direction that the policy survives with at least 90% success. It is stored as a polar polygon (8 sides for postural, 3 for stepping). Each episode samples a direction uniformly and a magnitude in [k_l·f(θ), k_u·f(θ)], with k_l = 0.8 (postural) or 0.7 (stepping) and k_u = 1.1. **Every 4 iterations** each direction's boundary grows if success > 0.9 and shrinks otherwise. Initial boundaries were 5 N (postural) and 41 N (stepping). Recovery from backward and left pushes improved by about 50% and 38% over baselines. The thesis reports that very large pushes early hurt learning and very small ones teach little. — [Kumar thesis (ar5iv)](https://ar5iv.labs.arxiv.org/html/2201.01361); [search summary of same](https://arxiv.org/pdf/2201.01361)
- Honda fall-prediction data set: pushes of uniform 0-500 N for **0.1 s**, horizontal, direction uniform over 360°, applied −0.05 to +0.25 m relative to the CoM, about 1,000 pushes on a 42.1 kg simulated humanoid (CoM 0.59 m). — [US8554370B2 (Goswami & Kalyanakrishnan)](https://patents.google.com/patent/US8554370)
- NAO (learning-based push recovery) succeeded with an impulse of 22 Ns (secondary summary; full text not read). — [Experimental study ... NAO (IEEE)](https://ieeexplore.ieee.org/document/8515159/)

**Reward terms used for recovery**
- Yang et al.: r = r_pose + r_CoM_pos + r_CoM_vel + r_GRF + r_contact + r_power, with shaped terms exp(−α(x_target − x)²). Torso/pelvis roll and pitch target 0. The CoM horizontal target is the support-polygon centre and the vertical target is a fixed height. The **CoM horizontal velocity target is derived from the capture point**, applied only while a foot is in contact. GRF target is half body weight per foot. Penalties are −2 if no foot is on the ground and −10 if the upper body touches the ground. Power penalty is Σ|τ·q̇|. Weights are not given in the HTML. — [Yang et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2002.02991)
- Duburcq et al. (Atalante): r = Σ w_i·exp(−κ_i c_i²), with terms for odometry drift, reference posture, relative foot pose, **foot placement pulling the pelvis toward the capture point**, **ZMP inside the support polygon**, balanced vertical forces (sum ≈ mg), friction (tangential force), and pelvis angular velocity. Weights are not listed. — [Duburcq et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2203.01148)
- Kumar: balance reward r = −w_p‖C̄ − C‖² − w_d‖Ċ‖², with w_p = 1.0 and w_d = 0.1 (postural) or 0.5 (stepping), i.e. CoM position error plus CoM velocity. — [Kumar thesis (ar5iv)](https://ar5iv.labs.arxiv.org/html/2201.01361)
- Poddar et al. (2026, Unitree H1-2): capture point, CoM state and centroidal momentum are used as **privileged critic inputs and shaping rewards**, while the actor uses only proprioception. A single policy shows ankle, hip and stepping strategies, with a 93.4% recovery rate. Ablating the balance-informed structure broke learning (for stand-up). — [Poddar et al., Embedding Classical Balance Control Principles in RL for Humanoid Recovery (arXiv 2603.08619)](https://arxiv.org/abs/2603.08619)
- Digit Single Contact reward weights: x/y velocity 0.15 each, yaw 0.1, roll/pitch orientation 0.2, single-foot contact 0.1 (0.2 s grace period, constant when standing), sparse feet airtime 1.0, plus small style terms (base height 0.05, foot orientation/position 0.05, base acceleration 0.1, action difference 0.02, torque 0.02). Task terms alone produced hopping. — [van Marum et al.](https://arxiv.org/html/2404.19173v2)

### Inferences
- **Normalization (computed).** Using Δv/√(g·h) as a dimensionless push size:
  - Valkyrie's best no-step rejection, 0.57 m/s ÷ √(9.81×1.1) = 3.28, gives ≈ 0.17.
  - Valkyrie with stepping, 1.73/3.28, gives ≈ 0.53.
  - The user's 0.3 m/s ÷ √(9.81×0.25) = 1.57 gives **≈ 0.19**, so it is at or above the ankle/hip-only limit and needs stepping.
  - A 0.3 m/s push on 1.75 kg is an impulse of 0.525 Ns (computed).
- A Region-of-Attraction-style adaptive curriculum maps directly onto velocity pushes:
  - Keep a per-direction (e.g. 8 bins) max Δv.
  - Sample in [0.7-0.8, 1.1] × the boundary.
  - Expand or shrink each bin on a 90% success rule.
  - This avoids both wasting samples on unrecoverable pushes and the "ramp" failure reported for Atalante.
- An instantaneous velocity kick (MuJoCo qvel change) is an idealized impulse. Several papers (Digit ++, Atalante, Kumar) instead use 0.2-0.5 s force pulses, which are physically more realistic and give the policy time to react during the push. A mix of both is reasonable.
- Push intervals in the literature (3-5 s; on average about 2 s for Digit with very short pulses) are similar to the user's 2-4 s. With 2-4 s spacing, though, a second push may land before recovery steps finish. A minimum gap of about 1.5-2 s of "settled" state could isolate the effect.
- Cheap reward additions for this robot: an ICP-based term (e.g. exp(−‖ξ − p_support‖²/σ²), with ξ = x_CoM + v_CoM/ω) in the critic or reward, and torso roll/pitch and angular-rate terms. Moving CoM/ICP to privileged critic inputs matches Poddar et al. and Yang et al.

### Gaps
- Numeric reward weights for Yang et al. and Duburcq et al. could not be extracted (PDF text unreadable here).
- No RL push-recovery paper on a robot as small as 1.75 kg with reported push magnitudes was found. The DARwIn-OP papers (ROMELA) exist, but their PDF could not be read.

---

## 2. Ankle, hip and stepping strategies; gait clocks vs free recovery steps

### Takeaway
A single RL policy can learn ankle, then hip, then foot-tilt, then stepping strategies in order of push size, without hand-coding. Stepping emerges when the reward makes the capture point/CoM over the support polygon pay off. **Clock or periodic-contact rewards and double-support rewards are reported to suppress recovery steps.** The most direct evidence comes from Digit: a clock-free "single contact" reward made the policy move its feet freely, and clocks hurt standing and recovery. Alternatives are to let the policy modulate the clock frequency, or to drop the clock entirely.

### Cited Findings
- Yang et al.: the strategy depends on push size.
  - At 24 Ns: ankle and hip.
  - At 72 Ns: adds foot tilt.
  - At 240 Ns: adds stepping (8 steps).
  - At 78 Ns lateral: ankle, hip and foot tilt.
  - Stepping was **not hand-programmed** and emerged from reward plus random push exploration.
  - Lateral stepping was limited because the stance leg blocks the swing leg.
  - — [Yang et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2002.02991)
- The follow-up on iCub reports four human-like strategies emerging naturally: ankle, hip, toe and stepping. — [Yang et al. search summary (IROS 2020)](https://ras.papercept.net/images/temp/IROS/files/3133.pdf)
- Duburcq et al. use **no gait clock**. Stepping is reactive from instantaneous proprioception. Capture-point foot-placement and ZMP rewards make stepping happen when needed, while smoothness regularization plus termination conditions keep stepping as a last resort. — [Duburcq et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2203.01148)
- van Marum et al. (Digit) on clocks:
  - The clock-free policy handles standing, walking and recovery steps internally, with no standing/walking switching.
  - The authors argue that **clocks make standing awkward and push the policy toward low foot velocity, which hampers disturbance recovery**.
  - They argue that **double-support rewards penalize recovery steps** and bias transitions toward the nearest stance. Their single-contact reward therefore returns a constant when standing.
  - Single Contact RL beat the clock-based controller in X pushes.
  - The clock-based controller showed discontinuities at standing-walking switches.
  - — [van Marum et al.](https://arxiv.org/html/2404.19173v2)
- The periodic-reward-composition framework (Siekmann et al., Cassie) builds rewards from probabilistic periodic costs on foot forces and velocities; this is the "clock reward" that van Marum et al. compare against. — [Siekmann et al., Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/pdf/2011.01387)
- Duan et al. (Cassie stepping stones) let the policy **output the clock period** (gait cycle limited to 0.65-1.2 s, updated at 40 Hz) to adjust to sudden step-size changes. — [Duan et al., Learning Dynamic Bipedal Walking Across Stepping Stones](https://ar5iv.labs.arxiv.org/html/2205.01807)
- BITeno (Biomimetics 2024): a separate learned "stimulus" policy sets **left/right clock frequencies** online from robot state. The authors note that fixed-frequency periodic signals make it hard to learn an adaptive gait, and report recovery within 1.5-2.0 s after a −1.2 m/s torso-velocity change. — [Adaptive Gait Acquisition through Learning Dynamic Stimulus Instinct (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11201531/)
- Model-based evidence: fixed slow gait frequency caused large roll oscillation and an excessive number of steps to recover, and an adaptive frequency law reduced attitude error. — [Adapting Gait Frequency for Humanoid Push-recovery via Hierarchical MPC](https://arxiv.org/html/2409.14342)
- Capture Steps (Missura/Behnke, RoboCup) adapt **both step timing and foot placement** from a LIP fit to the gait.
  - NimbRo-OP2 (134.5 cm, 17.5 kg) at 100 Hz.
  - Step time shrinks by 0.01 s per iteration and can go to 0 (immediate step).
  - Hand pushes: open-loop CPG 22 falls in 58 pushes, capture steps 18 falls in 97 pushes (push strengths not controlled).
  - — [Missura et al., Capture Steps: Robust Walking for Humanoid Robots (ar5iv)](https://ar5iv.labs.arxiv.org/html/2011.02793)
- Egle et al. use RL to adapt step timing and the allowed step region (including cross-over steps for big lateral pushes) on top of a model-based step controller. — [arXiv 2411.01000](https://arxiv.org/pdf/2411.01000)

### Inferences
- With a fixed clock, a push arriving early in stance may have to wait up to about half a gait cycle before the swing foot is "allowed" to move. For this robot the ICP diverges with time constant 1/ω ≈ 0.16 s (computed), so a 0.1-0.2 s wait already roughly doubles the ICP error. This is a plausible large contributor to the 50% falls if the policy uses a rigid phase-contact reward.
- Practical options, from least to most invasive:
  1. Down-weight or zero the clock/contact-schedule reward for about 0.5-1.0 s after a push, or whenever |ICP − support centre| exceeds a threshold. This is an inference by analogy; no paper found that does exactly this.
  2. Let the policy output a phase-rate (frequency) offset, as in Duan et al. and BITeno.
  3. Remove the clock and use single-contact plus airtime rewards (van Marum).
- Lateral pushes are the hard case for small 8-DOF bipeds (no hip yaw in some designs, and the stance leg blocks the swing leg, per Yang et al.). Check per-direction fall statistics.

### Gaps
- No paper found that explicitly "relaxes the clock reward during recovery" with reported numbers. That suggestion is an inference.
- No numbers found on how much clock-based policies lose in push survival compared with clock-free ones, beyond van Marum's qualitative X/Y comparison.

---

## 3. Separate recovery policy: initial-state distribution and handoff

### Takeaway
The cited works that train distinct behaviours draw initial states from reference trajectories or noisy nominal poses (Atalante, ANYmal) and learn or script the switching. No source found used states harvested from rollouts of the walker under perturbation. For a walker-to-recoverer handoff, that harvesting approach (a reset buffer of states recorded just before predicted falls) is the logical choice, but it is unverified here. Single-policy approaches (Yang, Duburcq, van Marum, Poddar) avoid the handoff problem altogether and are the dominant practice.

### Cited Findings
- Duburcq et al. start each episode from a state sampled uniformly from a set of reference trajectories (standing, walking at various speeds, turning) plus Gaussian noise, so the recovery policy sees walking-like states. — [Duburcq et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2203.01148)
- ANYmal recovery controller (Lee, Hwangbo, Hutter 2019):
  - Three behaviour policies (self-righting, stand-up, locomotion) plus a **learned behaviour selector** at 50 Hz that outputs a categorical distribution over behaviours.
  - The selector observes the previously chosen behaviour and the previous action.
  - Selector training episodes start from the initial-state distributions of randomly chosen behaviours.
  - Locomotion starts from a Gaussian around the standing pose.
  - The hand-crafted FSM alternative waited 0.5 s after self-righting for the state estimator to converge.
  - The authors report >97% success over >100 trials.
  - The paper does not use states from other policies' rollouts.
  - — [Lee et al., Robust Recovery Controller for a Quadrupedal Robot (ar5iv)](https://ar5iv.arxiv.org/html/1901.07517)
- Kumar's thesis relay networks define the initial-state distribution ρ₀ as a Gaussian near the goal state of the next policy in a chain (the excerpt read gives no balance-specific ρ₀). — [Kumar thesis (ar5iv)](https://ar5iv.labs.arxiv.org/html/2201.01361)
- Kumar's safe-fall policies switch at each new contact by picking the actor whose critic value is highest. This is a value-based selection rule. — [Kumar thesis (ar5iv)](https://ar5iv.labs.arxiv.org/html/2201.01361)

### Inferences
- If a separate recovery policy is used:
  1. Roll out the current walker with the full push and randomization distribution.
  2. Log states from the 0.1-0.5 s window after each push (or after a fall-predictor trigger), including the ones that eventually fell.
  3. Start recovery episodes from a mix of about 70-80% buffer states and some nominal states. Refresh the buffer as the walker changes.
- Hand back to the walker once its critic V(s) exceeds a threshold, or once tilt, tilt rate and ICP have stayed inside the walker's "safe" band for several control steps. Lee et al.'s selector uses the previous behaviour as an input, which is a learned form of hysteresis.
- Given the 50 Hz policy and 0.16 s ICP time constant, any detection-plus-switch latency costs a large part of the recoverable window (see section 5). The literature therefore favours one policy trained on pushes over a switched architecture for a robot this small and fast.

### Gaps
- No paper found that explicitly trains a biped recovery policy from walker-rollout reset buffers with reported numbers.

---

## 4. Fall detection and prediction: criteria, thresholds, lead times, debounce

### Takeaway
Learned predictors (decision lists, CNNs) trade lead time against false positives. Reported lead times run from about 0.2 s (the minimum considered useful for reflexes) to more than 1 s for slow faults on large humanoids. Debouncing over a window (all or 80% of recent samples positive) and a high classifier threshold (0.9 instead of 0.5) are the main tools to remove false positives. For a 0.25 m CoM robot, dynamics are about 2× faster than a 1 m humanoid (time scales ∝ √h), so lead times scale down accordingly.

### Cited Findings
- Kalyanakrishnan & Goswami frame the problem as signalling an imminent fall with enough lead time, while minimizing false alarms, so the robot can switch from balance maintenance to fall control.
  - They introduce parameters to trade off false-positive rate against lead time.
  - Several settings beat hand-coded predictors on both measures.
  - The learned predictors are decision lists of depth 5-10 over 16 features, tested on a simulated Asimo-like robot.
  - — [Kalyanakrishnan & Goswami, Predicting Falls of a Humanoid Robot through ML (AAAI/IAAI 2010)](https://ojs.aaai.org/index.php/AAAI/article/view/18815); [Learning to Predict Humanoid Fall (IJHR 2011)](https://www.cs.utexas.edu/~shivaram/readings/b2hd-KalyanakrishnanGoswami2011.html)
- Honda patent from the same work:
  - Features: CoM displacement, linear momentum and its derivative, angular momentum about the CoM and its derivative (15 reals), plus foot-contact mode (1 of 16), sampled at 250 Hz.
  - "Fallen" means a non-foot ground contact or the CoM below a threshold (example 0.33 m on a 0.59 m CoM robot, about 56%).
  - Negative examples are weighted 4×.
  - **Debounce window τ_his** of 0.016 / 0.044 / 0.060 s (conservative / near-zero FPR / balanced). A fall is declared when all, or a set fraction such as **80%**, of the classifications in the window are "falling".
  - Label cut-off τ+ is −0.3 to +0.3 s relative to the onset of height drop.
  - A predictor that never fires has zero false positives and zero lead time, which motivates the trade-off.
  - Numeric lead-time and FPR values were not available in the patent.
  - — [US8554370B2](https://patents.google.com/patent/US8554370)
- Mungai, Prabhakaran & Grizzle (Digit, 2023), 1D-CNN fall predictor:
  - Input: windows of 10 samples at about 40 Hz (≈0.25 s) of CoM position/velocity and sagittal joint angles/rates.
  - Fall label: CoM < 10% of initial height.
  - Average lead time 1.52-1.64 s in simulation at **0 FPR**.
  - Sigmoid threshold 0.5 gave 0.27 FPR at 1.79 s lead; **raising it to 0.9 gave 0 FPR at 1.66 s**.
  - On hardware, mean response time 0.99 s at 0 FPR.
  - **0.2 s minimum useful lead time**, motivated by reflexive algorithms (citing Höhn & Gerth).
  - These are mostly slower fault-induced falls during standing, not pushes.
  - — [Fall Prediction for Bipedal Robots: The Standing Phase (arXiv 2309.14546)](https://arxiv.org/html/2309.14546v2)
- Höhn & Gerth, "Probabilistic Balance Monitoring for Bipedal Robots", IJRR 28 (2009) 245-256. Bibliographic confirmation only; method details not retrieved. — [UT Austin reading list entry](https://cs.utexas.edu/~shivaram/readings/b2hd-HohnGerth2009.html)
- A later patent on biped balance monitoring notes that friction-cone/physics-model fall predictors only work when standing stably without slipping. It also notes that the Kalyanakrishnan-style approach needs CoM and angular-momentum signals that are hard to obtain on hardware. — [US11511431B2](https://patents.google.com/patent/US11511431)
- SafeFall pairs a lightweight GRU fall predictor with an RL protective policy (date and numbers not confirmed). — [search result, arXiv 2309.14546 listing context](https://arxiv.org/abs/2309.14546v1)
- Termination thresholds used in RL training (Atalante): pelvis height > 0.3 (units in the paper), roll ∈ [−0.4, 0.4] rad, pitch ∈ [−0.25, 0.7] rad, and feet hulls kept ≥ 0.02 apart. — [Duburcq et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2203.01148)
- Hysteresis example from a RoboCup gait: after a support exchange, another exchange is blocked until the feet are 55 mm apart vertically (on a 134.5 cm robot). — [Missura et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2011.02793)

### Inferences
- For an IMU-only robot (BNO055), a practical detector is a 2-D threshold in (tilt θ, tilt rate θ̇). Equivalently, use the inverted-pendulum "energy/ICP" estimate ξ ≈ h·(θ + θ̇/ω), with ω = 6.26 rad/s computed for h = 0.25 m. Declare "falling" when ξ leaves the reachable capture region (section 5) for N consecutive samples.
- Debounce scaled from the Honda patent's 16-60 ms windows to a 50 Hz loop is 1-3 samples (20-60 ms) with an 80% rule. Use separate enter and exit thresholds (e.g. enter at 1.0×, exit at 0.6× the boundary) and a minimum dwell of about 0.3-0.5 s before handing back.
- Lead-time expectations scale with √h. Full-size humanoids (h ≈ 1 m) give 0.2-1.6 s, so expect about half that for h = 0.25 m: tens of ms to several hundred ms for push-induced falls. This is a scaling estimate, not a measured number.
- A learned predictor is easy to get in simulation. Label states from the training rollouts with "falls within T s" for T ∈ {0.2, 0.4} and train a small MLP/GRU on IMU plus joint history. Pick the threshold for zero FPR on held-out non-fall rollouts, following the 0.5→0.9 threshold result of Mungai et al.
- The PPO critic value itself is a cheap fall predictor. This follows Kumar's value-based switching; no paper found reports its lead times on bipeds.
- BNO055 fused-orientation latency and its internal 100 Hz fusion rate should be modelled as observation delay in sim. This is a general sim-to-real inference; no source was checked for BNO055 specifics.

### Gaps
- Exact lead-time and FPR numbers from Kalyanakrishnan & Goswami (IJHR 2011) and the Höhn & Gerth method could not be extracted (PDF/abstract unavailable to the tools).
- No published tilt or tilt-rate thresholds were found for RoboCup kid-size robots (NAO, Darwin-OP, OP3) in the sources read.

---

## 5. Capture-point theory: recoverable region for this robot

### Takeaway
With ω = √(g/h) = 6.26 s⁻¹ (time constant 0.16 s) for h = 0.25 m, the capture point shifts by Δv/ω ≈ 4.8 cm for a 0.3 m/s push. That is already at or beyond a typical small-robot foot half-length (3-4 cm). So 0.3 m/s pushes generally **require at least one quick step**. One- to three-step capturability limits the instantaneous velocity change that can be recovered from standing to about 0.3-0.5 m/s. That limit depends strongly on how fast a step can be taken (0.15-0.25 s) and how long it can be (8-12 cm).

### Cited Findings
- Capturability (Koolen, de Boer, Rebula, Goswami, Pratt, IJRR 2012, "part 1") defines N-step capturability as the ability to come to a stop in N or fewer steps. It uses the 3D LIP model with a point foot, a finite foot, and a reaction mass. The 1-step capture region is a segment and the 2-step region is much larger. — [Zaytsev 2013 citation of Koolen et al.](https://www.cmu.edu/dynamic-walking/files/abstracts/Zaytsev_2013_DW.pdf); [Koolen et al. summary via patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8942848)
- ICP dynamics: ξ̇ = ω(ξ − r_CMP), with ω² = g/z₀, and the CMP equals the foot position for a point foot. — [US8942848 (IHMC patent)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8942848)
- Koolen-style analysis ignores swing-leg limits. Later work adds swing-leg dynamics and step-time margins. — [arXiv 1601.06616](https://www.arxiv.org/abs/1601.06616)
- Capture Steps fit the LIP constant C = √(g/h) experimentally per robot rather than using geometric h. — [Missura et al. (ar5iv)](https://ar5iv.labs.arxiv.org/html/2011.02793)

### Inferences (computed with the LIP model; idealized)
- ICP: ξ = x + ẋ/ω. A sudden Δv while standing over the ankle gives ξ = Δv/ω. With ω = 6.264 s⁻¹: 0.3 m/s → 4.8 cm, 0.5 m/s → 8.0 cm.
- Recursion with a finite foot of half-length r, max step length l, and minimum step duration Δt (from push to touchdown): d₀ = r, d_N = (d_{N−1} + l − r)·e^{−ωΔt} + r. The largest recoverable Δv from standing is ω·d_N.

| Δt (s) | l (m) | r (m) | d₁ (cm) | d₃ (cm) | Δv max, 0-step | 1-step | 3-step |
|---|---|---|---|---|---|---|---|
| 0.15 | 0.08 | 0.03 | 6.1 | 7.8 | 0.19 m/s | 0.38 m/s | 0.49 m/s |
| 0.15 | 0.10 | 0.04 | 7.9 | 10.0 | 0.25 | 0.49 | 0.63 |
| 0.20 | 0.08 | 0.03 | 5.3 | 6.1 | 0.19 | 0.33 | 0.38 |
| 0.20 | 0.10 | 0.04 | 6.9 | 7.9 | 0.25 | 0.43 | 0.49 |
| 0.25 | 0.08 | 0.03 | 4.7 | 5.1 | 0.19 | 0.29 | 0.32 |
| 0.25 | 0.10 | 0.04 | 6.1 | 6.6 | 0.25 | 0.38 | 0.41 |
| 0.30 | 0.10 | 0.03 | 4.5 | 4.8 | 0.19 | 0.28 | 0.30 |

- Reading the table:
  - The ankle strategy alone (0-step) caps at about 0.19-0.25 m/s for a 3-4 cm foot half-length.
  - Step speed matters more than step length. Going from Δt = 0.25 s to 0.15 s raises 1-step capacity by about 30%.
  - A 0.3 m/s push from standing is recoverable with one step only if the step lands within about 0.2-0.25 s.
  - Pushes during walking add to the existing ICP offset and use up margin.
- The step time of a hobby-servo robot is limited by servo speed and torque (3.4 Nm). Measure the fastest realistic swing in sim (push to touchdown, including the 20 ms policy period and servo lag) and use it as Δt.
- Lateral steps are typically shorter (leg collision) and should be computed with a smaller l. Lateral pushes are likely the binding constraint.
- Hip/upper-body strategies (CMP outside the foot via angular momentum) add a small margin. With an 8-DOF legs-only robot and little upper-body mass this is limited. This is an inference; Koolen's reaction-mass model quantifies it.
- Suggested curriculum target: train up to about 1.2× the computed 1-3-step limit in the easy directions, and treat pushes above the 3-step limit as unrecoverable. Exclude them, or terminate without heavy penalty, so they do not dominate the fall statistics. At 0.3 m/s the push sits right at the robot's theoretical limit. A 50% fall rate may therefore be partly physical rather than a policy deficiency.

### Gaps
- The exact closed-form N-step formulas were not retrieved from Koolen et al. (2012). The recursion above is derived from the cited ICP dynamics and should be checked against Section 3 of that paper.
- The robot's actual foot dimensions, maximum step length and minimum swing time were not given; the table brackets plausible values.
