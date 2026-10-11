# Architecture choice for biped balance recovery: separate recovery policy + switching supervisor vs. unified robust policy vs. MoE / distillation hybrids

Context: the user has a small-biped RL walking policy with 0% falls under nominal sim physics and about 50% falls under domain randomization (DR) plus 0.3 m/s random pushes. The goal is "staying up" recovery while walking, not getting up off the ground, without degrading the gait.

Research note: about 14 search/fetch calls. Several sources were read only at the abstract or search-snippet level, and this is flagged per item. Items marked "[from prior knowledge, not re-fetched this session]" are well-known papers whose details I did not re-verify here, so the report writer should treat their specifics with some caution.

---

## Q1. Published systems using a separate recovery/balance controller with a switching supervisor, and their hardware results

### Takeaway
Hardware-validated "separate controller + supervisor" designs come mostly from quadrupeds (ANYmal) or from fall *mitigation* (protective falling) on humanoids. For staying up while walking on bipeds, the successful hardware systems (Cassie, OP3, G1/H1, Booster T1) almost all use a single robust locomotion policy whose push recovery emerges from DR and perturbation training. I found no hardware-validated biped system that switches from a walker to a separate learned *stay-upright* policy and reports better fall rates than a single robust walker.

### Cited Findings
**Separate controllers with a selector or supervisor (mostly quadruped or post-fall):**
- ANYmal (Lee, Hwangbo, Hutter 2019): a hierarchical controller of four NN policies, namely three behaviors plus one learned *behavior selector* that coordinates them. Each was trained individually in sim and deployed directly. On hardware it recovered from arbitrary fall configurations in under 5 s, with a success rate above 97% over more than 100 tests. Its scope is get-up after a fall, not balance while walking (abstract only read). — [arXiv 1901.07517](https://arxiv.org/abs/1901.07517)
- Quadruped fall prediction + RL recovery (HIT): an analytical capturability-based predictor reaches up to 95% accuracy about 395 ms ahead, and the system switches to the fall controller about 0.06 s after detection (search snippet). — [HIT Scholar](https://scholar.hit.edu.cn/en/publications/fall-prediction-control-and-recovery-of-quadruped-robots/)
- Older humanoid work frames the supervisor problem as predicting whether the *balance controller* can still prevent a fall. A predictor trigger then switches from balance maintenance to fall control. This used supervised learning (search snippet). — [AAAI, Predicting Falls of a Humanoid Robot through ML](https://ojs.aaai.org/index.php/AAAI/article/view/18815)
- **SafeFall (2025, Unitree G1, 29 DoF):** a GRU fall predictor (1 layer, 64 hidden units, IMU + joint encoders, inference under 0.5 ms) runs alongside the nominal controller. When it flags an *unavoidable* fall, control passes to a protective RL policy trained with PPO and an asymmetric actor-critic. Both run at 50 Hz, with PD at 200 Hz. Results:
  - Hardware: peak impulse in a 100 ms window was 286.1 vs. 367.1 for a damping baseline, a 22.1% reduction.
  - Sim: peak contact force down 68.3% and peak joint torque down 78.4%.
  - Predictor tuning: 0.41 s lead time at a 0.06% false-alarm rate, by masking the ambiguous safe/falling transition region out of the loss. The unmasked baseline reached 1.22 s lead time but a 4.1% false-alarm rate.
  - Real-world: zero false positives in controlled push tests. "Gentle pushes that allowed recovery did not trigger protection."

  Note that this is fall *mitigation*, not staying up. The supervisor deliberately fires only when the walker can no longer recover. — [arXiv 2511.18509](https://arxiv.org/html/2511.18509)
- SafeFall's critique of prior switching work: "Several existing works detect imminent instability and then trigger independent recovery controllers," and those works "rarely address early prediction of unavoidable falls or tightly coupling prediction with control." — [arXiv 2511.18509](https://arxiv.org/html/2511.18509)
- **Standing Tall (2025, Agility Digit, 1.6 m, 48 kg):** a 1D-CNN fall classifier plus lead-time classifier and regressor, transferred sim-to-real zero-shot. Results:
  - Online lead time averaged 1.1 s with a maximum error of 0.03 s (0.2 s required). Hardware online predicted lead time was 1.04 s.
  - The paper explicitly did *not* build a recovery controller. On hardware it fell back to Agility's base controller, which already has fall prevention.
  - Recovery rate was reported only in sim (0.97), "due to the difficulty of replicating the same fault multiple times on the hardware".
  - Hardware false-positive rate is unknown because "the recovery controller is triggered whenever falls are detected."
  - Hardware trials were small: the text says 13 trials, but the breakdown is 8 abrupt + 6 incipient, an internal inconsistency in the paper or the extraction. 4/8 abrupt and 3/6 incipient were classified critical.
  - The robot was on a slack gantry. — [arXiv 2506.01141](https://arxiv.org/html/2506.01141)

**Single robust walkers with emergent push recovery (bipeds, hardware):**
- Cassie (Li, Peng, Abbeel, Levine, Berseth, Sreenath; IJRR 2024): one RL framework uses a dual-history (long- and short-term I/O history) architecture and is trained in sim with zero-shot transfer. The same walking policy shows "Walking (Push Recovery)." The authors credit I/O history for adapting to both time-invariant dynamics shifts and time-variant changes such as contact events. They also credit **task randomization** as a key source of robustness and "compliance to disturbances." There is no separate recovery controller. — [arXiv 2401.16889](https://arxiv.org/abs/2401.16889v1); [Berkeley PDF](https://hybrid-robotics.berkeley.edu/publications/Cassie_RL_Versatile_Locomotion.pdf)
- Its precursor (Li et al. 2021, Cassie) shows a parameterized walking policy recovering from front and back pushes while varying velocity and height. — [arXiv 2103.14295](https://ar5iv.labs.arxiv.org/html/2103.14295)
- Booster Gym (Booster T1): a single end-to-end RL locomotion policy on hardware. When a 10 kg ball hits the robot during stepping in place, it "regains a stable gait within a few steps" (search snippet). — [arXiv 2506.15132](https://arxiv.org/pdf/2506.15132)
- DeepMind OP3 soccer (Haarnoja et al. 2023; Robotis OP3, 51 cm, 3.5 kg, 20 servos, a small hobby-class biped). Without DR and perturbations, zero-shot transfer failed: robots "would fall over with every one or two steps." Random torso impulses during training plus targeted dynamics randomization and high-frequency control fixed this. An upright-torso reward (within 11.5°) was added because forward-leaning gaits made the real robot fall forward. The final policy "reacts to prevent falling in the first instance." — [arXiv 2304.13653 (HTML)](https://arxiv.org/html/2304.13653)
- Hwangbo et al. 2019 (ANYmal, Science) used separately trained locomotion and fall-recovery (get-up) policies [from prior knowledge, not re-fetched this session]. — [arXiv 1901.08652](https://arxiv.org/abs/1901.08652)

### Inferences
- In the literature, a "switching supervisor" is mainly used to change *objective*: get up after a fall, or fall safely when a fall is unavoidable. It is rarely used to stay upright. Staying upright under pushes is treated as part of the walking policy's job and is solved with DR, push curricula, and history/latent inputs.
- For the user's case, about 50% falls at 0.3 m/s pushes most plausibly means the walker was never trained against that distribution. The literature-standard fix is to train the walker itself (or a fine-tuned copy) under those pushes, not to bolt on a recovery policy.

### Gaps
- I found no paper with hardware results for a biped that switches from an RL walker to a separate learned *stay-upright* RL policy while walking.
- Lee et al. 2019: the details of the selector inputs and switching logic were not read (abstract only).

---

## Q2. Direct comparisons of separate vs. unified policies, and known failure modes of switching

### Takeaway
Head-to-head comparisons of "walker + recovery switch" vs. "one policy" on biped hardware are essentially absent. The closest evidence is:
- (a) DeepMind OP3: distilling separate skills into one agent beat "simply composing the skills" (qualitative).
- (b) Quadruped multi-skill work: abrupt switching between separately trained experts fails from distribution mismatch, and learned or blended transitions fix it.
- (c) Unified humanoid fall-safety policies (FIRM, H1-2 balance-principled RL) beating per-condition specialists on robustness.

The main switching failure modes are out-of-distribution handoff states, transients at the switch, and threshold tuning (false alarms vs. late triggers).

### Cited Findings
**OP3: composition vs. distillation into one agent (closest to a direct comparison, real hardware, small biped):**
- The get-up and soccer teachers were trained separately, then distilled into one agent through KL regularization toward the applicable teacher. The teacher was chosen by a *hand-coded upright check*: "the distillation step assumes we can manually choose the correct skill (either get-up or soccer) for each state."
- The regularization weight adapts: it is 1 (behavior cloning) while predicted Q is below a threshold and 0 (pure RL) above it.
- Rationale quoted: distillation "stops after the agent's performance surpasses a pre-set threshold, which enabled the final behaviors to be more diverse, fluent, and robust than simply composing the skills". It also "enabled the agent to improve beyond any simple scheduling of the skill policies." Training end-to-end *without* skill teachers gave poor local optima such as rolling on the ground or only standing.
- Hardware, versus a scripted baseline: walking 181% faster, turning 302% faster, get-up 63% faster, kicking 34% faster. Get-up took 0.93 s real vs. 0.73 s sim. Get-up + kick succeeded in 50/50 real set-piece trials. Scoring was 29/50 real vs. 35/50 sim.
- Caveat: the paper does not report a *quantitative* ablation of switching vs. distilled agent on hardware. — [arXiv 2304.13653 (HTML)](https://arxiv.org/html/2304.13653); [abstract](https://arxiv.org/abs/2304.13653)

**Switching failure modes in the literature (mostly quadruped):**
- Abrupt switching between separately trained experts: in a quadruped study, a hierarchical RL baseline with a residual policy "often failed to handle abrupt switching between separately trained expert policies, likely due to distribution mismatches." Its bounding-to-trotting switch on rough terrain caused stumbling and lower survival time. The paper also reports that failures increase on unseen terrains, "especially during transitions between distinct skills" (search snippet). — [arXiv 2607.13579](https://arxiv.org/pdf/2607.13579)
- Learned switching criteria (quadruped): a high-level policy that *blends* gait policies, with switching thresholds optimized in an outer loop, gave smoother transitions. Discrete switching between single skills "failed to transition to galloping in the real world" (search snippet). — [Frontiers / PMC12957656](https://pmc.ncbi.nlm.nih.gov/articles/PMC12957656/); [arXiv 2502.06676](https://arxiv.org/html/2502.06676v1)
- Multi-expert learning (Yang et al., quadruped): a gating network blends experts. Transitional skills emerged, such as "transition from various failure poses to trotting," without hand-engineering (search snippet). — [arXiv 2012.05810](https://arxiv.org/pdf/2012.05810)
- Zero-shot learning of transitions between unseen skill pairs is listed as an open problem (aggregator source, low weight). — [Emergent Mind open problem](https://www.emergentmind.com/open-problems/scalable-zero-shot-skill-transition-learning)

**Supervisor threshold trade-off (false alarms vs. lead time):**
- SafeFall's unmasked predictor gave 1.22 s lead time but a 4.1% false-alarm rate. The masked predictor gave 0.41 s at 0.06%. An earlier trigger buys time but takes control away from a walker that could have recovered. — [arXiv 2511.18509](https://arxiv.org/html/2511.18509)
- Standing Tall notes that hardware false positives and response time could not be measured. Validating a supervisor on hardware is itself hard. — [arXiv 2506.01141](https://arxiv.org/html/2506.01141)

**Unified vs. specialist robustness (humanoid fall-safety, mostly post-fall):**
- FIRM (UMich, ICRA 2026, Unitree G1): one diffusion-distilled policy for mitigation + recovery, compared against HoST, where "a separate policy is trained for each terrain".
  - Sim success rate, HoST vs. FIRM: flat 99.4% vs. 96.3%, uneven 23.2% vs. 93.2%, wave 10.2% vs. 55.9%.
  - Hardware: flat mat G1-default 10/10, HoST 1/10, FIRM 10/10. Slippery G1-default 7/10, HoST 0/10, FIRM 8/10.
  - The paper does *not* discuss handoff failures and has *no* separate-controller-with-switching baseline. Its argument is that falling and rising are "inseparable phases of a single physical process." — [arXiv 2511.07407](https://arxiv.org/html/2511.07407v1)
- Balance-principled unified recovery (Unitree H1-2, 2026): "A single unified policy … spanning the full recovery spectrum." The critic is given capture-point and recoverability information so "the actor [can] decide when to resist a disturbance" vs. step vs. brace.
  - Training: impulses of 50–300 N over 0.1–0.2 s.
  - Sim strategy bands: 50–100 N ankle, 100–200 N hip, 200–300 N stepping, ≥300 N hand bracing.
  - Isaac Lab: 93.4% recovery over 10k trials.
  - Hardware: 10/10 upright recoveries from manual pushes of unreported magnitude, with 3 trials showing transient high-frequency joint oscillations.
  - It has no switching baseline. — [arXiv 2603.08619](https://arxiv.org/html/2603.08619)

### Inferences
- These are the switching failure modes, for the report writer:
  1. **OOD handoff states.** The recovery policy is invoked in states drawn from the walker's failure distribution, which it may not have trained on. The walker is then re-entered from post-recovery states it never saw.
  2. **Transients and discontinuities** in joint targets at the switch (abrupt PD setpoint jumps).
  3. **Chattering** near the threshold, which typically needs hysteresis or a minimum dwell time. This is my inference; no source quantified chattering for bipeds.
  4. **Threshold trade-off.** A late trigger means the recovery policy gets unrecoverable states. An early trigger causes false alarms that interrupt walking.

  The quadruped evidence above directly documents (1) and (2). SafeFall documents (4).
- The evidence that exists (OP3, FIRM, quadruped multi-expert) consistently favors a *single deployed network*, possibly trained with help from separate teachers, over runtime switching.

### Gaps
- I found no paper reporting quantitative hardware fall rate, gait smoothness, or handoff-failure rate for walker + switch vs. unified on a biped.
- I found no source quantifying chattering for learned biped supervisors.

---

## Q3. MoE, distillation (teacher-student, DAgger), multi-critic and multi-skill merges: which have biped or humanoid hardware results?

### Takeaway
"Train specialist teachers, distill into one student, then RL-fine-tune" is the dominant hardware-proven pattern:
- OP3 bipeds: walk/kick + get-up.
- ANYmal parkour: experts + DAgger + RL fine-tune.
- G1 fall-safety: FIRM diffusion distillation.

Multi-critic training (HoST on G1) and MoE/residual experts (MoRE) have humanoid hardware results, but for standing up or gait switching, not push recovery while walking.

### Cited Findings
- **OP3 (biped, hardware):** separate teachers, then a KL-regularized single student with adaptive BC/RL weighting (details in Q2). This is the clearest biped hardware precedent for "separate skills to one policy." — [arXiv 2304.13653](https://arxiv.org/html/2304.13653)
- **FIRM (G1, hardware):** PPO per-demonstration skill priors are distilled (about 4.5M samples) into one diffusion policy outputting 23-dim joint offsets. An adapter retrieves keyframe goals from a codebook every 5 steps. One policy covers mitigation only, recovery only, and the full sequence. — [arXiv 2511.07407](https://arxiv.org/html/2511.07407v1)
- **Parkour in the Wild (ANYmal D, quadruped):** terrain-specific experts are merged into one policy with DAgger, then fine-tuned with RL, with hardware validation (search snippet). — [arXiv 2505.11164](https://arxiv.org/pdf/2505.11164)
- **Light-Loco-Parkour (2026):** experts are distilled into one student *without skill labels*, followed by an RL "transition stage" for switching between behaviors. Platform not confirmed from snippet. — [arXiv 2608.02653](https://arxiv.org/pdf/2608.02653)
- **MoRE (humanoid):** a mixture of latent residual experts plus multiple discriminators, with two-stage training (terrain, then gait-commanded switching) and real-world tests reported (search snippet). — [arXiv 2506.08840](https://arxiv.org/html/2506.08840v1)
- **HoST (RSS 2025, Unitree G1):** stand-up from diverse postures using a **multi-critic** architecture plus a terrain curriculum, smoothness regularization, and a speed limit. It runs directly on G1 in lab and outdoors (search snippet). Its weakness on non-training terrains is shown by FIRM's comparison (Q2). — [arXiv 2502.08378](https://arxiv.org/pdf/2502.08378); [RSS 2025](https://roboticsconference.org/2025/program/papers/64/)
- **HiFAR (Tsinghua + Booster Robotics, IROS 2025):** a multi-stage curriculum for fall recovery (planar 2D falls first, then full 3D), tested on a real humanoid. A get-up, not stay-up, method (search snippet). — [arXiv 2502.20061](https://arxiv.org/html/2502.20061v1)
- **UniReLo (2026):** a unified humanoid policy from fall recovery to locomotion, with real-world recovery-to-locomotion across 5 terrains (gravel, grass, slopes). Only the figure caption was seen. — [arXiv 2606.08922](https://arxiv.org/pdf/2606.08922)
- **VIGOR (2026):** a vision-guided single policy integrating fall mitigation + stand-up, claimed zero-shot to diverse terrains (snippet). — [arXiv 2602.16511](https://arxiv.org/pdf/2602.16511)
- **Berseth et al. ICLR 2018:** progressive RL with distillation merges expert terrain policies for *simulated* bipeds. **Sim only.** — [UBC project page](https://www.cs.ubc.ca/~van/papers/2018-ICLR-distill/index.html)
- **Multi-expert gating (Yang et al.):** MoE gating blends experts, with emergent recovery transitions on a quadruped (snippet). — [arXiv 2012.05810](https://arxiv.org/pdf/2012.05810)
- **Balance-principled critic (H1-2):** this is a "privileged/structured critic" alternative to multi-critic or MoE. Capture-point rewards plus privileged critic inputs plus a curriculum let one actor pick resist/step/brace. "removing the balance-informed structure causes stand-up learning to fail entirely." — [arXiv 2603.08619](https://arxiv.org/html/2603.08619)

### Inferences
- For "stay up while walking," the cheapest hybrid with precedent is the following. Keep the current walker as a teacher for nominal states and train a push-robust policy (or fine-tune the walker under DR + pushes). Then either:
  - fine-tune directly with a KL or BC regularizer to the original walker on nominal states, to preserve gait (OP3-style adaptive regularization); or
  - distill both into one student with DAgger, labeling states by a hand-coded "disturbed vs. nominal" rule (OP3 used an upright check for exactly this), then RL fine-tune.
- MoE gating is mostly demonstrated for terrain or gait selection, not push recovery on bipeds. Its added complexity is not clearly justified for a small biped.

### Gaps
- I found no biped or humanoid hardware paper that uses MoE specifically to combine a walker with a stay-upright expert.
- I did not fetch full papers for MoRE, Light-Loco-Parkour, UniReLo, or HiFAR, so their quantitative results are not captured.
- I did not find Unitree H1 or Berkeley Humanoid papers with explicit separate-vs-unified recovery ablations.

---

## Q4. Practitioner recommendations for small or hobby bipeds

### Takeaway
For small bipeds, the hardware-proven recipe is a single policy trained with targeted DR, an accurate actuator model, push perturbations, an upright-posture reward, and behavior regularization. Separate skills are used only for qualitatively different objectives such as get-up, and are ideally distilled back into one network. No quantitative push-test data exists for hobby bipeds like Open Duck Mini or Microduck.

### Cited Findings
- **OP3 (51 cm, 3.5 kg, hobby-class servos):** zero-shot transfer required "sufficiently high-frequency control, targeted dynamics randomization, and perturbations during simulated training." Without them robots fell "every one or two steps."
  - "basic behavior regularization" (an upright-torso reward, and a torque-peak penalty so robots "rarely break knee gears") was needed because the robots are "inherently fragile."
  - Hardware degraded quickly from loose hip joints and miscalibrated encoders.
  - Battery life was 5–10 minutes. — [arXiv 2304.13653](https://arxiv.org/html/2304.13653)
- **Microduck (Pollen Robotics / Hugging Face; 25 cm, about 800 g):** the RL pipeline is MuJoCo + PPO + DR, exported to ONNX for an ARM chip. Demos include walking and "self-recovery from falls." One secondary blog cites a BAM actuator model plus randomization of voltage, delay, friction, and ±1° backlash (treat cautiously). There are no push-force numbers. — [Menon Lab blog](https://themenonlab.blog/blog/microduck-reinforcement-learning-robotics-getting-started/); [microduck_rl](https://gittrend.io/repo/pollen-robotics/microduck_rl)
- **Open Duck Mini v2** ships MJCF/URDF models for MuJoCo RL training (vendor listing). — [Tindie](https://www.tindie.com/products/wsk/open-duck-mini-v2-raspberry-pi-4b-8gb/)
- **MuJoCo Playground** supports humanoids with zero-shot sim-to-real. It is a framework paper, not a push-recovery study. — [arXiv 2502.08844](https://arxiv.org/html/2502.08844v1)
- **Booster Gym (T1, child-size humanoid)** shows a single end-to-end policy regaining gait within a few steps after a 10 kg ball impact. — [arXiv 2506.15132](https://arxiv.org/pdf/2506.15132)
- **Cassie (Li et al. 2024)** credits I/O history (long + short) and task randomization for disturbance compliance. This suggests adding observation history to a small-biped walker that fails under DR. — [arXiv 2401.16889](https://arxiv.org/abs/2401.16889v1)
- **Reactive stepping push recovery (Atalante exoskeleton)** transferred sim-to-real with only instantaneous proprioceptive observations. It covers standing, not walking. — [arXiv 2203.01148](https://arxiv.org/abs/2203.01148v3)

### Inferences
These are synthesized recommendations for the user, not directly published as such.

1. **First, close the training/test gap.** The 50% fall rate is under conditions (DR + 0.3 m/s pushes) the policy likely did not train on. Fine-tune the existing walker with a push curriculum and DR. Add a KL/BC penalty toward the original policy (or reward terms on gait) on nominal states to avoid degrading the gait (OP3 adaptive regularization precedent). Add proprioceptive history or an estimator/teacher-student latent (Cassie dual-history precedent).
2. **Prefer one deployed network.** The evidence (OP3, FIRM, H1-2 unified, quadruped switching failures) favors one deployed network over a runtime switch for staying upright. On a small biped, the time from push to unrecoverable state is short, so supervisor detection latency plus handoff transients eat most of the recovery window. This is inferred from small-robot dynamics; it is not measured in any source.
3. **If modularity is wanted:**
   - Train a separate "push-recovery expert" under heavy pushes.
   - Distill walker + expert into one student via DAgger, using a hand-coded disturbance label (e.g., IMU angular-rate or tilt threshold) only for *teacher selection during training*, as OP3 did with an upright check.
   - RL fine-tune the student.

   This gets modular training without runtime handoff.
4. **Reserve a runtime IMU supervisor** for switching to a *different objective*: protective fall or get-up once a fall is unavoidable (SafeFall/Standing Tall pattern). Use a conservative, masked-label predictor so it does not steal control from a recoverable walker. Add hysteresis and dwell time.

### Gaps
- No quantitative push-recovery results on hobby bipeds such as Open Duck Mini or Microduck.
- No published practitioner guide comparing switching vs. unified specifically for small bipeds.
- The legged_gym, humanoid-gym, and Isaac Lab READMEs were not fetched in this session. Whether they offer recovery-specific guidance is unverified.
