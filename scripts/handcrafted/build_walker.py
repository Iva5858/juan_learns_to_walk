"""A walking policy for Juan whose every weight is set by hand (no training, no optimizer, no data).

    python scripts/handcrafted/build_walker.py            # -> out/handcrafted/juan_handcrafted.npz

The network is an ordinary robot/policy.py actor (ELU hidden layers, linear output), so the robot runs it
unchanged. It is a compiled hand-written controller:

    target offset from home [rad] = gait pattern(clock, command) + balance reflexes(gravity, gyro)

Units: obs_mean 0 / obs_var 1 (the net sees raw observations), action_scale 1 rad for every joint (an output
of 0.2 = 0.2 rad from home), so every weight below is in radians per observation unit.

Building blocks made of ELU units (E(z) = z for z > 0, exp(z) - 1 otherwise):
  * pass-through   E(x + B) - B = x for x > -B  (B = 12: every input is clipped to +-10)
  * rectifier      E(K x) / K ~ max(x, 0)       (K = 40: off by 1/K = 0.025 at most, smooth at 0)
  * clamp          [E(K(x + a)) - E(K(x - a))] / K - a ~ clip(x, -a, a): with a = stride amplitude from the
                   command, clip(cos, -a, a) is a product-free "amplitude x waveform": 0 when standing,
                   a clipped cosine that grows with the command.

Layers: 123 -> L1 (signals: pass-throughs, rectified clock halves, triangle-wave knots, command level) -> L2
(pass-throughs, the stride clamp, capped lift, optional gate x feedback products) -> 8 outputs (linear sums:
every output is a hand-written formula); an optional L3 rate limiter. Units of blocks whose gains are 0 are
pruned (prune()), so the final net (see P) is 123 -> 19 -> 8 -> 8. Results: out/handcrafted/results.txt.

Gait clock (Controller / env): phase advances 1 per gait period; left foot swings while sin(2 pi phase) > 0,
right foot while sin < 0. Joint conventions: hip roll + = left out / right in, hip pitch + = leg back,
knee + = bend, ankle + = toes down. Foot kept parallel to the torso by ankle = -(hip pitch + knee) offsets.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "out" / "handcrafted"

NAMES = ("left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
HOME = np.array([0.0, -0.4, 0.8, -0.4, 0.0, -0.4, 0.8, -0.4])
LO = np.array([-0.122173, -1.91986, -0.698132, -1.29154, -0.436332, -1.91986, -0.698132, -1.29154])
HI = np.array([0.436332, 1.18682, 1.98968, 1.25664, 0.122173, 1.18682, 1.98968, 1.25664])
OBS_DIM, ACTOR_DIM = 123, 97
# newest history frame starts at 60: gravity 60-62, gyro 63-65, q-home 66-73, 0.1 qd 74-81, prev action 82-89
GX, GY, GZ, WX, WY, WZ = 60, 61, 62, 63, 64, 65
VX, VY, WCMD, SIN, COS, HSIN, HCOS = 90, 91, 92, 93, 94, 95, 96
LHR, LHP, LKN, LAN, RHR, RHP, RKN, RAN = range(8)

B = 12.0   # pass-through offset
K = 40.0   # rectifier / clamp sharpness
KR = 1000.0  # rate limiter sharpness: its limits are a few hundredths of a rad, so the corners must be sharper

# ---------------------------------------------------------------- the hand-chosen numbers
P = dict(
    # gait clock (meta): s per full left+right cycle; the Controller shrinks it linearly to period_fast at
    # fast_speed. Saturated at 0.4 m/s: faster commands get the 0.4 m/s gait (it falls when pushed harder).
    period=0.6, period_fast=0.5, fast_speed=0.4,
    # stride: hip pitch swings +-stride * clip(wave, -amp, amp), amp = min(command / v_full, 1) + feedback
    stride=0.096, v_full=0.4,
    # stride waveform: (1 - tri) cos + tri * triangle wave (the triangle sweeps the stance leg back at a constant
    # rate: the body moves at a steady speed instead of stopping and lurching every step)
    tri=1.0,
    # stride amplitude feedback: amp += stride_fb * (gravity x + stride_fb_tau * gyro y): longer steps while
    # tipping forward, shorter (even backward) while tipping back
    stride_fb=2.0, stride_fb_tau=0.1,
    # swing foot lift: knee bends lift * shape, hip flexes lift_hip of it, the ankle keeps the foot flat.
    # shape = (1 - lift_soft) r(d1) + lift_soft r(d2), r(d) = min(max(x - d, 0), lift_cap (1 - d)) / (1 - d),
    # x = sin (left) or -sin (right) minus the roll term below, d1 = lift_dead, d2 = (1 + d1) / 2: a convex
    # profile (the foot leaves and lands slowly), both feet down around the switch, capped height
    lift=0.3, lift_hip=0.5, lift_dead=0.1, lift_soft=0.6, lift_cap=1.0,
    # lift_roll: the lift threshold moves with the roll: x = sin - lift_roll * (tilt left - roll_tau * roll rate
    # right) for the left leg, mirrored for the right: a leg lifts less (or not at all) while the robot tips
    # toward it, more while it tips away (onto the other foot)
    lift_roll=3.0, roll_tau=0.25,
    # side-to-side sway: both hip rolls + sway * (sin cos(lead) + cos sin(lead)). Both hip rolls + tilt the
    # torso left (with no ankle roll a hip roll on a planted foot turns the torso, it cannot shift it). Off:
    # it only raised landing impacts.
    sway=0.0, sway_lead=0.0,
    # forward lean with the command: both hip pitches -lean * min(command, v_full) (torso tips forward)
    lean=0.2,
    # --- reflexes tried and switched off (0): each made the gait fall more often with the 60-100 ms servo
    # delay. pitch signal = gravity x + pitch_tau * gyro y; roll signal = -gravity y + roll_tau * gyro x.
    pitch_tau=0.1,
    ankle_pitch=0.0,         # ankle strategy, both feet: toes down (+) when tipping forward
    # foot placement, swing leg only (hip pitch -, foot kept flat), from the STANCE leg's lean: alpha = torso
    # tilt + stance hip + stance knee / 2 (the angle of the stance hip-ankle line from vertical, + = body ahead
    # of the foot) and its rate (gyro + joint speeds): roughly a capture point
    place_pitch=0.0,         # rad of swing hip per rad of alpha
    place_rate=0.0,          # rad of swing hip per rad/s of alpha rate
    place_comp=0.0,          # + torso tilt: the swing hip target in the world frame instead of the torso frame
    place_roll=0.0,          # swing leg only: foot out to the side it falls to
    # stance legs (every leg, minus the swing leg via the gate): hip moves that turn the torso back upright.
    roll_stance=0.0,         # hip rolls + roll_stance * roll signal (tilted right -> torso turned back left)
    pitch_stance=0.0,        # hip pitches + pitch_stance * pitch signal (nose down -> legs back = torso up)
    gate_gain=3.0,           # swing gate = clip(gate_gain * sin, 0, 1) (left), clip(-gate_gain * sin, 0, 1) (right)
    # heading (no hip yaw joint): the swing foot lands steer * sin(heading error) further forward (left
    # leg) or back (right leg) when the robot has turned left
    steer=0.0,
    # rate limiter (rad per 20 ms step; 0 = off): each output moves at most this far from the previous action
    # (in the observation). Off: it delays the lift against the stride and the robot falls backward.
    rate_roll=0.0, rate_hip=0.0, rate_knee=0.0, rate_ankle=0.0,
)
PREV = 82   # previous action in the newest history frame
TRI_KNOTS = (0.7, 0.9, 0.98)
TRI_SLOPES = (0.705, 1.095, 1.99, 6.4)   # (2/pi) arcsin through (0.7, 0.494), (0.9, 0.713), (0.98, 0.872), (1, 1)
M = 1.5   # gate product range: gate x signal is exact for |signal| < M (rad)


class Layer:
    """Neurons declared by name; every weight not set explicitly is 0 = no connection."""

    def __init__(self, n_in):
        self.n_in, self.rows, self.bias, self.names = n_in, [], [], {}

    def add(self, name, inputs, bias=0.0):
        """inputs: {index in the previous layer: weight}."""
        w = np.zeros(self.n_in)
        for i, v in inputs.items():
            w[i] += v
        self.names[name] = len(self.rows)
        self.rows.append(w)
        self.bias.append(bias)
        return self.names[name]

    def __getitem__(self, name):
        return self.names[name]

    def arrays(self):
        return np.array(self.rows), np.array(self.bias, dtype=np.float64)


def build(p):
    # ---- L1: signals from the observation
    l1 = Layer(OBS_DIM)
    for name, idx in (("gx", GX), ("gy", GY), ("wx", WX), ("wy", WY), ("sin", SIN), ("cos", COS),
                      ("vx", VX), ("hsin", HSIN)):
        l1.add(name, {idx: 1.0}, B)                         # = x + B
    for j in range(8):
        l1.add(f"prev{j}", {PREV + j: 1.0}, B)              # previous action + B (for the rate limiter)
    Q, QD = 66, 74   # newest joint angles - home, 0.1 x joint speeds
    for side, (h, k) in (("L", (LHP, LKN)), ("R", (RHP, RKN))):
        l1.add(f"alpha{side}", {GX: 1.0, Q + h: 1.0, Q + k: 0.5}, B)            # stance leg lean + B
        l1.add(f"alphadot{side}", {WY: 1.0, QD + h: 10.0, QD + k: 5.0}, B)      # its rate + B
    # triangle wave tri = (2/pi) arcsin(cos), piecewise linear in cos: slope TRI_SLOPES[0] through 0, steeper
    # past each knot (odd-symmetric pairs of rectifiers)
    for i, k in enumerate(TRI_KNOTS):
        l1.add(f"tri_p{i}", {COS: K}, -K * k)              # / K = max(cos - k, 0)
        l1.add(f"tri_m{i}", {COS: -K}, -K * k)             # / K = max(-cos - k, 0)
    lr, rt = p["lift_roll"], p["roll_tau"]
    d1 = p["lift_dead"]
    d2 = (1 + d1) / 2
    for name, sg in (("relu_sin", 1.0), ("relu_nsin", -1.0)):          # / K: left lift, right lift
        wts = {SIN: sg * K, GY: -sg * K * lr, WX: sg * K * lr * rt}
        for tag, d in (("", d1), ("2", d2)):
            l1.add(name + tag, wts, -K * d)
            l1.add(name + tag + "_cap", wts, -K * d - K * p["lift_cap"] * (1 - d))  # cap: subtracted
    # command level a = clip(v / v_full, 0, 1): below v_full, v_hi sits at E(-K..) = -1, so
    # a = (v_lo - v_hi - 1) / K
    l1.add("v_lo", {VX: K / p["v_full"]})
    l1.add("v_hi", {VX: K / p["v_full"]}, -K)
    # swing gates g = clip(gate_gain * sin, 0, 1) = (gate_lo - gate_hi - 1) / K while 0 < g < 1
    g = p["gate_gain"]
    l1.add("gL_lo", {SIN: K * g})
    l1.add("gL_hi", {SIN: K * g}, -K)
    l1.add("gR_lo", {SIN: -K * g})
    l1.add("gR_hi", {SIN: -K * g}, -K)

    # ---- L2: pass-throughs, stride clamp, gated reflexes
    l2 = Layer(len(l1.rows))
    for name in ("gx", "gy", "wx", "wy", "sin", "cos", "vx", "hsin") + tuple(f"prev{j}" for j in range(8)):
        l2.add(name, {l1[name]: 1.0})                       # still x + B
    # = K min(max(x, 0), cap) + 2 (each rectifier output >= -1; their difference >= -1)
    for side, name in (("L", "relu_sin"), ("R", "relu_nsin")):
        for tag in ("", "2"):
            l2.add(f"swing_{side}{tag}", {l1[name + tag]: 1.0, l1[name + tag + "_cap"]: -1.0}, 2.0)
    # stride amplitude amp = a + stride_fb * (forward tilt + stride_fb_tau * pitch rate): longer steps when
    # tipping forward, shorter (even backward) when tipping back. Continuous, so no joint jumps at touchdown.
    kf, tf = p["stride_fb"], p["stride_fb_tau"]
    amp = {l1["v_lo"]: 1 / K, l1["v_hi"]: -1 / K, l1["gx"]: kf, l1["wy"]: kf * tf}
    amp_c = -1 / K - B * kf * (1 + tf)
    l2.add("level", {l1["v_lo"]: 1 / K, l1["v_hi"]: -1 / K}, B - 1 / K)                 # = a + B (command only)
    l2.add("a", amp, amp_c + B)                                                          # = amp + B
    # wave = (1 - tri) cos + tri * triangle, as {L1 index: weight} + constant
    w = p["tri"]
    wave = {l1["cos"]: 1 - w + w * TRI_SLOPES[0]}
    for i in range(len(TRI_KNOTS)):
        d = (TRI_SLOPES[i + 1] - TRI_SLOPES[i]) / K * w
        wave[l1[f"tri_p{i}"]], wave[l1[f"tri_m{i}"]] = d, -d
    wave_c = -B * (1 - w + w * TRI_SLOPES[0])
    for sgn, tag in ((1.0, "p"), (-1.0, "m")):                         # E(K(wave + a)), E(K(wave - a))
        wts = {i: K * v for i, v in wave.items()}
        for i, v in amp.items():
            wts[i] = wts.get(i, 0.0) + sgn * K * v
        l2.add(f"clamp_{tag}", wts, K * wave_c + sgn * K * amp_c)
    # gate x signal = [E(K(s + M(g - 1))) - E(K(-s + M(g - 1)))] / K: = s when g = 1, 0 when g = 0
    # (|s| < M). s is linear in the pass-throughs, g = (lo - hi - 1) / K.
    # place = how far forward to put the swing foot [rad of hip flexion]: forward fall + heading
    kp, kd, kc, st = p["place_pitch"], p["place_rate"], p["place_comp"], p["steer"]
    roll = {l1["gy"]: -1.0, l1["wx"]: p["roll_tau"]}, B * (1 - p["roll_tau"])
    for side in "LR":
        sgn_steer = st if side == "L" else -st
        stance = "R" if side == "L" else "L"
        # (+ pitch_stance x pitch signal: the gated term also takes back the swing leg's stance term)
        ps, pt = p["pitch_stance"], p["pitch_tau"]
        place = ({l1[f"alpha{stance}"]: kp, l1[f"alphadot{stance}"]: kd, l1["gx"]: kc + ps, l1["wy"]: ps * pt,
                  l1["hsin"]: sgn_steer}, -B * (kp + kd + kc + ps + ps * pt + sgn_steer))
        lo, hi = l1[f"g{side}_lo"], l1[f"g{side}_hi"]
        for sname, (sw, sc) in (("place", place), ("roll", roll)):
            for sgn, tag in ((1.0, "p"), (-1.0, "m")):
                w = {i: K * sgn * v for i, v in sw.items()}
                w[lo] = w.get(lo, 0.0) + M          # K M g = M (lo - hi - 1)
                w[hi] = w.get(hi, 0.0) - M
                l2.add(f"{sname}_{side}_{tag}", w, K * sgn * sc - M - K * M)

    # ---- output: linear formulas in L2 terms. Helper: a signal s as {L2 index: weight} + constant.
    n2 = len(l2.rows)
    W, b = np.zeros((8, n2)), np.zeros(8)

    def sig(name):            # pass-through signal x = l2[name] - B
        return {l2[name]: 1.0}, -B

    def gated(name, side):    # swing gate x signal
        return {l2[f"{name}_{side}_p"]: 1 / K, l2[f"{name}_{side}_m"]: -1 / K}, 0.0

    def add(j, terms, gain):
        w, c = terms
        for i, v in w.items():
            W[j, i] += gain * v
        b[j] += gain * c

    # lift shape = (1 - soft) min(max(x - d1, 0), cap (1 - d1)) / (1 - d1) + soft * (the same with d2)
    ws = p["lift_soft"]
    n1, n2_ = (1 - ws) / (1 - d1) / K, ws / (1 - d2) / K
    swingL = ({l2["swing_L"]: n1, l2["swing_L2"]: n2_}, -2 * (n1 + n2_))
    swingR = ({l2["swing_R"]: n1, l2["swing_R2"]: n2_}, -2 * (n1 + n2_))
    clampc = ({l2["clamp_p"]: 1 / K, l2["clamp_m"]: -1 / K, l2["a"]: -1.0}, B)  # clip(wave, -a, a)

    # stride: left hip back at the start of its swing (phase 0, cos 1), forward at the end (cos -1)
    for hip, ank, s in ((LHP, LAN, 1.0), (RHP, RAN, -1.0)):
        add(hip, clampc, s * p["stride"])
        add(ank, clampc, -s * p["stride"])                  # foot flat: ankle = -hip
    # lift
    for (hip, knee, ank), sw in (((LHP, LKN, LAN), swingL), ((RHP, RKN, RAN), swingR)):
        add(knee, sw, p["lift"])
        add(hip, sw, -p["lift"] * p["lift_hip"])
        add(ank, sw, -p["lift"] * (1 - p["lift_hip"]))      # ankle = -(hip + knee)
    # sway (both hip rolls +: feet left of the torso = torso over the right foot during the left swing)
    for roll_j in (LHR, RHR):
        add(roll_j, sig("sin"), p["sway"] * np.cos(p["sway_lead"]))
        add(roll_j, sig("cos"), p["sway"] * np.sin(p["sway_lead"]))
    # lean with the command
    for hip in (LHP, RHP):
        add(hip, sig("level"), -p["lean"] * p["v_full"])   # = -lean * command, up to v_full
    # reflexes: ankle strategy on both feet
    for ank in (LAN, RAN):
        add(ank, sig("gx"), p["ankle_pitch"])
        add(ank, sig("wy"), p["ankle_pitch"] * p["pitch_tau"])
    # foot placement, swing leg only
    for side, (hr, hp, an) in (("L", (LHR, LHP, LAN)), ("R", (RHR, RHP, RAN))):
        if any(p[k] for k in ("place_pitch", "place_rate", "place_comp", "pitch_stance", "steer")):
            add(hp, gated("place", side), -1.0)
            add(an, gated("place", side), 1.0)              # foot flat
        add(hr, gated("roll", side), -p["place_roll"] - p["roll_stance"])  # tilt right -> swing foot to the right
    # stance terms on every leg (the gated terms above remove them from the swing leg)
    for hr, hp in ((LHR, LHP), (RHR, RHP)):
        add(hr, sig("gy"), -p["roll_stance"])               # roll signal = -gy + roll_tau wx
        add(hr, sig("wx"), p["roll_stance"] * p["roll_tau"])
        add(hp, sig("gx"), p["pitch_stance"])
        add(hp, sig("wy"), p["pitch_stance"] * p["pitch_tau"])

    w0, b0 = l1.arrays()
    w1, b1 = l2.arrays()
    rates = np.array([p["rate_roll"], p["rate_hip"], p["rate_knee"], p["rate_ankle"]] * 2)
    if not rates.any():
        return [(w0, b0), (w1, b1), (W, b)], {"l1": l1.names, "l2": l2.names}
    # ---- L3: rate limiter. desired d = W L2 + b (the formulas above), previous action a:
    # out = a + clip(d - a, -r, r) = a + [E(KR(d - a + r)) - E(KR(d - a - r))] / KR - r
    l3 = Layer(n2)
    for j in range(8):
        l3.add(f"prev{j}", {l2[f"prev{j}"]: 1.0})            # a + B
    for j in range(8):
        r = rates[j] if rates[j] > 0 else 5.0               # 5 rad/step = no limit
        for sgn, tag in ((1.0, "p"), (-1.0, "m")):
            w = {i: KR * W[j, i] for i in range(n2) if W[j, i] != 0}
            w[l2[f"prev{j}"]] = w.get(l2[f"prev{j}"], 0.0) - KR
            l3.add(f"rate{tag}{j}", w, KR * b[j] + KR * B + sgn * KR * r)
    W3, b3 = np.zeros((8, len(l3.rows))), np.zeros(8)
    for j in range(8):
        r = rates[j] if rates[j] > 0 else 5.0
        W3[j, l3[f"prev{j}"]] = 1.0
        W3[j, l3[f"ratep{j}"]] = 1 / KR
        W3[j, l3[f"ratem{j}"]] = -1 / KR
        b3[j] = -B - r
    w2, b2 = l3.arrays()
    return [(w0, b0), (w1, b1), (w2, b2), (W3, b3)], {"l1": l1.names, "l2": l2.names, "l3": l3.names}


def prune(layers, names):
    """Drop hidden units nothing reads (every outgoing weight 0: blocks whose gains are 0). The network computes
    exactly the same function; only the unused units go."""
    layers = [(w.copy(), b.copy()) for w, b in layers]
    keys = list(names)
    for i in range(len(layers) - 2, -1, -1):
        used = np.flatnonzero(np.any(layers[i + 1][0] != 0, axis=0))
        w, b = layers[i]
        layers[i] = (w[used], b[used])
        wn, bn = layers[i + 1]
        layers[i + 1] = (wn[:, used], bn)
        if i > 0:  # the inputs of layer i are the units of layer i - 1: unchanged here, pruned next round
            pass
        old = names[keys[i]]
        rank = {int(u): r for r, u in enumerate(used)}
        names[keys[i]] = {n: rank[j] for n, j in old.items() if j in rank}
    return layers, names


def save(p, path):
    layers, names = prune(*build(p))
    data = {"n_layers": len(layers), "actor_dim": ACTOR_DIM,
            "obs_mean": np.zeros(OBS_DIM), "obs_var": np.ones(OBS_DIM), "obs_eps": 1e-8, "clip_obs": 10.0,
            "mirror": False}
    for i, (w, bb) in enumerate(layers):
        data[f"w{i}"], data[f"b{i}"] = w, bb
    meta = {"control_hz": 50.0, "history": 3, "action_scale": np.ones(8), "action_filter": 0.5, "home": HOME,
            "joint_lo": LO, "joint_hi": HI, "gait_period": p["period"], "gait_period_fast": p["period_fast"],
            "fast_speed": p["fast_speed"], "joints": np.array(NAMES), "run": "juan_handcrafted",
            "forward_range": (0.0, 0.6), "lateral_range": (0.0, 0.0), "yaw_rate_range": (0.0, 0.0)}
    data.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    path.with_suffix(".json").write_text(json.dumps({"params": p, "neurons": names}, indent=1))
    print(f"saved {path}: layers {[w.shape for w, _ in layers]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "juan_handcrafted.npz"))
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. stride=0.3 (hand adjustments)")
    a = ap.parse_args()
    p = dict(P)
    for kv in a.set:
        k, v = kv.split("=")
        assert k in p, k
        p[k] = float(v)
    save(p, Path(a.out))


if __name__ == "__main__":
    main()
