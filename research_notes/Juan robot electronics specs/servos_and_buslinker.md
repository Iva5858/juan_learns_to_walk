# Hiwonder HTD-35H serial bus servo and the Hiwonder BusLinker / TTL-USB debug board (for "Juan": 8 leg servos, 3S LiPo, Pi 4 over USB, 50 Hz)

How to read this file:
- This file builds on `../Biped fall recovery best practices/hobby_servo_sim_to_real.md`. It does not repeat the general sim-to-real material, the BAM/ToddlerBot identification recipes, or the first-pass 115200-baud arithmetic from that file. It goes deeper on this specific servo and board.
- Every claim carries an evidence tag:
  - **[OFFICIAL-reseller]**: Hiwonder spec text as reproduced on an authorized reseller page (openelab.io). Hiwonder's own site did not render its spec text to the fetch tool.
  - **[LIB-CODE]**: behaviour encoded in an open-source library that implements the LewanSoul/Hiwonder protocol. It is the library author's reading of the official protocol PDF.
  - **[3rd-party]**: a forum post or third-party listing.
  - **[RECALLED]**: my memory of the official LewanSoul/Hiwonder protocol PDF. I could not re-read it this session because the Purdue-hosted copies are image or CID-font PDFs that would not text-extract. Verify each one.
  - **[ESTIMATE]**: my own arithmetic or inference.
- Source-access problems this session:
  - Robotshop pages and the robotshop-hosted LX-16A and LX-224 manuals returned HTTP 403.
  - Hiwonder.com product pages rendered without specs.
  - The official protocol PDFs (Purdue mirrors) could not be text-extracted: there was no poppler, and the fonts are CID-encoded.
  - Everything marked [RECALLED] should be checked against a locally opened copy of the "LewanSoul/Hiwonder Bus Servo Communication Protocol" PDF.

---

## 1. Official HTD-35H specifications (torque, speed, voltage, current, mass, size, gears, range, resolution, protections)

### Takeaway
Hiwonder officially publishes only a single operating point:
- **35 kg·cm (≈3.43 N·m) stall torque and 0.18 s/60° (≈5.82 rad/s no-load) at 11.1 V.**
- Operating range 9–12.6 V.
- 100 mA no-load and 3 A stall current.
- 64 g, 51.1 × 20.14 × 40 mm, metal gears.
- 0–240° over 0–1000 counts (0.24°/count), with "accuracy" of 0.2° (one listing says 0.3°).

No torque or speed curve versus voltage, no deadband, no backlash and no operating-temperature range is published. Torque is explicitly "a reference value, not a recommended continuous-load rating."

### Cited Findings
**Performance** — [openelab HTD-35H](https://openelab.io/products/htd-35h-bus-servo):
- [OFFICIAL-reseller] Torque is "35kg.cm at 11.1V". No other voltage point is listed.
- [OFFICIAL-reseller] Speed is "0.18 seconds per 60 degrees at 11.1V".
- [OFFICIAL-reseller] Working voltage is "9 to 12.6V".
- [OFFICIAL-reseller] Current is "100mA no-load" and "3A stalled rotor".

**Mechanical** — [openelab HTD-35H](https://openelab.io/products/htd-35h-bus-servo):
- [OFFICIAL-reseller] Dimensions "51.1 x 20.14 x 40mm", weight "64g", cable "200mm".
- [OFFICIAL-reseller] "Metal gears", with a primary plus an assistant (dual) output shaft.
- [OFFICIAL-reseller] Three serial-bus connectors, for daisy-chaining.

**Range, resolution and modes** — [openelab HTD-35H](https://openelab.io/products/htd-35h-bus-servo):
- [OFFICIAL-reseller] Servo mode covers "0 to 240 degrees", and the control value "0 to 1000" maps onto 0–240°.
- [OFFICIAL-reseller] "Servo accuracy 0.2 degrees".
- [OFFICIAL-reseller] There are two modes, servo (position) mode and "deceleration motor mode" (continuous rotation).

**Feedback and protections** — [openelab HTD-35H](https://openelab.io/products/htd-35h-bus-servo):
- [OFFICIAL-reseller] Feedback: "Temperature, voltage and position; angle readback".
- [OFFICIAL-reseller] Protections: "stall and over-temperature protection". The FAQ says these do not replace mechanical stops or system-level emergency controls.
- [OFFICIAL-reseller] "Torque is a reference value, not a recommended continuous-load rating." The page advises a supply "sized for the complete mechanism" with the 3 A stall in mind.

**Conflicts and naming**
- [3rd-party] One listing gives accuracy as 0.3° instead of 0.2°. — [search summary of retailer listings](https://openelab.io/a/s/products/htd-35h-bus-servo), contradicted by [openelab HTD-35H 0.2°](https://openelab.io/products/htd-35h-bus-servo)
- [OFFICIAL-reseller] Don't confuse the **HTS-35H** with the HTD-35H. The HTS-35H is a different 35 kg·cm high-voltage bus servo with a **brass** gear train and a single output shaft. — [openelab HTS-35H](https://openelab.io/a/s/products/hts-35h-bus-servo)
- [3rd-party] The Purdue-hosted "motors_manual.pdf" that search engines index as an HTD-35H manual is actually titled for the HTS-30HS and refers to the HTS-20L. It is a related-family manual, not the HTD-35H's. — [Purdue PDF](https://engineering.purdue.edu/477grp4/Files/refs/motors_manual.pdf)

**Sister models for comparison**
- [3rd-party] LX-16A stall-current numbers conflict between listings ("2.4–3 A" vs "1.1–1.9 A"), and the LX-16A speed is listed as 0.19 s/60° in one place and 0.14 s/60° in another. Hiwonder listings are internally inconsistent, so treat all single-point numbers as nominal. — [Pimoroni LX-16A](https://shop.pimoroni.com/products/metal-gear-serial-servo), [openelab LX-16A](https://openelab.io/products/lx-16a-bus-servo)

### Inferences
**Unit conversions** [ESTIMATE]:
- 35 kg·cm × 9.81/100 = **3.43 N·m** (stall, at 11.1 V).
- 60°/0.18 s = 333°/s = **5.82 rad/s** (no-load, 11.1 V).
- 1 count = 240/1000 = **0.24° = 4.19 mrad**. "0.2° accuracy" is therefore about one count.

**Voltage scaling for a DC-motor model** [ESTIMATE]:
- Stall torque ∝ V and no-load speed ∝ V (roughly, ignoring driver drop).
- At 12.6 V (full 3S): τ_stall ≈ 3.9 N·m and ω0 ≈ 6.6 rad/s.
- At 10.5 V (sagged): τ_stall ≈ 3.2 N·m and ω0 ≈ 5.5 rad/s.
- At 9.0 V (spec floor): τ_stall ≈ 2.8 N·m and ω0 ≈ 4.7 rad/s.
- None of this is published by Hiwonder. Randomize sim τ_max and q̇_max over about ±15–20% around the 11.1 V values to cover the 3S discharge range.

**MuJoCo torque-speed line** [ESTIMATE]:
- τ_max(q̇) = τ_stall·(1 − |q̇|/ω0), clipped at 0, with τ_stall ≈ 3.4 N·m and ω0 ≈ 5.8 rad/s.
- Real hobby servos reach their rated stall torque only briefly. With the 3 A stall rating and thermal protection, the usable **continuous** torque is plausibly 30–50% of stall, i.e. about 1–1.7 N·m.
- Size Juan's knee and hip loads against about 1.5 N·m continuous, not 3.4.

**Electrical figures** [ESTIMATE]:
- From 3 A at 11.1 V, effective winding-plus-driver resistance R ≈ 11.1/3 ≈ 3.7 Ω.
- Torque constant Kt ≈ 3.43 N·m / 3 A ≈ 1.14 N·m/A at the output, including the gearbox.
- Back-EMF Ke ≈ 11.1 V / 5.82 rad/s ≈ 1.9 V·s/rad at the output.
- The mismatch between Kt and Ke implies gearbox inefficiency (roughly 60%) and no-load friction. Use these only as priors for a BAM-style or voltage-based servo model.

**Battery current budget** [ESTIMATE]:
- 8 servos × 3 A stall = **24 A** worst case.
- A realistic peak during a fall-recovery push is perhaps 8–15 A, and standing is about 1–3 A total.

### Gaps
- No torque-versus-voltage or speed-versus-voltage curve is published, and no data point at 12.6 V or 9 V.
- No deadband, no backlash, no gear material breakdown (all-metal vs metal plus copper), no motor type (cored vs coreless), no operating or storage temperature range, and no IP rating were found.
- The position sensor is not stated on the reseller page. The newer BusLinker V3 page describes "magnetic encoder" servos as a different family, which implies the HTD-35H is a potentiometer servo. That is unconfirmed (see section 3).
- Stall-protection behaviour (threshold, how long until it trips, whether it unloads or limits) is not documented on the product page (see section 2).

---

## 2. Communication protocol: frame, baud, half-duplex, commands, response delay, what is and is not exposed

### Takeaway
The HTD-35H speaks the LewanSoul/Hiwonder "Bus Servo Communication Protocol": 115200 8N1, half-duplex single-wire TTL.

Frame: `0x55 0x55 | ID | Len | Cmd | Params… | Checksum`, with checksum = `~(ID+Len+Cmd+ΣParams) & 0xFF`.

Command table:
- 1 = MOVE_TIME_WRITE (pos 0–1000, time 0–30000 ms)
- 2 = MOVE_TIME_READ
- 7 / 8 = MOVE_TIME_WAIT_WRITE / READ
- 11 / 12 = MOVE_START / MOVE_STOP
- 13 / 14 = ID write / read
- 17 / 18 / 19 = ANGLE_OFFSET adjust / write / read (±125 counts = ±30°)
- 20 / 21 = ANGLE_LIMIT write / read
- 22 / 23 = VIN_LIMIT write / read (mV)
- 24 / 25 = TEMP_MAX_LIMIT write / read (50–100 °C, default 85)
- 26 = TEMP_READ
- 27 = VIN_READ
- 28 = POS_READ (signed 16-bit)
- 29 / 30 = OR_MOTOR_MODE write / read (speed −1000…1000)
- 31 / 32 = LOAD_OR_UNLOAD write / read
- 33 / 34 = LED_CTRL write / read
- 35 / 36 = LED_ERROR write / read (bits: stall = 4, over-voltage = 2, over-temp = 1)

**Not exposed:** PID or compliance gains, velocity, current, load or torque feedback, a live fault-status register, sync-read, and (on this family) baud-rate change.

### Cited Findings
**Bus and frame**
- [OFFICIAL-reseller] The servo uses "UART serial command" with "default baud rate 115200". IDs run 0–253 and the default ID is 1. — [openelab HTD-35H](https://openelab.io/products/htd-35h-bus-servo)
- [LIB-CODE] austin-bowen `lewansoul_servo_bus.py` [GitHub source](https://github.com/austin-bowen/lewansoul-servo-bus):
  - The frame is `| Header: byte[2] = 0x55 0x55 | ID: byte | Length: byte | Command: byte | Parameter: byte[0..n] | Checksum: byte |`.
  - Length counts the bytes from the length byte to the checksum inclusive.
  - Checksum is `~(servo_id + length + command + sum(parameters)) & 0xFF`.
  - Constants: `MIN_ID=0`, `MAX_ID=253`, `BROADCAST_ID=254`.
- [LIB-CODE] Command numbers in the same file: `_SERVO_MOVE_TIME_WRITE = 1`, `_MOVE_TIME_READ = 2`, `_MOVE_TIME_WAIT_WRITE = 7`, `_MOVE_TIME_WAIT_READ = 8`, `_MOVE_START = 11`, `_MOVE_STOP = 12`, `_ID_WRITE = 13`, `_ID_READ = 14`, `_ANGLE_OFFSET_ADJUST = 17`, `_ANGLE_OFFSET_WRITE = 18`, `_ANGLE_OFFSET_READ = 19`, `_ANGLE_LIMIT_WRITE = 20`, `_ANGLE_LIMIT_READ = 21`, `_VIN_LIMIT_WRITE = 22`, `_VIN_LIMIT_READ = 23`, `_TEMP_MAX_LIMIT_WRITE = 24`, `_TEMP_MAX_LIMIT_READ = 25`, `_TEMP_READ = 26`, `_VIN_READ = 27`, `_POS_READ = 28`, `_OR_MOTOR_MODE_WRITE = 29`, `_OR_MOTOR_MODE_READ = 30`, `_LOAD_OR_UNLOAD_WRITE = 31`, `_LOAD_OR_UNLOAD_READ = 32`, `_LED_CTRL_WRITE = 33`, `_LED_CTRL_READ = 34`, `_LED_ERROR_WRITE = 35`, `_LED_ERROR_READ = 36`. — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus)
- [3rd-party] Two Purdue-hosted copies of the official protocol confirm that MOVE_TIME_WRITE is command 1 with length 7. Params 1–2 are the angle (low byte then high byte, 0–1000 → 0–240°) and params 3–4 are the time (0–30000 ms). One copy apparently says "0~100", which looks like a typo. — [Purdue protocol copy 1](https://engineering.purdue.edu/477grp4/Team/journal/img%20-%20Juho/week7/servo%20bus%20protocol.pdf), [Purdue protocol copy 2](https://engineering.purdue.edu/477grp4/Files/refs/motors_manual.pdf)

**Parameter ranges and defaults** [LIB-CODE] — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus):
- Angle offset is limited to ±30°, converted as `int(round(offset_degrees * 125 / 30))`, i.e. ±125 counts.
- ANGLE_OFFSET_ADJUST (17) is volatile and ANGLE_OFFSET_WRITE (18) saves it.
- POS_READ is unpacked as a **signed** 16-bit value, documented as "May be negative" (i.e. slightly outside 0–1000).
- VIN_READ returns millivolts.

**Voltage limit** [LIB-CODE] — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus):
- The library clips VIN_LIMIT to **4500–12000 mV** and documents: "If the supply voltage goes outside the specified range, the servos will be powered off. This setting will persist after a servo loses power."
- **This range comes from the 7.4 V LX-16A protocol. For the HTD-35H on a 12.6 V pack, a 12.0 V upper limit would unload the servos on a freshly charged 3S battery.** Read the factory VIN_LIMIT of your servos (cmd 23) before relying on it (see Inferences).

**Temperature limit** [LIB-CODE] — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus):
- TEMP_MAX_LIMIT range 50–100 °C, "Default is 85 C".
- "If the servo temperature exceeds this limit, then the motor will be powered off. This setting persists after a servo loses power."

**Other commands** [LIB-CODE] — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus):
- OR_MOTOR_MODE (29) takes mode 0 (servo) or 1 (motor), plus a speed of −1000…1000. Motor mode is open-loop duty, not closed-loop velocity.
- LOAD_OR_UNLOAD (31) takes 1 for torque on and 0 for torque off (limp).
- LED_ERROR bits: stalled = 4, over_voltage = 2, over_temp = 1.
- The author notes that LED_ERROR_READ "is NOT the error condition that the servo is currently experiencing. There appears to be no way to get that information."

**Velocity is not readable**
- [LIB-CODE] The library's `velocity_read()` is synthesized by reading the position twice with a `time.sleep(period_s)` in between, because there is no velocity register. — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus)

**Library behaviour and caveats**
- [LIB-CODE] Write commands are fire-and-forget, so no reply is read for them. Reads are request-and-reply only, one servo at a time: `_send_and_receive_packet` checks that the reply ID and command match. The default serial timeout is 1.0 s. — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus)
- [3rd-party] A BusLinker-oriented fork tested on the HTD-45H with "Buslinker V2.5" says servo responses "are different than HiWonder's documented protocol." The specific difference is not stated. — [EtcetFelix/HiWonder-BusLinker-servo-bus](https://github.com/EtcetFelix/HiWonder-BusLinker-servo-bus)

**Protocol details recalled from the PDF** [RECALLED], verify these:
- The doc says the servo replies only to read commands. Broadcast ID 254 is accepted for writes, but no reply is given to broadcast.
- MOVE_TIME_WAIT_WRITE (7) preloads a target and MOVE_START (11) executes it. Sending 7 to each servo and then a single broadcast `MOVE_START` to ID 254 makes all 8 servos start on the same byte, which removes the staggered-write skew.
- The doc states the host must release the bus (switch to receive) immediately after sending, because the servo answers right away. No numeric "return delay time" register exists, unlike Dynamixel. — supported by the earlier notes' citation of the [Purdue protocol copy](https://engineering.purdue.edu/477grp4/Files/refs/motors_manual.pdf)

### Inferences
**Not exposed by this protocol family** [ESTIMATE, confirmed by omission from the full command list above]:
- P, I or D gains, compliance margin or slope, and punch.
- Torque or current limit, and present load or current.
- Present velocity.
- Moving or in-position status, and live error or fault status.
- Sync or bulk read and write.
- Baud-rate change. There is no baud command in 1–36, so the HTD-35H is fixed at 115200. The BusLinker V3's 1 Mbps applies to Hiwonder's newer magnetic-encoder servos, not this family.
- Consequences:
  - Sim kp and kd must be fitted, not read.
  - The only torque-related knob is LOAD/UNLOAD.
  - The only speed-shaping knob is the time parameter of MOVE_TIME_WRITE.

**Commanding at 50 Hz** [ESTIMATE]:
- **time = 0** asks the firmware to go as fast as possible. Expect the full step response of the internal P loop: overshoot or ringing on large steps with inertia, and saturation at about 5.8 rad/s.
- **time = 20 ms** (equal to the control period) makes the firmware linearly interpolate setpoints over the period. That acts like a first-order-hold and smooths the 50 Hz staircase.
- Pick one and reproduce it in sim. If you use time = 20 ms, model the servo setpoint as a ramp from the previous target to the new one over 20 ms. This is cheap in MuJoCo by updating ctrl every physics substep.
- This choice is probably more important than the exact kp.

**Safety settings to apply at bring-up** [ESTIMATE]:
- **VIN_LIMIT:** check it first. If the factory maximum reads 12000 mV, raise it if firmware allows (the HTD-35H is rated to 12.6 V). Set the minimum to about 9.0–9.5 V so the servos unload before the LiPo is over-discharged. **Unloading in the middle of a stance means a fall**, so prefer a Pi-side low-voltage warning (VIN_READ) at about 10.5 V, well before the servo's own cut-off.
- **ANGLE_LIMIT** (cmd 20): set it per joint to the mechanical-safe range. This is the firmware-side joint limit.
- **TEMP_MAX_LIMIT:** leave it at 85 °C or lower it to about 70–75 °C. Have the Pi poll TEMP_READ at about 1 Hz (round-robin, one servo per tick) and derate or stop at about 65 °C.

### Gaps
- I could not re-verify from the official PDF:
  - the HTD-35H factory VIN_LIMIT default;
  - whether VIN_LIMIT accepts values above 12000 mV on high-voltage models;
  - the exact broadcast semantics for MOVE_START.
- No official servo response-turnaround time was found. Measure it with a logic analyzer.
- The stall-protection algorithm is not documented anywhere I found: the current or time threshold, whether it unloads fully, and how it recovers (needs a new LOAD command or a power cycle?).
- The LED_ERROR bits show that a stall condition exists, but the trip criteria are unpublished.

---

## 3. BusLinker / TTL-USB debug board: chip, latency, half-duplex split, power pass-through, replacement by Pi UART plus buffer

### Takeaway
"BusLinker" names two different generations:
- **V2.x (the TTL/USB debugging board, 25 × 40 mm).** This is the one paired with LX and HTD servos. A retailer lists V2.4 as **CH340C**-based with a 6–8.4 V input.
- **V3.0 (Type-C, 5–14 V, up to 1 Mbps, 8 baud levels).** It targets Hiwonder's newer magnetic-encoder servos.

No current rating for the screw terminals is published for either generation. The half-duplex split echoes TX onto RX, so the host must discard the echoed bytes. Replacing the board with the Pi's own UART plus a tri-state buffer is feasible and removes USB polling latency, but it does not change the 115200-baud wire time.

### Cited Findings
**V2.x**
- [OFFICIAL-reseller] The TTL/USB board is a "TTL/USB serial bus servo debugging interface", board size "25 × 40mm". It has a USB port, a serial (TXD/RXD) port for an MCU, a servo connector, and "battery positive/negative terminals". Its primary functions are "parameter adjustment and servo-angle reading". No chip, voltage or current rating is stated. — [openelab TTL/USB debug board](https://openelab.io/products/ttl-usb-servo-tester)
- [3rd-party] A Romanian retailer lists "TTL/USB Module, BusLinker V2.4" with a **CH340C** chip and a 6–8.4 V supply input. It was seen via a search snippet only; the page returned 403. — [sigmanortec BusLinker V2.4](https://sigmanortec.ro/en/ttl-usb-module-buslinker-v24-for-serial-servomotors)

**V3.0**
- [OFFICIAL-reseller] Input "5–14V DC" through a 5.5×2.1 mm barrel jack and a terminal block.
- [OFFICIAL-reseller] Communication modes: USB serial, UART and single-wire TTL, selected through an onboard selector.
- [OFFICIAL-reseller] Ports: 5264, PH2.0 and 1.25T servo ports.
- [OFFICIAL-reseller] "Maximum baud rate 1,000,000 bps, with 8 configurable baud-rate levels".
- [OFFICIAL-reseller] It is for "Hiwonder magnetic encoder intelligent serial bus servos".
- [OFFICIAL-reseller] No chip and no current rating are given. It says to "treat the PC data cable as a communication connection rather than the sole power source for a high-load servo system" and to keep servo power separate from logic power.
- Source: [openelab BusLinker V3.0](https://openelab.io/products/buslinker-v3-servo-board)

**Half-duplex echo**
- [3rd-party/LIB-CODE] The recommended full-to-half-duplex circuit for a USB serial adapter "will cause all data sent by the computer to be 'echoed' back". The `ServoBus` class therefore reads and discards `len(packet)` bytes after each send (`discard_echo=True` by default), and calls `reset_input_buffer()` before each send. The README includes a converter schematic and an optional power-isolation schematic. — [EtcetFelix/HiWonder-BusLinker-servo-bus](https://github.com/EtcetFelix/HiWonder-BusLinker-servo-bus), [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus)
- [3rd-party] An open-hardware CH340C half-duplex USB-TTL board exists. Its relationship to the BusLinker is unknown. — [OSHPark CH340C-Half-Duplex](https://oshpark.com/shared_projects/DSqf9Deg)

**Pi-side Python alternatives to the board**
- [3rd-party] Other options are Waveshare's "Bus Servo Adapter (A)" (a generic half-duplex adapter used with Feetech and similar servos) and the PyPI `BusServoControl` library, which uses the same 0x55 0x55 header and inverted-sum checksum. — [Waveshare wiki](https://www.waveshare.com/wiki/Bus_Servo_Adapter_(A)), [PyPI BusServoControl](https://pypi.org/project/BusServoControl/)

### Inferences
**Which board Juan has** [ESTIMATE]:
- If it is the small 25 × 40 mm board with screw terminals, it is the V2.x **CH340-family** board.
- Check with `lsusb`: `1a86:7523` means a CH340.

**USB latency** [ESTIMATE]:
- The CH340 has **no FTDI-style `latency_timer`** to tune.
- On Linux the ch341 driver hands received bytes to user space on the next USB full-speed bulk-IN poll. Typical round trips (write, then reply available to `read()`) are about **1–3 ms** of USB overhead per transaction, on top of wire time.
- 8 sequential request-and-reply reads in Python therefore take roughly **8 × (1.2 ms wire + ~1.5 ms USB + Python overhead) ≈ 20–30 ms**. That is **longer than the 20 ms budget**.
- Measure it. If it is confirmed, reading all 8 positions every tick at 50 Hz through the BusLinker is not achievable.

**"Not the sole power source"** [ESTIMATE]:
- The V2.x "battery terminals" feed straight through to the servo connector's V+ line. There is likely no regulation, but there may be a reverse-polarity diode or a fuse; unknown.
- The **6–8.4 V input rating reported for V2.4** conflicts with running HTD-35H servos at 11.1–12.6 V through it.
- **Do not route the 8-servo 3S power through the BusLinker's small screw terminals or its thin servo lead.** Use the board for signal plus ground only, and feed the servo chain V+ from a separate power distribution (a 14–16 AWG trunk with an inline fuse of about 15–20 A), with a common ground.
- Each servo's 3-pin bus connector and its 200 mm lead are the real current bottleneck. Split the 8 servos into two daisy-chain branches (left leg and right leg) fed separately at the power end.

**Direct Pi UART instead of the board** [ESTIMATE, widely used pattern]:
- Wiring:
  - Use the PL011 UART (disable Bluetooth via `dtoverlay=disable-bt` so `/dev/ttyAMA0` is the PL011, not the mini-UART, whose baud drifts with the core clock).
  - Add a 74HC126 or 74LVC1G125/126 tri-state buffer so TX drives the bus only while a GPIO direction pin is high.
  - Tie RX to the bus through a resistor, and add a 10 kΩ pull-up on the bus.
- Logic levels: the Pi is 3.3 V. The servo bus is commonly treated as 5 V TTL and usually accepts 3.3 V high, but verify. If needed, use a 74LVC buffer powered at 3.3 V on the Pi side, with level shifting on the RX path.
- Direction switching:
  - Option A: Linux RS485 mode (`TIOCSRS485` with RTS as the direction pin) toggles the direction in the driver.
  - Option B: toggle the direction pin in Python after `tcdrain()`. This is jittery (100 µs–1 ms), so a fast servo reply could be clipped.
- Gains:
  - It removes the USB poll latency (about 1–3 ms per transaction).
  - It can drop the echo, if RX is gated.
  - It makes 8 reads plausible in about 10–12 ms of wire time plus turnaround.
- Simpler alternative: an RP2040 or ESP32 co-processor on the bus that polls continuously and streams packed state to the Pi. The earlier notes recommended the same thing. A USB-CDC MCU still incurs USB latency once per tick, but not 8 times.

### Gaps
- The exact USB chip on Juan's board revision is unconfirmed (CH340C per the V2.4 retailer only).
- No published current rating for the BusLinker terminals or traces.
- No official schematic of the BusLinker half-duplex stage was found. Hiwonder's protocol PDF reportedly shows a tri-state-buffer reference circuit [RECALLED], not re-verified.
- No published measurement of CH340 round-trip latency with these servos.

---

## 4. Measured timings and known practical issues (read time for N servos, achievable rate, read failures, angle jumps, overshoot)

### Takeaway
I found **no published measurement** of per-read latency, 8-servo poll time or achievable closed-loop rate for the HTD-35H or any LX-family servo on a BusLinker. The only empirical hints are:
- example code that inserts about 10 ms delays between move and read;
- community reports of read timeouts traced to half-duplex wiring;
- a library author's note that replies through the BusLinker V2.5 differ from the documented protocol.

Plan to measure, and design the loop so it does not need all 8 positions every 20 ms.

### Cited Findings
- [3rd-party] The tve/lx16a-servo ESP32 example uses a 10 ms delay between move and read (carried over from the earlier notes). — [tve/lx16a-servo](https://github.com/tve/lx16a-servo)
- [3rd-party] An Arduino forum user with an LX-16A and the `madhephaestus/lx16a-servo` library reports "I keep getting a read timeout error, and the write function isn't working either". No resolution appeared in the excerpt. That library's scope is listed as LX-16A, LX-224, HTS-35H and LX-15D. — [Arduino forum](https://forum.arduino.cc/t/bus-servo-control-problem/1117048), [lx16a-servo library](https://www.arduino.cc/reference/en/libraries/lx16a-servo)
- [3rd-party] The LX-224 manual states the interface is "a half-duplex UART asynchronous serial interface". This was seen via snippet; the PDF itself returned 403. — [LX-224 manual](https://cdn.robotshop.com/rbm/a5e9ab51-86bc-497b-adac-003de3088fb7/e/e9a77171-ee0b-427e-859e-9b48aa549975/65a47b22_lx-224-serial-bus-servo-user-manual.docx.pdf)
- [3rd-party] Replies through the BusLinker V2.5 differ from the documented protocol, per the HTD-45H tester. — [EtcetFelix repo](https://github.com/EtcetFelix/HiWonder-BusLinker-servo-bus)
- [3rd-party] Talking directly to the servos at 115200 is much faster than going through Hiwonder's Serial Bus Servo Controller (the 40-servo controller board), which runs at 9600 baud. — [EtcetFelix repo](https://github.com/EtcetFelix/HiWonder-BusLinker-servo-bus)

### Inferences
**Wire-time budget at 115200 8N1 (86.8 µs per byte)** [ESTIMATE]:

| Transaction | Bytes | Wire time |
|---|---|---|
| MOVE_TIME_WRITE ×8 (back to back, no reply) | 80 | 6.9 ms |
| MOVE_TIME_WAIT_WRITE ×8 + one broadcast MOVE_START (6 B) | 86 | 7.5 ms |
| POS_READ ×1 (6 B request + 8 B reply) | 14 | 1.22 ms, plus servo turnaround |
| POS_READ ×8 | 112 | 9.7 ms, plus 8 turnarounds, plus 8 USB round trips via BusLinker |
| VIN_READ or TEMP_READ ×1 | 6 + 8 or 7 | ~1.2 ms |

**Proposed 50 Hz schedule for Juan** [ESTIMATE]:
1. At t = 0, send all 8 MOVE_TIME_WRITE frames in one `write()` call (about 7 ms on the wire, but non-blocking for Python).
2. Then read 4 of 8 positions this tick, alternating halves each tick. Each joint is then 20–40 ms stale. Model that in sim.
3. Or read all 8 only if measured round trips are ≤ 1.5 ms each.
4. Every 25th tick, swap one position read for a VIN_READ or TEMP_READ (round-robin) for health monitoring.
5. Run the whole bus in a dedicated thread that always has the latest state ready for the policy.
6. Many RL bipeds on weak buses drop joint-velocity observations and feed the last actions instead. With 0.24° quantization, finite-difference velocity at 50 Hz has a resolution of about 0.21 rad/s per count, so it is noisy. Prefer action history to raw velocity.

**Likely failure patterns to guard against in code** [ESTIMATE]:
- Echo bytes not discarded, which misaligns the frame parser.
- A partial reply (timeout) on a busy or noisy bus.
- Header resync needed after a bad byte. Scan for 0x55 0x55 rather than assuming alignment.
- Checksum failures from motor noise on long daisy chains.
- Recommended handling:
  - Use short serial timeouts (about 5 ms per read, not the library default of 1 s).
  - On failure, reuse the last good value and count the failures. The watchdog trips after N consecutive failures.
- POS_READ can return slightly negative values or values above 1000 near the ends ([LIB-CODE] signed parse). Clip, but log them.

**time = 0 overshoot** [ESTIMATE, no source found]:
- Expect overshoot of a few counts on large time = 0 steps with a leg's inertia, because the internal controller is a stiff P loop on a potentiometer.
- The policy's small per-step deltas (rate-limited, as in Open Duck) keep this negligible.

### Gaps
- No third-party oscilloscope or logic-analyzer measurement of LX or HTD reply turnaround, and no 8-servo poll benchmark, was found.
- No reports specific to the HTD-35H of angle jumps, potentiometer noise or overshoot were found. YouTube and forum measurement content was not reachable within the budget.

---

## 5. Python libraries (and protocol compatibility of HTD-35H with LX-16A, LX-224 and HTD-45H)

### Takeaway
The HTD-35H uses the same LewanSoul/Hiwonder bus protocol as the LX-16A, LX-224, HTS-35H and HTD-45H:
- the same 0x55 0x55 framing;
- the same command numbers;
- 0–1000 over 240°;
- 115200 baud.

Any LX-16A library works, though protection defaults (VIN_LIMIT) may differ. For Juan, **austin-bowen/lewansoul-servo-bus**, or a tiny custom driver modelled on it, is the best base: it handles echo and checksums and has the full command set. pylx16a is friendlier but adds per-call overhead.

### Cited Findings
**Protocol compatibility**
- [OFFICIAL-reseller] The HTD-35H spec (UART, 115200, IDs 0–253, 0–1000 over 0–240°, servo and motor modes, position, voltage and temperature readback) is identical in form to the LX-16A's. — [openelab HTD-35H](https://openelab.io/products/htd-35h-bus-servo), [openelab LX-16A](https://openelab.io/products/lx-16a-bus-servo)
- [3rd-party] The HTD-45H was tested with a LewanSoul-protocol library through the BusLinker V2.5, which empirically confirms protocol compatibility for the HTD family. — [EtcetFelix repo](https://github.com/EtcetFelix/HiWonder-BusLinker-servo-bus)

**Libraries**
- [LIB-CODE] **austin-bowen/lewansoul-servo-bus**:
  - Python 3, pyserial.
  - Complete command set 1–36, `discard_echo` and `verify_checksum` options, a thread lock, and context-manager power handling.
  - Unit conventions:
    - Angles are in degrees.
    - **Temperature defaults to Fahrenheit**. Pass `units='C'`.
  - Write-command semantics:
    - `move_time_write(id, deg, time_s)` truncates its arguments to 0–240° and 0–30 s.
    - `set_powered()` maps to LOAD_OR_UNLOAD.
  - Exiting the context manager powers all servos off by default (`on_exit_power_off=True`). Beware: on a standing biped this drops the robot.
  - Source: [GitHub](https://github.com/austin-bowen/lewansoul-servo-bus)
- [3rd-party] **EtcetFelix/HiWonder-BusLinker-servo-bus**: a fork of the above, adjusted for BusLinker V2.5 reply differences. — [GitHub](https://github.com/EtcetFelix/HiWonder-BusLinker-servo-bus)
- [3rd-party] **pylx16a** (ethanlipson/PyLX-16A on GitHub, plus the PyPI `pylx16a` package):
  - Requires Python ≥ 3.10. Install with `pip install pylx16a`.
  - API: `LX16A.initialize("/dev/ttyUSB0")`, `LX16A(1)`, `set_angle_limits(0, 240)`, `move(...)`. Full docs are in `documentation.md`.
  - Sources: [PyPI pylx16a](https://pypi.org/project/pylx16a/1.0.2/), [GitHub ethanlipson/PyLX-16A](https://github.com/ethanlipson/PyLX-16A)
  - A different PyPI package named `lx16a` also exists. — [PyPI lx16a](https://pypi.org/project/lx16a)
  - [3rd-party] There is also a separate fork at [sethfischer/PyLX-16A](https://github.com/sethfischer/PyLX-16A).
- [3rd-party] **BusServoControl** (PyPI): reads position, temperature and voltage, with the same frame and checksum. — [PyPI](https://pypi.org/project/BusServoControl/)
- [3rd-party] **Arduino/ESP32 libraries**:
  - `madhephaestus/lx16a-servo` covers LX-16A, LX-224, HTS-35H and LX-15D.
  - `tve/lx16a-servo` covers ESP32. These are useful if bus I/O is offloaded to an MCU.
  - Sources: [Arduino lib reference](https://www.arduino.cc/reference/en/libraries/lx16a-servo), [tve/lx16a-servo](https://github.com/tve/lx16a-servo)
- [3rd-party] Hiwonder's own robot kits (TonyPi and similar) ship a Python bus-servo module (often named `BusServoCmd.py` / `hiwonder` SDK) in their downloadable source packages. I could not retrieve it, so I could not confirm its API or timing. — search only, no URL with the file found.

### Inferences
- [ESTIMATE] For the 50 Hz loop, write a ~150-line driver:
  - Pre-build the 8 MOVE_TIME_WRITE frames into one `bytes` object and send it with a single `write()`.
  - Do the reads with a non-blocking parser that resyncs on 0x55 0x55, discards echo by count, and uses per-read timeouts of 3–5 ms.
  - Library abstractions (per-call locks, 1 s timeouts, `reset_input_buffer()` per call) are fine for setup scripts but add jitter in the loop.

### Gaps
- The exact pylx16a source was not fetched (404 on the guessed raw path), so its timeout and echo defaults were not verified.
- The official Hiwonder SDK file was not retrieved.

---

## 6. Known failure modes for walking robots (gear stripping, overheating, low voltage)

### Takeaway
Hiwonder itself warns that 35 kg·cm is a reference, not a continuous rating. The servo protects itself by **unloading** on over-temperature (default 85 °C), on out-of-range input voltage (VIN_LIMIT), and on stall. For a biped every one of these is a "go limp → fall" event. No HTD-35H-specific gear-stripping or field failure reports were found.

### Cited Findings
- [OFFICIAL-reseller] Hiwonder:
  - "Torque is a reference value, not a recommended continuous-load rating."
  - The servo has "stall and over-temperature protection".
  - Protections don't replace mechanical stops or e-stops.
  - Check "allowable side load".
  - Source: [openelab HTD-35H](https://openelab.io/products/htd-35h-bus-servo)
- [LIB-CODE] Exceeding TEMP_MAX_LIMIT (default 85 °C) means "the motor will be powered off". Leaving the VIN_LIMIT window means "the servos will be powered off". Both settings persist across power cycles. — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus)
- [LIB-CODE] There is no readable live fault status, so the host cannot ask why a servo went limp. — [austin-bowen/lewansoul-servo-bus](https://github.com/austin-bowen/lewansoul-servo-bus)
- [primary, from earlier notes] ToddlerBot saw rising motor temperature push the robot out of distribution and cause more falls. — [ToddlerBot arXiv 2502.00893](https://arxiv.org/html/2502.00893v4)

### Inferences
**Thermal** [ESTIMATE]:
- Juan's knees and hips holding a 1.75–2 kg robot in a crouch is the main heating case.
- The detection signal is a servo that goes limp with no error readable over the bus. Detect it on the Pi side:
  - POS_READ diverging from the target while LOAD_OR_UNLOAD_READ (32) reads 0;
  - or TEMP_READ ≥ the limit.
- Mitigations:
  - Add a standing-posture reward in sim for straighter knees.
  - Have the policy observe nothing thermal, but stop experiments at about 65 °C.
  - Rest between runs.

**Low voltage** [ESTIMATE]:
- On a sagging 3S pack, torque and speed fall roughly linearly with voltage (section 1).
- Below VIN_LIMIT the servo unloads. Monitor pack voltage with VIN_READ (any one servo) at about 1 Hz and stop at about 10.5 V under load.

**Over-voltage** [ESTIMATE]:
- A full 3S pack is 12.6 V, which is exactly the HTD-35H's spec ceiling. Regenerative spikes when legs are back-driven during falls can briefly exceed it.
- If the factory VIN_LIMIT maximum is around 12–12.6 V, a fresh pack plus a back-driving transient could trigger an over-voltage unload.
- Consider a bulk capacitor (about 1000 µF, 25 V) on the servo power rail, and check what the LED_ERROR over-voltage bit shows after falls.

**Gears** [ESTIMATE]:
- Metal gears survive overload better than plastic. The usual biped failure is **impact back-driving** during falls, which loads the output gear and spline while the servo holds stiffly.
- Mitigation: on fall detection, send LOAD_OR_UNLOAD = 0 broadcast to ID 254 (one 6-byte frame, under 1 ms) so the legs go limp before impact, or command a compliant crouch.
- The dual output shaft (assistant shaft) should be used with a U-bracket so side loads aren't carried by the output spline alone. Hiwonder warns about side load.

**Stall** [ESTIMATE]:
- Stall protection likely trips after the servo is held near stall for some seconds. The threshold is unpublished.
- A biped stuck in a deep crouch, or pushing against the ground in a failed get-up, can trip it.
- Make get-up motions short, and include a "relax" phase.

### Gaps
- No public HTD-35H teardown, gear-failure or field-reliability reports were found.
- No measured temperature-rise rate under a known holding torque was found.
- The stall-protection trip time and current are unknown. Measure them on the bench: hold the servo against a stop at 11.1 V, log TEMP_READ and the time until it unloads, and watch the LED.
