# Juan's compute and sensing electronics: Raspberry Pi 4 B, Adafruit BNO055 IMU, Pi Camera Module v2.1

Context: "Juan" is a 43 cm, 1.75–2 kg biped. It has 8 Hiwonder HTD-35H bus servos on a USB-serial debug board, a BNO055 on the Pi's I2C header, a Pi 4 B, a PPO policy at 50 Hz trained in MuJoCo, and a camera planned for later.

These notes deliberately go deeper than the earlier `research_notes/Biped fall recovery best practices/hobby_servo_sim_to_real.md`. That file already covers:
- the BNO055's 100 Hz fusion rate,
- the advice to use IMU mode rather than NDOF,
- heading jumps,
- general Open Duck and ToddlerBot IMU practice,
- a rough Pi inference estimate.

Evidence tags:
- **[primary]**: a datasheet, official documentation, or library source code.
- **[anecdotal]**: a forum post, blog, or GitHub issue.
- **[inference]**: my own arithmetic or reasoning.

The Bosch datasheet used here is **BST-BNO055-DS000-18, Rev 1.8, October 2021**. I downloaded it and extracted the text in this session, which closes the "datasheet could not be parsed" gap from the earlier notes.

---

## 1. BNO055: fusion modes, output rates, policy outputs, noise and bias figures, latency, calibration, known failure modes

### Takeaway
- **Mode:** run the BNO055 in **IMU mode** (OPR_MODE 0x08, accelerometer plus gyro). That mode has no magnetometer, so it has no heading jumps, and it fuses at 100 Hz.
- **What to read:** the gravity vector (0x2E–0x33) or the quaternion (0x20–0x27), plus the gyro (0x14–0x19).
- **Sensor noise is tiny compared with the noise used in typical RL randomization.** In fusion mode the gyro is about 0.1 °/s rms (about 0.002 rad/s) and the accelerometer about 1–2 mg rms.
- **What matters more for sim realism:**
  - accelerometer zero-g offset: ±80 mg typical, ±150 mg max, which is a 4.6° to 8.5° tilt error before calibration;
  - gyro zero-rate offset: ±1 °/s typical, ±3 °/s max;
  - background auto-calibration, which **cannot be disabled**;
  - the fusion trusting the gyro alone while it detects motion;
  - roll and pitch errors under sustained acceleration (anecdotally up to about ±20°).

### Cited Findings

**Modes and rates [primary]** — [Bosch BNO055 datasheet Rev 1.8](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf)
- Fusion modes (Table 3-3/3-5):

  | Mode | OPR_MODE | Sensors used | Orientation |
  |---|---|---|---|
  | IMU | 0x08 | accel + gyro | relative |
  | COMPASS | 0x09 | accel + mag | absolute |
  | M4G | 0x0A | accel + mag (mag used as a gyro substitute) | relative |
  | NDOF_FMC_OFF | 0x0B | all three | absolute |
  | NDOF | 0x0C | all three, with Fast Magnetometer Calibration | absolute |

  The default after power-on is CONFIGMODE, in which all outputs read zero and fusion is halted.
- Fusion output data rates (section 3.6.3):
  - **IMU: 100 Hz** for accel input, gyro input, algorithm calls and fusion output.
  - NDOF and NDOF_FMC_OFF: 100 Hz fusion, with the magnetometer at 20 Hz.
  - M4G: 50 Hz.
  - COMPASS: 20 Hz.
  - The datasheet says IMU mode's "calculation is fast (i.e. high output data rate)."
- The data rate tolerance is ±1% with the internal oscillator.
- Separately, the internal 32 kHz clock "can have clock deviation up to +3%." An external 32 kHz crystal can be selected. On a Pi, the host clock will drift relative to the BNO's 100 Hz.
- In fusion modes Bosch fixes the sensor configuration (Table 3-7), and the user cannot change it:

  | Sensor | Range | Bandwidth | Resolution | Other |
  |---|---|---|---|---|
  | Accelerometer | ±4 g | 62.5 Hz | 14 bit | |
  | Gyroscope | 2000 °/s | 32 Hz | 16 bit | |
  | Magnetometer | | | | forced mode, 20 Hz ODR |

  - **Implication: in fusion mode, gyro saturation is ±2000 °/s and accelerometer saturation is ±4 g.**
  - In non-fusion modes (e.g. AMG or ACCGYRO), the gyro bandwidth can be set to 523/230/116/64/47/32/23/12 Hz and the accelerometer bandwidth to 7.81–1000 Hz.
- Timing:
  - Start-up (off to config mode): 400 ms. Power-on reset to config mode: 650 ms.
  - Mode switch: CONFIGMODE to any mode takes **7 ms**; any mode to CONFIGMODE takes **19 ms** (Table 3-6).
- Supply current in 9DOF at 100 Hz: up to **12.3 mA**. That is negligible for the robot's power budget.

**Outputs and their scaling [primary]** — [Bosch datasheet](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf)
- All fusion modes output a quaternion and Euler angles. They also separate the measured acceleration into **linear acceleration** and **gravity vector** outputs.
- Scaling:

  | Output | Scale | Resolution |
  |---|---|---|
  | Gravity, linear acceleration, raw accel | 1 m/s² = 100 LSB | 0.01 m/s² |
  | Quaternion | 1 = 2^14 LSB | |
  | Euler | 1° = 16 LSB | |
  | Gyro | 1 dps = 16 LSB, or 1 rad/s = 900 LSB | |

- Register map, contiguous block:

  | Data | Registers |
  |---|---|
  | GYR_DATA | 0x14–0x19 |
  | Euler | 0x1A–0x1F |
  | Quaternion (QUA_Data_w_LSB first) | 0x20–0x27 |
  | Linear acceleration | 0x28–0x2D |
  | Gravity (GRV_Data_X_LSB first) | 0x2E–0x33 |
  | Temperature | 0x34 |
  | CALIB_STAT (SYS/GYR/ACC/MAG, 0–3 each) | 0x35 |

- An axis-remap register lets you rotate the sensor frame on-chip to match how it is mounted.

**Datasheet noise and bias numbers (for the IMU model in sim) [primary]** — [Bosch datasheet, Table 0-2](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf)
- **Accelerometer** (specified in accelerometer-only mode, ±2 g, 25 °C):
  - noise density **150 µg/√Hz typical (190 max)**;
  - **zero-g offset ±80 mg typical, ±150 mg max** "over life-time";
  - offset temperature drift ±1 mg/K typical (±3.5 max);
  - sensitivity tolerance ±1% typical (±4% max);
  - cross-axis 1% (2% max);
  - **alignment error relative to the package 0.5° typical, 2° max**;
  - nonlinearity 0.5% FS.
- **Gyroscope** (specified in gyro-only mode):
  - output noise **0.1 °/s rms typical (0.3 max) at 47 Hz bandwidth**, i.e. a noise density of 0.014 °/s/√Hz;
  - **zero-rate offset ±1 °/s typical, ±3 °/s max** (offset cancellation off);
  - offset temperature drift ±0.015 °/s per K (±0.03 max);
  - sensitivity tolerance ±1% (±3% max);
  - cross-axis ±1% (±3% max).
- **Magnetometer:** zero-B offset ±40 µT before calibration and ±2 µT after fusion-mode calibration. Heading accuracy is ±2.5° only "with a fully calibrated sensor and ideal tilt compensation."
- The datasheet gives **no specification for fusion latency or group delay, and no orientation accuracy figure for fused roll and pitch**. I searched the extracted text and none exists.

**How the fusion behaves (from the datasheet's own notes) [primary]** — [Bosch datasheet §3.3.3](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf)
- Note 1: "The sensor fusion algorithm performs automatic background calibration of all sensors. **This feature cannot be disabled.**"
- Note 2: "When the device is in motion, the accelerometer data is temporarily ignored and the fusion relies on the gyroscope for pitch and roll. If the accelerometer data cannot be used for an extended period of time (e.g. due to vibration, or to constant movement), then this may cause the pitch and roll values to drift over time." In IMU mode the heading also drifts, because no magnetometer is used.
- Note 3: "The sensor fusion algorithm was primarily designed to track human motion. If the device is subjected to large accelerations for an extended period of time … the device may incorrectly interpret this large acceleration as the gravity vector."

**Orientation errors under acceleration [anecdotal]**
- A user driving circles reported pitch and roll near 5° at 10 km/h. At 20–30 km/h the values oscillated **up to ±20°**, about 90° out of phase with each other.
- A Bosch moderator attributed this to the BSX fusion, which "does not expect large accelerations due to movement to last that long."
- — [Bosch community: Pitch/Roll under high centripetal acceleration](https://community.bosch-sensortec.com/t5/MEMS-sensors-forum/BNO055-Pitch-Roll-Readings-Under-High-Centripetal-Acceleration/m-p/7580)
- A sibling Bosch fusion part (BHI160B) reportedly needed up to about 10 s for its gravity output to settle. That is an analogy only, not a BNO055 measurement. — [Bosch community BHI160B](https://community.bosch-sensortec.com/t5/MEMS-sensors-forum/BHI160B-Gravity-output-takes-a-long-time-to-settle/m-p/52292)
- The earlier notes already list heading jumps, NDOF overwriting restored offsets, and calibration loss after many orientation changes. Those are not repeated here.

**Calibration profile save and restore [primary]** — [Bosch datasheet §3.6.4, §3.11](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf)
- Procedures:
  - Accelerometer: "6 different stable positions for a period of few seconds," including each axis perpendicular at least once, with slow movement between positions.
  - Gyroscope: "a single stable position for a period of few seconds."
  - Magnetometer: figure-8 movements, faster in NDOF than in FMC_OFF.
- **The offsets can be read only after full calibration, and only after switching to CONFIG_MODE.**
- To restore: select CONFIG_MODE, write the offsets and radius, then switch to the fusion mode.
- "Once the sensor enters the internal calibration routine, the calibration profile is overwritten with the newly obtained sensor offsets."
- Restored offsets take effect only on the write of the last byte (e.g. ACC_OFFSET_Z_MSB).
- Ranges: accelerometer offsets ±500 LSB; magnetometer offsets ±6400 LSB.
- "Incorrect offsets may result in unreliable orientation data even at calibration accuracy level 3."
- Adafruit's CircuitPython library exposes `offsets_accelerometer`, `offsets_gyroscope`, `offsets_magnetometer`, `radius_accelerometer` and `radius_magnetometer`. Each access switches the chip to CONFIG mode and back. [primary] — [adafruit_bno055.py source](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/blob/main/adafruit_bno055.py)
- A calibration example now ships with the library ([issue #66](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/issues/66), resolved by #119). — [bno055_calibrator.py](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/blob/main/examples/bno055_calibrator.py)

**Fusion latency [anecdotal; no primary figure exists]**
- One user reported that the sensor can take up to about 6 ms to produce data at 100 Hz. — [Particle forum: BNO055 inconsistent timing](https://community.particle.io/t/bno055-imu-inconsistent-timing-on-photon/19842)
- No source measured the end-to-end delay from motion to fused output. The datasheet does not give it.

### Inferences
- **IMU noise model for MuJoCo** [inference from the datasheet figures]:
  - **Gyro white noise:** 0.014 °/s/√Hz × √(32 Hz × about 1.57 for a second-order filter's noise bandwidth) ≈ 0.1 °/s ≈ **0.002 rad/s rms**.
  - **Gyro bias:** the fusion auto-calibrates the offset at rest, so the residual is probably well under the ±1 °/s raw spec. Randomize a per-episode bias of **±0.5–1 °/s (about ±0.01–0.02 rad/s)** to be safe.
  - **Gravity noise:** accelerometer noise of 150 µg/√Hz over about 62.5 Hz of bandwidth is about 1.5 mg rms. The fused gravity vector is smoother still, but it is quantized at 0.01 m/s², which is about 0.06° of tilt.
  - **Tilt bias, the term that matters most:** a 2° max package alignment error, plus breakout-to-chassis mounting error, plus up to 80 mg (about 4.6°) of accelerometer offset if uncalibrated. Randomize a **per-episode constant tilt offset of ±2–3°** if the robot is calibrated and offsets are restored, or **±5°** if not.
  - Open Duck's uniform noise (gyro 0.1, gravity 0.1) is about 50× the sensor noise. It is really modeling vibration, impacts and unmodeled dynamics. That is reasonable to keep, but know what it represents.
- **Model the fusion behavior, not just white noise:**
  - (a) During impacts and fast motion the gravity estimate is effectively gyro-integrated (datasheet Note 2). Over a few hundred ms it can drift by the gyro bias × time.
  - (b) Under sustained acceleration, such as a long push or a fall, the estimate is pulled toward the specific-force direction (Note 3).
  - A cheap sim model: compute gravity from a complementary filter with a randomized time constant (e.g. 0.5–2 s) fed by simulated accelerometer and gyro, instead of using the true orientation. This reproduces both effects. It is an inference, not something Bosch documents.
- **Delay:** 100 Hz fusion, so a sample is 0–10 ms old, plus filter lag, plus 1–5 ms of I2C read time (section 2). Randomizing the IMU delay over 0–20 ms with occasional 30–40 ms tails is consistent with this. The earlier notes' 0–40 ms (0–2 policy steps) is a generous upper envelope.
- **Gravity vs quaternion:** both come from the same filter. The gravity vector (6 bytes) is cheaper to read and is directly "projected gravity" in the sensor frame. A quaternion (8 bytes) also gives yaw, which IMU mode lets drift, so it is not useful to the policy anyway.
- **Avoid reading Euler angles.** They have gimbal-lock and wrap-around issues near ±90° and ±180°.
- **The background calibration cannot be disabled, and it overwrites restored offsets.** The gravity estimate can therefore shift slightly during a run as the chip re-estimates offsets. Check `CALIB_STAT` (0x35) before enabling the policy. In IMU mode, SYS calibration status depends only on the accelerometer and gyro. It is also worth logging CALIB_STAT continuously to catch recalibration events.

### Gaps
- No measured BNO055 fusion group delay or step-response latency from any source.
- No Bosch figure for fused roll and pitch accuracy, static or dynamic.
- The ±20° error under sustained acceleration comes from a single anecdote in a car, at about 0.3–0.6 g sustained. No biped-relevant measurement (short impacts of 2–4 g) was found.

---

## 2. Pi I2C clock-stretching incompatibility, workarounds, Python libraries, and achievable read rates

### Takeaway
- **The cause:** the BNO055 clock-stretches, which the datasheet states explicitly. The Pi's hardware I2C mishandles stretching that happens right after an ACK, which corrupts bits. Symptoms are `None` values, temperatures around −92 °C, and garbage fusion data.
- **The three workarounds:**
  - (1) slow the hardware bus (`dtparam=i2c_arm_baudrate=10000` in Adafruit's guide);
  - (2) software I2C via `dtoverlay=i2c-gpio`;
  - (3) UART mode at 115200 baud (PS1 pin high).
- **Option (1) at 10 kHz is too slow for a 50 Hz loop** if you read more than a few bytes. A 32-byte burst takes about 30 ms of wire time.
- **Better options:** i2c-gpio, UART on a proper PL011, or a hardware bus at about 100 kHz with retries, validated by an error counter.
- **The Adafruit library adds overhead:** every property access does an extra I2C read of the mode register. For a 50 Hz loop, do one burst read of the registers you need.

### Cited Findings
- The BNO055's I2C interface "uses clock stretching." It supports standard and fast mode (fSCL max 400 kHz), and its default address is 0x29 (0x28 alternate). — [Bosch datasheet §4.6](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf) [primary]
- UART mode runs at **115200 bps, 8N1**, with reads and writes of up to 128 bytes:
  - read request: `0xAA 0x01 <reg> <len>`;
  - reply: `0xBB <len> <data…>` or `0xEE <status>`;
  - error codes include 0x07 BUS_OVER_RUN_ERROR and 0x0A RECEIVE_CHARACTER_TIMEOUT.
  - — [Bosch datasheet §4.7](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf) [primary]
- A scope analysis of a Pi 3 with a BNO055 [anecdotal but measured] — [softsolder: Pi I2C bus timing vs BNO055 clock stretching](https://softsolder.com/2018/09/04/raspberry-pi-i2c-bus-timing-vs-bno055-clock-stretching/), [softsolder 2018-08-22](https://softsolder.com/2018/08/22/raspberry-pi-3-i2c-vs-bosch-bno055-absolute-orientation-sensor/):
  - The BNO055 holds SCL low after the ACK. The Pi samples SDA before SCL has actually been released, and reads a wrong bit.
  - At a nominal 100 kHz the bus actually ran at **62.5 kHz (a 16 µs low period)**, and about **6% of reads** were corrupted. One example was a temperature 128 °C too high, i.e. bit 7 flipped.
  - At 200 kHz it "seems to work fine" in limited testing. The author flagged this as unverified.
- Adafruit's BNO055 guide says: "you must enable i2c slowdown on the Raspberry Pi device tree overlay." — [Adafruit BNO055 guide, Python & CircuitPython](https://learn.adafruit.com/adafruit-bno055-absolute-orientation-sensor/python-circuitpython) [primary vendor doc]
- Adafruit's clock-stretching fixes guide tests `dtparam=i2c_arm_baudrate=10000` with a BNO055 and reports much better results. — [Adafruit: Raspberry Pi I2C clock stretching fixes, example fixes](https://learn.adafruit.com/raspberry-pi-i2c-clock-stretching-fixes/example-fixes) [primary vendor doc, seen via search snippet; the page body did not render in my fetch]
- Library GitHub issues (Pi 4 users):
  - #67: `None` values and temperatures around −102 °C. — [issue #67](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/issues/67)
  - #61: fusion outputs intermittently `None`; fixed by applying the clock-stretching step. — [issue #61](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/issues/61)
  - #58: random −92/−93 °C. A maintainer said "I have enabled software clock stretching, but it won't work if it's below 10kHz and the bad values still happen at 10kHz." — [issue #58](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/issues/58)
  - [anecdotal]
- The library ships an i2c-gpio example. It uses `dtoverlay=i2c-gpio` in config.txt and reads through `adafruit_extended_bus.ExtendedI2C(<bus number>)`. — [bno055_i2c-gpio_simpletest.py](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/blob/main/examples/bno055_i2c-gpio_simpletest.py) [primary]
- A Bosch community user reported that switching to the kernel's software i2c-gpio driver "correctly implements i2c clock stretching" and that reads every 10 ms worked without problems. Another Pi user reported that **raising** the bus to 400 kHz cured errors. — [Bosch community thread (via search snippet)](https://community.bosch-sensortec.com/t5/MEMS-sensors-forum/BNO055-Temperature-fluctuation/m-p/13220) [anecdotal; the two reports conflict on direction (slower vs faster), which is consistent with softsolder's finding that the failure depends on exact timing]
- Dexter Industries (GoPiGo) found hardware I2C with a BNO055 unusable on the Pi. They implemented clock stretching in software in their `di_i2c.py` driver. — [Dexter forum: The real skinny about the Pi I2C clock stretching bug](https://forum.dexterindustries.com/t/the-real-skinny-about-the-raspberry-pi-i2c-clock-stretching-bug/8080), [Dexter forum: Stretching time](https://forum.dexterindustries.com/t/stretching-time-the-i2c-black-hole/7920) [anecdotal]
- **Read speed reports for the Adafruit library** [anecdotal]:
  - Reading `acceleration` over an i2c-gpio bus took **about 3.9–4.5 ms per property**, with prints inside the timing. The maintainer suggested removing prints and raising the clock. — [issue #71](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/issues/71)
  - UART mode was originally capped at 10 Hz by a 0.1 s sleep. That was fixed to about 100 Hz. — [issue #48](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/issues/48), [PR #50](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/pull/50)
  - An open 2025-era issue reports UART reads of **15–20 ms, with occasional 100–200 ms spikes**, and a broken short-timeout path (a `TypeError`). This was on a CircuitPython board, not a Pi. — [issue #129](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/issues/129)
  - A Pi 4 user saw "UART read error: 7" crashes after 1–2 s in UART mode. The maintainer said to try I2C first. — [issue #96](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/issues/96)
- **Library source** (v5.4.22, April 2026) [primary] — [adafruit_bno055.py](https://github.com/adafruit/Adafruit_CircuitPython_BNO055/blob/main/adafruit_bno055.py):
  - Every public data property (`gyro`, `quaternion`, `gravity`, `euler`, `linear_acceleration`) first evaluates `self.mode`, which is a register read, then reads the data registers.
  - Setting `mode` sleeps 10–20 ms, and the `offsets_*` properties toggle the mode.
  - The UART class forces 115200 baud.
  - `IMUPLUS_MODE = 0x08` is the IMU mode constant.

### Inferences
- **I2C wire-time arithmetic** [inference]:
  - A burst read is about 315 bit-times: start, address and register write (about 27 bits), repeated start and address (about 9 bits), then 32 data bytes × 9 bits, covering 0x14–0x33 (gyro, Euler, quaternion, linear acceleration and gravity).
  - Wire time per bus speed:

    | Bus speed | Wire time for the 32-byte burst |
    |---|---|
    | 10 kHz | **about 32 ms** (longer than the 20 ms control period) |
    | 62.5 kHz (Pi's actual "100 kHz") | about 5 ms |
    | 100 kHz | about 3.2 ms |
    | 400 kHz | about 0.8 ms |

  - Gyro plus gravity alone (0x14–0x19 and 0x2E–0x33) are not contiguous. They need two 6-byte reads, or one 32-byte read.
- **The Adafruit property API doubles the transaction count.** Reading `gravity` and `gyro` separately means four transactions (two mode reads and two data reads). At 10 kHz that is roughly 20–30 ms.
  - Use the library only for setup: set mode, axis remap, and the offset save and restore.
  - Then read the registers directly in one transaction. Options:
    - `smbus2` `i2c_rdwr` with a `write(0x14)` plus `read(32)` message pair;
    - `smbus2.read_i2c_block_data(0x28, 0x14, 32)` (the SMBus block limit is 32 bytes);
    - Blinka's `i2c.writeto_then_readfrom`.
- **Recommended order of attack:**
  1. Try `dtoverlay=i2c-gpio` (software, bit-banged; it handles clock stretching correctly in the kernel driver). Use a 32-byte burst every 10–20 ms, and count failed or implausible reads, e.g. |gravity| outside 9.0–10.6 m/s² or a quaternion norm outside 0.95–1.05.
  2. If that is not clean, use UART mode on a **PL011** UART, not the mini UART. That means `dtoverlay=disable-bt`, or one of the Pi 4's extra UARTs (`dtoverlay=uart3` and similar; see section 3).
     - A 32-byte read over UART is 4 request bytes + 34 reply bytes = 38 bytes × 86.8 µs ≈ **3.3 ms of wire time**.
     - Write your own minimal reader with a timeout and resync on 0xEE errors, rather than relying on the library's sleep-based UART path.
  3. A hardware bus at 100–400 kHz may work for a given board, but softsolder's 6% error rate at 62.5 kHz shows it must be validated, not assumed.
- **Plausibility checks:** whatever the bus, gate each sample with plausibility checks and hold the last good value on failure. Model this in sim as an occasional repeated sample.
- **Thread structure:** read the BNO055 in its own thread or process at 100 Hz, keep only the latest timestamped sample, and have the 50 Hz control loop take it without blocking. Keep only one thread on the I2C bus; a Dexter forum poster linked multi-thread access to lockups.

### Gaps
- I found no published benchmark of i2c-gpio throughput or error rate with a BNO055 on a Pi 4 specifically.
- I did not verify the exact `i2c-gpio` overlay parameters (`i2c_gpio_sda`, `i2c_gpio_scl`, `i2c_gpio_delay_us`) or the resulting bus number. Check the Pi kernel overlay README.
- The 200 kHz and 400 kHz "works" reports are single anecdotes.

---

## 3. Raspberry Pi 4 B hardware: power, undervoltage, thermals, weight, UARTs, USB-serial latency

### Takeaway
- **Power:** the Pi 4 needs a clean **5.1 V, 3 A** supply.
  - It draws about 0.6 A (about 3 W) at idle and about 1.25 A (about 6.4–7.3 W) under full CPU stress.
  - It flags undervoltage below **4.63 V ±5%**.
  - It can supply only **1.2 A total** to USB devices.
- **Thermals:** throttling starts at **80 °C**. A bare board under stress reached throttling in about 1–3 minutes, depending on firmware. A heatsink plus a small fan holds it in the 50s °C.
- **UART:** the GPIO 14/15 UART defaults to the **mini UART**, whose baud rate depends on the core clock. Use `dtoverlay=disable-bt` or one of UART2–5 (PL011) for any sensor UART.
- **USB-serial adapters:** these can add up to 16 ms per transaction, through the FTDI latency timer, unless tuned.

### Cited Findings
- **Power** [primary] — [Raspberry Pi docs: Power supply](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#power-supply) (source: [power-supplies.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/power-supplies.adoc)):
  - "All models require a 5.1V supply." For the Pi 4: recommended PSU 3.0 A, maximum total USB peripheral draw **1.2 A**, typical bare-board active current **600 mA**.
  - Measured Pi 4B current (June 2019 image, with HDMI, keyboard, mouse and Ethernet; amps at 5 V):

    | Condition | Max | Avg |
    |---|---|---|
    | Boot | 0.85 | 0.7 |
    | Idle | | 0.6 |
    | H.264 playback | 0.85 | 0.78 |
    | **Stress** | **1.25** | **1.2** |
    | Halt | 0.023 | |

  - "The Camera Module requires 250mA." GPIO pins combined can safely draw 50 mA, and 16 mA per pin.
  - "Low-voltage detection circuitry that will detect if the supply voltage drops below **4.63 V (±5%)**. This will result in an entry being added to the kernel log." The docs warn that low-quality supplies "can corrupt storage or cause unpredictable behaviour."
- The red PWR LED "is off or flickering when there is undervoltage." — [Raspberry Pi docs: LED warning flash codes](https://www.raspberrypi.com/documentation/computers/configuration.html#led-warning-flash-codes) [primary]
- `vcgencmd get_throttled` bits [primary] — [Raspberry Pi docs: vcgencmd](https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd):

  | Bit | Mask | Meaning |
  |---|---|---|
  | 0 | 0x1 | undervoltage now |
  | 1 | 0x2 | Arm frequency capped |
  | 2 | 0x4 | currently throttled |
  | 3 | 0x8 | soft temperature limit active |
  | 16 | 0x10000 | undervoltage has occurred |
  | 17 | 0x20000 | frequency capping has occurred |
  | 18 | 0x40000 | throttling has occurred |
  | 19 | 0x80000 | soft temperature limit has occurred |

- The official product brief lists [primary] — [Pi 4 product brief RP-008344-DS-2](https://pip-assets.raspberrypi.com/categories/545-raspberry-pi-4-model-b/documents/RP-008344-DS-2-raspberry-pi-4-product-brief.pdf):
  - input power of "5V DC via USB-C connector (minimum 3A)" or via the GPIO header;
  - operating temperature 0–50 °C;
  - BCM2711 quad Cortex-A72 at **1.8 GHz**.
  - Conflict: an April 2024 edition of the brief says 1.5 GHz. Newer board revisions and firmware run at 1.8 GHz and older ones at 1.5 GHz. Check `vcgencmd measure_clock arm`.
- **Thermal** [primary] — [Raspberry Pi docs: Frequency management and thermal control](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#frequency-management-and-thermal-control):
  - The limit is 85 °C. "Between 80°C and 85°C, the Arm cores will be progressively throttled back. If the temperature reaches 85°C, both the Arm cores and the GPU will be throttled."
  - The Pi 4 has no soft limit.
  - DVFS steps are 1500/1000/750/600 MHz, and the default governor is `ondemand`.
  - "Heatsinks are not necessary to prevent overheating damage … However, a heatsink or small fan can reduce thermal throttling."
- **Thermal testing by Raspberry Pi** (bare board, stress-ng plus glxgears) [primary] — [Raspberry Pi blog: Thermal testing Raspberry Pi 4](https://www.raspberrypi.com/news/thermal-testing-raspberry-pi-4/):

  | Firmware | Power, idle to load | First throttle | Notes |
  |---|---|---|---|
  | Launch | 2.89 W → 7.28 W | after 65 s | dropped to 1 GHz, dipping to 750 MHz |
  | Sept 2019 | 2.36 W → 6.67 W | after 155 s | |
  | Beta | 2.1 W → 6.41 W | after 177 s | |

  - Vertical mounting gave about 2 °C less.
  - Reader comments [anecdotal]:
    - heatsink plus a 0.35 W fan: 52 °C after 48 min of stress, no throttling;
    - Pimoroni fan shim: ≤59 °C;
    - closed official case: about 65 °C at idle.
- **Weight:** 46 g per one retailer ([Little Bird Electronics](https://littlebirdelectronics.com.au/products/raspberry-pi-4-model-b?x=2784146)), but 60 g per another ([Digitec](https://www.digitec.ch/en/s1/product/raspberry-pi-4-4gb-model-b-single-board-computer-kits-11267867)), which possibly includes packaging. The official brief gives no weight. [conflicting, retailer data]
- **UARTs** [primary] — [Raspberry Pi docs: Configure UARTs](https://www.raspberrypi.com/documentation/computers/configuration.html#configure-uarts) (source: [interfaces.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/configuration/interfaces.adoc)):
  - The Pi 4 has UART0 (PL011), UART1 (mini UART), and **UART2–UART5 (PL011, disabled by default)**.
  - On the Pi 3 and 4, the **primary UART (GPIO 14 TX pin 8, GPIO 15 RX pin 10, `/dev/serial0`) is UART1, the mini UART**. UART0 (`/dev/ttyAMA0`) goes to Bluetooth.
  - The mini UART has "no break detection, no framing errors detection, no parity bit, and no receive timeout interrupt" and smaller FIFOs.
  - The mini UART's clock "is derived from the VPU core clock … when the core clock frequency changes, the UART baud rate also changes."
  - Fixes:
    - `dtoverlay=disable-bt` makes UART0 (PL011) primary;
    - `dtoverlay=miniuart-bt` moves Bluetooth to the mini UART, but needs `core_freq=250` or `force_turbo=1`.
  - Disable the serial login console (raspi-config, Interface Options, Serial Port) when using the UART for a device.
  - All UARTs are 3.3 V.
- **USB-serial latency** [vendor and kernel docs]:
  - The FTDI default latency timer is **16 ms on Linux and Windows**. Set it with `echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`. It may revert on reconnect, so use a udev rule. — [Granite Devices wiki](https://granitedevices.com/wiki/FTDI_Linux_USB_latency)
  - A 2017 kernel change stopped `ftdi_sio` from defaulting to low-latency (1 ms) mode. Set it explicitly with `setserial /dev/ttyUSB0 low_latency`. — [LKML patch: ftdi_sio fix extreme low-latency setting](https://lkml.iu.edu/hypermail/linux/kernel/1706.0/05857.html)
  - Zaber recommends 2 ms as a practical setting. — [Zaber FTDI latency guide](https://software.zaber.com/motion-library/docs/guides/communication/ftdi_latency)
  - Open Duck's runtime sets `latency_timer` to 1 through udev (cited in the earlier notes).

### Inferences
- **Power budget (5 V rail)** [inference]:
  - Pi under the control loop plus Wi-Fi: about 0.7–1.0 A.
  - Camera: 0.25 A.
  - BNO055: about 0.012 A.
  - USB-serial servo adapter logic: tens of mA, assuming the servos are powered from the 11.1 V pack and not from the Pi's USB.
  - Total is about 1.0–1.4 A typical, with up to about 1.5 A peaks.
  - A 5 V buck converter from the 3S LiPo should be rated at least **5.1–5.2 V and 3 A continuous, with low ripple**, because servo current spikes on the shared battery can sag the input. Feed it through short, thick wires to the USB-C port or to GPIO pins 2/4. The GPIO path bypasses the Pi's polyfuse and input protection.
  - **Log `vcgencmd get_throttled` every few seconds and abort or flag runs with bits 0, 2, 16 or 18 set.** Undervoltage throttling silently changes the loop timing and inference speed.
- **Heatsink or fan:** a sealed robot torso is effectively the "closed official case" case (about 65 °C at idle). A Pi 4 running a control loop, Wi-Fi and H.264 streaming in a closed torso will likely reach 80 °C and throttle. Fit a heatsink plus a small fan or a fan shim, and/or vent the torso. Mass impact is about 5–15 g; that figure is not sourced.
- **Weight:** budget about 46 g for the board alone, plus about 10–20 g for heatsink, fan and standoffs, plus 3 g for the camera and about 3 g for its cable.
- **UART choices for the BNO055 (if not using I2C):**
  - `dtoverlay=uart3` puts a PL011 on GPIO 4/5. On the Pi 4, UART2–5 map to GPIO pairs 0/1, 4/5, 8/9 and 12/13, though I did not verify this mapping in this session.
  - This keeps Bluetooth (needed for a gamepad) on UART0 and avoids the mini UART's clock dependence.
- **Servo bus USB adapter:** identify the chip with `lsusb`. FTDI has the `latency_timer` knob. CH340 and CP210x behave differently and are not covered by these sources. Either way, measure the round-trip time for 8 reads plus 1 write before settling on the sim latency range.

### Gaps
- No official Pi 4 weight.
- No source measured Pi 4 power with Wi-Fi active, camera streaming, and a Python control loop together. The figures above are additive estimates.
- The CH340/CP210x latency behavior on Linux was not researched (it is outside the sources found).

---

## 4. Linux timing for a 50 Hz loop: jitter, PREEMPT_RT, CPU isolation, OS choice, small-MLP inference

### Takeaway
- **Scheduling jitter is small next to the 20 ms period.**
  - Stock Pi 4 kernel: hundreds of µs worst case.
  - PREEMPT_RT: about 90–200 µs.
  - Python garbage collection, blocking I/O, and USB-serial or I2C stalls dominate instead.
- **What to do:**
  - Use 64-bit Raspberry Pi OS Lite on the current stock kernel.
  - Run the control process with `SCHED_FIFO`, pinned to an isolated core, with the `performance` governor.
  - Run I/O in separate threads.
  - Run the MLP in NumPy or ONNX Runtime with one thread.
- PREEMPT_RT is optional. As of 2025 it needed a self-built kernel on Raspberry Pi OS.

### Cited Findings
- **Pi 4 cyclictest** on kernel 4.19:
  - **max 93 µs with PREEMPT_RT vs 301 µs with the standard kernel** (a mostly idle test). — [LeMaRiva: Pi 4B Preempt-RT 4.19 performance test](https://lemariva.com/blog/2019/09/raspberry-pi-4b-preempt-rt-kernel-419y-performance-test) [measured, blog]
  - A forum user's Pi 4 under stress reached a max of about 200 µs. The same test on a Pi 5 spiked to about 800 µs. — [Raspberry Pi forum: PREEMPT_RT performance on Pi 5](https://forums.raspberrypi.com/viewtopic.php?t=382231) [anecdotal]
- On a Pi 5 under stress-ng and iperf3 load (kernel 6.6.21), PREEMPT_RT cut the maximum observed latency by **294×** vs stock. That shows stock-kernel worst cases under heavy I/O can be very large. — [de Wit et al., OSPERT 2024](https://antonio.paolillo.be/publications/workshops/ecrtsOspert2024_dewit_rtlinux_paper.pdf) [primary paper; Pi 5, not Pi 4]
- Older Pi (B/B+, 2015): stock max about 1153–1247 µs, RT about 66–77 µs. — [Emlid RT kernel](https://blog.emlid.com/new-raspbian-with-real-time-kernel-jan-2015) [measured, old hardware]
- PREEMPT_RT was merged into mainline Linux 6.12 but must be enabled at build time. As of January 2025, users building for Raspberry Pi OS had to compile it themselves; no prebuilt RT 6.12 Pi kernel was available. — [LinuxCNC forum: RPi 5 with 6.12 PREEMPT_RT](https://forum.linuxcnc.org/9-installing-linuxcnc/54969-rpi-5-with-6-12-preempt-rt-using-posix-non-realtime?start=10), [LinuxCNC: Raspberry Pi OS PREEMPT RT 6.13 kernel cookbook](https://www.forum.linuxcnc.org/9-installing-linuxcnc/55048-raspberry-pi-os-preempt-rt-6-13-kernel-cookbook) [anecdotal; may be outdated by October 2026]
- A general real-time Linux guide covers `SCHED_FIFO` priority, `isolcpus`/`nohz_full`, locking memory (`mlockall`) and avoiding page faults. — [Shuhao Wu: Real-time programming with Linux, part 2](https://shuhaowu.com/blog/2022/02-linux-rt-appdev-part2.html) [practitioner blog]
- The default CPU governor is `ondemand`, with DVFS steps down to 600 MHz. — [Raspberry Pi docs: Frequency management](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#frequency-management-and-thermal-control) [primary]
- Open Duck Mini runs its ONNX walking policy on a **Pi Zero 2 W** under Raspberry Pi OS Lite 64-bit (cited in the earlier notes). A Pi 4 has more than enough compute.
- No published ONNX Runtime vs NumPy benchmark for a small MLP on a Pi 4 was found. The closest data are CNN-scale (tens of ms) or x86. — [ST wiki ONNX Runtime perf](https://wiki.st.com/stm32mpu/wiki/How_to_measure_the_performance_of_your_models_using_ONNX_Runtime), [OpenNN benchmark](https://www.opennn.net/?p=982)

### Inferences
- **Jitter budget:** even 1 ms of scheduling jitter is 5% of the 20 ms period. The main risks are:
  - (a) a blocking serial read that waits for a USB poll or a timeout;
  - (b) a corrupted I2C transaction that triggers retries;
  - (c) Python GC pauses (`gc.disable()` in the loop plus a manual `gc.collect()` at a safe point, or preallocated NumPy arrays);
  - (d) throttling from heat or undervoltage;
  - (e) Wi-Fi power-save and interrupt load while streaming.
- **Practical recipe:**
  - `isolcpus=3 nohz_full=3 rcu_nocbs=3` on the kernel command line.
  - Run the control process with `taskset -c 3 chrt -f 80`.
  - IMU and servo I/O threads on core 2, camera and streaming on cores 0–1.
  - `cpufreq` governor set to `performance`.
  - `iw dev wlan0 set power_save off`.
  - `mlockall`.
  - Use `time.monotonic_ns()` with an absolute-deadline sleep, `clock_nanosleep(TIMER_ABSTIME)`.
  - Log per-tick timestamps for compute, I/O and total, and report p50/p99/max.
- **Inference cost:** a 3 × 256 MLP is roughly 0.2 MFLOP. A Cortex-A72 at 1.5–1.8 GHz does this in tens of µs of math. Python and NumPy call overhead (about 5–10 µs per op) dominates, so expect **roughly 50–300 µs per inference**.
  - ONNX Runtime with `intra_op_num_threads=1` and `inter_op_num_threads=1` avoids thread wake-up jitter. Multi-threading a tiny model adds latency.
  - Use whichever is faster at p99 on the device. Benchmark both after warm-up.
- **Whether to build PREEMPT_RT:** only if logs show p99 loop lateness above about 1 ms that is attributable to scheduling, not I/O. On a stock kernel with an isolated core and no heavy I/O on that core, sub-millisecond p99 is plausible. The OSPERT 294× result is a reminder that worst-case spikes under heavy network load, which camera streaming is, can be large.

### Gaps
- There is no 2025–2026 Pi 4 cyclictest under combined camera streaming plus Wi-Fi load.
- There is no measured ONNX Runtime vs NumPy latency for a small MLP on a Pi 4. This must be benchmarked.
- I did not confirm whether Raspberry Pi now ships an official PREEMPT_RT kernel package as of October 2026.

---

## 5. Pi Camera Module v2.1 (Sony IMX219): specs, power, weight, cable, software stack, CPU cost of streaming

### Takeaway
- **Specs:** the v2 is a 3 g, 8 MP IMX219 module with a 62.2° × 48.8° FoV, f/2.0, and fixed focus from about 1 m to infinity.
- **Useful sensor modes:** 640×480 at about 206 fps, 1640×1232 at about 41.85 fps, and 3280×2464 at about 21 fps.
- **Power:** 250 mA.
- **Cable:** keep the CSI ribbon at or under about 25–30 cm and strain-relieve it. CSI is a short-reach interface and the ribbon is fragile.
- **Encoding on a Pi 4:** use **hardware** encoders, Picamera2's `H264Encoder` or `MJPEGEncoder` (V4L2), not the software `JpegEncoder`, which defaults to 4 threads and will compete with the control loop.

### Cited Findings
- **Hardware specification, Camera Module 2** [primary] — [Raspberry Pi docs: Camera hardware specification](https://www.raspberrypi.com/documentation/accessories/camera.html#hardware-specification) (source: [hardware_specification.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/accessories/camera/hardware_specification.adoc)):
  - size about 25 × 24 × 9 mm, **weight 3 g**;
  - 8 MP Sony **IMX219**, 3280 × 2464 pixels, 1.12 µm pixels, 1/4" optical format;
  - video modes **1080p47, 1640 × 1232p41, 640 × 480p206**;
  - fixed focus, approx. 1 m to infinity, focal length 3.04 mm, **horizontal FoV 62.2°, vertical 48.8°**, F2.0;
  - max exposure 11.76 s.
- `rpicam-hello --list-cameras` on an IMX219 shows modes 640×480 at 206.65 fps (a 1280×960 crop, binned), 1640×1232 at 41.85 fps (full FoV, binned), and 3280×2464 at 21.19 fps. — [Raspberry Pi docs: camera software](https://www.raspberrypi.com/documentation/computers/camera_software.html) [primary]
- **Power:** "The Camera Module requires 250mA." — [Raspberry Pi docs: Power supply](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#power-supply) [primary]
- **Cable and handling** [primary] — [Raspberry Pi docs: Install a camera](https://www.raspberrypi.com/documentation/accessories/camera.html):
  - The Pi 4 uses the standard 15-pin connector (Standard-Standard cable). The Pi 5 and Zero use a 22-pin mini connector.
  - "Cameras are sensitive to static. Earth yourself prior to handling the PCB."
  - "Take care not to bend the flexible cable at a sharp angle."
  - The supplied cable is 150 mm. — [SparkFun listing](https://www.sparkfun.com/raspberry-pi-camera-module-v2.html)
- **Cable length:**
  - Adafruit (via resellers) says CSI/DSI is meant for "maybe 25cm max." Their 1 m extensions worked in testing but are not guaranteed. — [Adafruit CSI/DSI extender listing (buyzero)](https://buyzero.de/products/adafruit-csi-or-dsi-cable-extender-thingy-for-raspberry-pi)
  - QuuLink estimates about 25–30 cm practical with passive cables, because of D-PHY timing. — [QuuLink on Tindie](https://www.tindie.com/products/fabianiquu/raspberry-pi-camera-cable-extension-20m-quulink)
  - Active extenders (THine, over Ethernet cable) reach up to 20 m. — [MagPi: THine cable extension kit review](https://magazine.raspberrypi.com/articles/review-thine-cable-extension-kit)
  - [vendor and anecdotal; no official maximum is published]
- **Software stack** [primary]:
  - Picamera2 is the libcamera-based Python library. It is supported only on Raspberry Pi OS Bullseye or later. Install it with `sudo apt install -y python3-picamera2` (add `--no-install-recommends` on Lite), not pip. — [Raspberry Pi docs: Picamera2](https://www.raspberrypi.com/documentation/computers/camera_software.html#picamera2), [picamera2 GitHub](https://github.com/raspberrypi/picamera2)
  - The legacy `picamera` and raspivid stack is not part of current OS images.
  - `rpicam-vid` uses hardware H.264 when available. On the Pi 4 the GPU can be overclocked (`gpu_freq=550`) for high frame rates. `libav` uses hardware H.264 on the Pi 4. — [Raspberry Pi docs: rpicam-vid](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-vid)
  - **The Pi 5 has no hardware H.264 encoder.** The Pi 4 does. Its hardware encoder has a bitrate ceiling of about 6–14 Mb/s and a floor of about 240 kb/s. — [Raspberry Pi whitepaper RP-010033: H.264 encoding performance on Pi 5](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010033-WP-1-H.264%20encoding%20performance%20on%20Raspberry%20Pi%205_series%20computers.pdf)
- **Picamera2 encoder implementations** [primary] — [picamera2/encoders source](https://github.com/raspberrypi/picamera2/tree/main/picamera2/encoders):
  - `H264Encoder` and `MJPEGEncoder` are V4L2 hardware encoders. `MJPEGEncoder` checks `_hw_encoder_available`.
  - `JpegEncoder` is **software (simplejpeg) with `num_threads=4` by default**.
  - `LibavH264Encoder` is a multi-threaded software encoder ("use_hw … can be set to True only for VC4 platforms", i.e. the Pi 4 and earlier).
  - The `mjpeg_server.py` example streams MJPEG over HTTP.
- **Streaming latency** [anecdotal]:
  - MJPEG over HTTP at 640×480 gave about 80–100 ms of latency on a Pi. — [Blue Robotics forum](https://discuss.bluerobotics.com/t/suffering-from-latency-and-low-resolution-with-camera/13616)
  - Hardware H.264 at 1080p30 from a Pi 4 to a browser over websockets. — [kroket.io: Pi 4 H.264 to web browser](https://kroket.io/blog/pi4-h264-camera-web.html)
- Pi 4 H.264 playback raised the board's average current from 0.6 A to 0.78 A in Raspberry Pi's table. That is a proxy for the cost of the video blocks, not of encoding. — [Raspberry Pi docs: Power supply](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#power-supply) [primary]

### Inferences
- **Streaming settings for Juan:**
  - 640×480 or 820×616 at 15–30 fps.
  - `MJPEGEncoder` (hardware) for the simplest browser display, or `H264Encoder` (hardware) for lower bandwidth.
  - The ISP, encoder and libcamera mostly run on the GPU, so the CPU cost on the Arm cores should be modest: networking, Python glue and buffer handling. This is unmeasured; measure it with `top` or `pidstat`.
  - Avoid `JpegEncoder` and `LibavH264Encoder` in software mode, and avoid OpenCV per-frame processing on the control core.
  - Run the camera in a separate process pinned away from the control core.
- **The camera competes for the 5 V rail** (+250 mA) **and for heat.** Expect a few °C extra on the SoC.
- **Mechanical:** a walking robot flexes the ribbon at the neck or torso joint, so route it to avoid any moving joint. A failure there kills the camera. Strain-relieve it at both connectors, since connector-flap retention is the usual failure point under vibration. The v2's fixed focus from about 1 m to infinity means close objects, under about 1 m, will be soft; the lens can be refocused with the tool.

### Gaps
- No measured Pi 4 Arm CPU utilization for a Picamera2 hardware H.264 or MJPEG stream at a given resolution was found.
- No official maximum CSI cable length. The 25–30 cm figure comes from vendors.
- No v2.1-specific power-draw measurement beyond Raspberry Pi's general 250 mA figure.

---

## 6. Phone and joystick teleoperation for a Pi robot (brief)

### Takeaway
There are two proven patterns:
- **(a) A Bluetooth gamepad** (8BitDo and similar) paired through `bluetoothctl`, read with `evdev` or pygame. It needs no network, works without Wi-Fi, and keeps the UART0/Bluetooth setup intact, so don't use `disable-bt` if you go this way.
- **(b) A Wi-Fi web app** served from the Pi, for example a small Flask, FastAPI or websockets server with an on-screen joystick, which can also host the MJPEG or H.264 stream.

Either way, map the inputs to the policy's velocity command at 50 Hz and add a dead-man timeout.

### Cited Findings
- Pairing an 8BitDo Lite 2 on a Pi with `bluetoothctl` (power on, agent on, scan, pair, connect, trust) to teleoperate a robot base. — [Cornell IRL: teleoperation](https://irl.tech.cornell.edu/mobile_hoverboard_robot_base/teleoperation) [practitioner doc]
- Reading Bluetooth controllers in Python with `evdev`. Different controller modes produce different event codes, so print the raw events first and stick to one mode. — [Core Electronics: Using USB and Bluetooth controllers with Python](https://core-electronics.com.au/guides/using-usb-and-bluetooth-controllers-with-python/) [tutorial]
- `pygame-controller` was built for Pi robots (used at Pi Wars). Bluepad32 is the option if a microcontroller should receive the gamepad instead. — [pygame-controller on PyPI](https://pypi.org/project/pygame-controller/0.2.0), [Bluepad32](https://github.com/ricardoquesada/bluepad32)
- Picamera2's `mjpeg_server.py` example shows serving the camera over HTTP, which can be combined with web controls. — [picamera2 examples](https://github.com/raspberrypi/picamera2)

### Inferences
- **Gamepad:** use the gamepad for first walking tests. It has lower and more predictable latency than Wi-Fi plus a browser, and it works anywhere. Map one button to a hard "torque off" (dead-man). Have the robot stop and hold or crouch if no event arrives for about 0.5 s.
- **Bluetooth and the UART:** if the BNO055 must use a UART, use `dtoverlay=uart3` or similar so Bluetooth keeps UART0. `disable-bt` would kill the gamepad.
- **Phone web app:** use a websocket that sends {vx, vy, yaw_rate} at 20–50 Hz alongside the MJPEG stream. Run it in a separate process at normal priority from the RT control loop, and pass commands through shared memory or a UDP socket.

### Gaps
- No measured gamepad-to-Pi or browser-to-Pi command latency figures were found.
