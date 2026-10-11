# Power system of Juan (43 cm, 1.75–2 kg biped): component specs, loads, run time, risks

Design under review:
- **Servo rail:** 3S 2200 mAh LiPo → XT60 → screw terminal on Hiwonder BusLinker → TTL bus → 8x HTD-35H. One 100 µF capacitor.
- **Compute rail:** 3S 650 mAh LiPo → XT30 → LM2596 buck module → USB-C → Raspberry Pi 4B, plus Pi Camera v2.1 and BNO055.
- **Simulation estimate (servo battery current):** about 2.7–3.7 A mean while walking, with peaks of about 9–10 A.

Research budget note: about 17 tool calls. I could not extract text from the TI LM2596 PDF datasheet in this session; LM2596 figures below come from TI's product page and vendor and hobbyist sources. Where a number is my own arithmetic or engineering judgement, it is placed under **Inferences**, not Cited Findings.

---

## 1. HTD-35H current draw (no-load, typical, stall), 8-servo totals, and inrush

### Takeaway
Hiwonder-channel listings give **100 mA no-load and 3 A stall** per servo at 9–12.6 V. Eight servos therefore draw about 0.8 A at idle (not counting holding load) and up to **24 A in theory if all eight stalled at once**. The simulation's 9–10 A peaks are therefore plausible but not a worst case. I found **no independent measured current data** for the HTD-35H.

### Cited Findings
- **Stall current:** 3 A. **No-load current:** 100 mA. **Working voltage:** 9–12.6 V. **Torque:** 35 kg·cm at 11.1 V. **Speed:** 0.18 s/60° at 11.1 V. — [OpenELAB HTD-35H listing](https://openelab.io/products/htd-35h-bus-servo); [RobotShop HTD-35H listing](https://ca.robotshop.com/products/hiwonder-hiwonder-htd-35h-high-voltage-serial-bus-servo-35kg-torque-with-three-connectors-and-data-feedback)
- OpenELAB says the 35 kg·cm torque figure is a **reference value, not a recommended continuous-load rating**. It also advises planning wiring for the 3 A stalled-rotor current. — [OpenELAB](https://openelab.io/products/htd-35h-bus-servo)
- The servo has **stall and over-temperature protection**, per the listings. — [RobotShop](https://ca.robotshop.com/products/hiwonder-hiwonder-htd-35h-high-voltage-serial-bus-servo-35kg-torque-with-three-connectors-and-data-feedback); [OpenELAB](https://openelab.io/products/htd-35h-bus-servo)
- **Minor conflict:** RobotShop lists 0.3° accuracy, while OpenELAB lists 0.2°. None of the sources state how the stall current was tested or for how long. — [servodatabase/search summary](https://servodatabase.com/servo/hiwonder/htd35h)

### Inferences
- **8-servo current by state:**
  - **Idle, torque off or unloaded:** about 8 × 0.1 = 0.8 A.
  - **Standing:** 0.8 A plus holding current. Holding current is load-dependent and unmeasured. Knees and hips carry the 1.75–2 kg mass, so I estimate roughly 1–2 A total for a crouched stance. This is a guess; measure it.
  - **Walking:** 2.7–3.7 A mean and 9–10 A peaks, per the simulation.
  - **All stalled** (for example, the robot falls and joints jam, or a policy fights a hard limit): up to 24 A until stall protection trips.
  - **Practical design envelope:** 10 A repetitive peaks and 15–25 A brief fault current.
- **The "3 A stall" figure is suspiciously round.** Many 35 kg·cm class 11 V servos are quoted at 2.5–4 A stall. Treat 3 A as nominal and verify it by measuring one servo at stall with a bench supply and current clamp.
- **Inrush at power-up has two parts:**
  - **Capacitor charging:** each servo's onboard capacitance plus the bulk capacitor charges when the battery is connected. This is a short (sub-ms) spike, limited by wire and pack resistance. See Section 6 for the LC overshoot this causes.
  - **Torque-enable jump:** if all servos are commanded at power-up to jump to a pose far from their current position, all eight move at once and draw near-stall current. **Mitigation:** read positions first, then ramp to the stand pose over 1–2 s, or enable servos sequentially.
- No published inrush figure for the HTD-35H was found (see Gaps).

### Gaps
- No independent bench measurements of HTD-35H current (no-load, holding, stall) were found. No official Hiwonder datasheet PDF was retrieved; the figures come from distributor listings.
- No data on the servo's internal capacitance or power-up inrush.
- No data on how long the stall protection takes to trip, or what current it cuts back to.

---

## 2. Is the 3S 2200 mAh pack adequate? C rating, run time, sag, weights, LiPo safety and monitoring

### Takeaway
Yes, with margin, provided it is a reputable **≥25C** pack.
- **Current demand is low:** 10 A peaks are only about 4.5C for a 2200 mAh pack, and even a 24 A all-stall fault is about 11C.
- **Run time:** expect roughly **25–35 min of walking** to a safe cutoff.
- **Main risk:** under-voltage near the servo's 9 V minimum when the pack is low. Set a cutoff of about **3.4–3.5 V/cell under load (10.2–10.5 V)**.
- **Weights:** typical 3S 2200 packs weigh about **165–190 g**, and 3S 650 packs about **59–63 g**.

### Cited Findings
- **3S 2200 mAh 25C pack weights (vendor-listed):**
  - Gens Ace: 164–173 g, with a stated ±20 g tolerance.
  - Ovonic: 168 g.
  - ZOP: 188 g.

  Sources: [ReadyMadeRC Gens Ace EC3](https://www.ReadyMadeRC.com/products/details/gens-ace-2200mah-3s-25c-lipo-battery-ec3); [BuddyRC Gens Ace Deans](https://www.buddyrc.com/products/gens-ace-2200mah-3s-11-1v-25c-lipo-battery-pack-with-deans-plug-1); [rcdrone Ovonic](https://rcdrone.top/products/ovonic-3s-2200mah-battery)
- **Ovonic internal resistance:** Ovonic claims "< 5 mΩ" internal resistance. This is a marketing claim with no test method given, and it is likely per-cell or best-case. — [rcdrone Ovonic](https://rcdrone.top/products/ovonic-3s-2200mah-battery)
- **3S 650 mAh XT30 pack weights:** Tattu 75C 59 g; TheFPV 80C 61 g; Auline 80C 63.2 g. — [GetFPV Tattu](https://www.getfpv.com/batteries/micro-quad-flight-batteries/tattu-650mah-3s1p-75c-11-1v-lipo-battery-pack-with-xt30-plug.html); [Unmanned Tech TheFPV](https://www.unmannedtechshop.co.uk/products/thefpv-650mah-3s-80c-battery-xt30); [Unmanned Tech Auline](https://www.unmannedtechshop.co.uk/products/auline-650mah-3s-11-1v-80c-xt30-lipo-battery)
- **Storage voltage:** about **3.8 V/cell** (some say 3.80–3.85 V; UMD says 3.7–3.8 V). — [Rutgers lithium battery guidance](https://construction.rutgers.edu/rehs/lithium-battery-guidance); [UMD Robotics drone lab](https://facilities.robotics.umd.edu/droneLab/safety/batteries.html); [FMS hobby](https://www.fmshobby.com/blogs/news/lipo-battery-storage-rc-planes)
- **Discharge floor:** never below **3.0 V/cell**, and ideally not below 3.2 V/cell. Hobbyists commonly set the cutoff at 3.4 V/cell, and some use 3.5–3.6 V/cell under high load. These numbers are anecdotal and forum-sourced. — [ARC tutorials safety](https://arc-tutorials.readthedocs.io/en/latest/safety.html); [Phantom Pilots forum](https://phantompilots.com/threads/charging-batteries.64651/post-635747); [RCTalk forum](https://www.rctalk.com/forum/threads/question-regarding-lipo-batteries.143537/page-2)
- **When to retire a pack:** dispose of it if there is any visible deformation (puffing) or if it was discharged below 3.0 V/cell. Store packs in a **LiPo bag or a metal or ceramic container**, away from conductive objects, and do not overfill the container. — [Ramapo College Li-ion policy](https://web.ramapo.edu/ehs/wp-content/uploads/sites/13/2019/09/Lithium-Ion-Battery-Policy-2019.pdf); [Rutgers](https://construction.rutgers.edu/rehs/lithium-battery-guidance)
- **The servo's minimum operating voltage is 9 V.** — [OpenELAB](https://openelab.io/products/htd-35h-bus-servo)

### Inferences
- **C rating:**
  - 10 A / 2.2 Ah = 4.5C.
  - Including a 2x margin, 20 A = 9C.
  - Any genuine 20–25C pack has a rating of 44–55 A, far above need.
  - A higher C rating mostly buys lower internal resistance, which means less sag. It is not about safety at these currents.
  - **Recommendation:** buy a ≥25C pack from a known brand (Gens Ace, Tattu, Turnigy Graphene, CNHL, Ovonic). Do not pay extra for 75–100C packs; they add weight for no benefit here.
- **Run time:** assume about 80% usable capacity (2200 → about 1760 mAh) before reaching about 3.5 V/cell under load.

  | Average current | Run time |
  |---|---|
  | 2.7 A | ≈ 39 min |
  | 3.7 A | ≈ 29 min |
  | 4 A | ≈ 26 min |
  | 1.5 A (standing) | ≈ 70 min |

  Budget **about 25 min of walking per pack** for planning. Cold packs and aging packs shorten this.
- **Voltage sag:**
  - A healthy 3S 2200 25C pack probably has about 15–30 mΩ total internal resistance. This is my estimate from typical hobby-pack values, not a cited measurement.
  - Add about 10–20 mΩ for the XT60, wiring and BusLinker terminal.
  - Total sag: about 0.1–0.2 V at 3.7 A and about 0.3–0.5 V at 10 A.
  - **Full pack:** 12.6 V resting → about 12.1 V at a 10 A peak.
  - **Near-empty pack:** about 3.6 V/cell (10.8 V) resting → about 10.3 V at 10 A, and lower as the pack ages.
  - The margin to the 9 V servo minimum is only about 1–1.3 V near end of discharge. An aged or cold pack, or thin servo leads, can push peak voltage below 9.5 V. That risks servo brownout/reset mid-step, which would make the robot fall.
- **Torque and speed versus voltage:** for a DC-motor servo, stall torque and no-load speed scale roughly linearly with supply voltage. At 10.5 V you have roughly 94% of the 11.1 V rated torque and speed; at 12.3 V roughly 110%. This is a first-order motor-physics inference, not a Hiwonder figure.
  - **Implication for training:** the RL policy will see about ±10% actuator strength variation across a pack. Consider randomizing actuator gain or torque by ±10–15% in simulation.
- **Recommended cutoff:** warn at **3.6 V/cell resting or 3.5 V/cell under load** (≈10.5 V on the bus). Stop and sit down at **3.4 V/cell under load (≈10.2 V)**. This also keeps the servos comfortably above 9 V.
- **Voltage monitoring options, ranked:**
  1. **Software readback of servo input voltage over the bus.** Hiwonder LX/HTD bus servos implement a "read Vin" command in their protocol. This comes from my knowledge of the Hiwonder bus protocol and was not verified against an HTD-35H document in this session. It is free, with no extra hardware: poll one or two servos at about 1 Hz, low-pass filter the reading, and trigger a sit-down behavior below 10.4 V.
  2. **A cheap 1–8S LiPo balance-lead buzzer alarm** (about 8–10 g, set to 3.5 V/cell). This is an independent hardware backstop that also catches a single bad cell.
  3. **A resistor divider into an ADC** (for example, an ADS1115 on the Pi's I2C) on each battery. This is the most flexible option, but the Pi has no native ADC.
- **Charging:** use a balance charger at ≤1C (2.2 A for the servo pack, 0.65 A for the logic pack). Charge on a non-flammable surface or in a LiPo bag, never unattended. Store at 3.8 V/cell. This is standard RC practice and consistent with the cited EHS pages.
- **Fall and fire safety:**
  - Mount the pack inside the torso, in a padded holder, with no sharp screws or edges touching it. Use a strap rather than tape alone.
  - Inspect the pack after any hard fall. Retire it if it is dented, puffed or warm.
  - Keep a LiPo bag or metal box nearby during testing.
  - The XT60 must not be able to short against the chassis. Keep the plug housings intact and insulate solder joints.

### Gaps
- No independent internal-resistance measurements or sag curves were found for specific 3S 2200 packs. The sag numbers above are estimates.
- No Hiwonder documentation retrieved in this session states the HTD-35H's brownout or reset voltage, or confirms that the Vin readback command exists for this model.
- No source quantifies how much HTD-35H torque drops at 10 V versus 11.1 V.

---

## 3. LM2596 module: real current, efficiency, heat, ripple, and fitness for a Pi 4 plus camera; better alternatives; USB-C versus GPIO

### Takeaway
The LM2596 *IC* is rated for 3 A at 150 kHz, but the cheap blue modules are realistically good for about **1.5–2 A continuous without a heatsink**, and clones are common. A Pi 4 with a camera can pull about 1.3–1.5 A typically and more than 2 A in bursts, so the LM2596 is **marginal**: it can work, but it carries a real risk of undervoltage warnings (below 4.63 V) and throttling. **Replace it** with a 5 V ≥5 A regulator, such as the Pololu D36V50F5 or a 5 V 5 A UBEC. Feed the Pi through USB-C, or through GPIO 5 V with care.

### Cited Findings
- **TI LM2596 IC:**
  - "3-A output load current"
  - 4.5–40 V input
  - 150 kHz fixed oscillator (±15%)
  - ±4% output accuracy on the adjustable version
  - Overcurrent protection and thermal shutdown

  TI's page says only "high efficiency" and gives no percentage. — [TI LM2596 product page](https://www.ti.com/product/LM2596)
- **Vendor module ratings conflict:** one lists 2 A rated and 3 A maximum "with adequate heat dissipation"; another LM2596S board lists 1.5 A max continuous. — [ShillehTek LM2596 manual](https://shillehtek.com/blogs/shillehtek-product-manuals/shillehtek-lm2596-dc-dc-adjustable-step-down-power-supply-module-manual); [ProtoSupplies LM2596S](https://protosupplies.com/product/lm2596s-adjustable-dc-dc-step-down-module/)
- A buying guide says the "3 A" ratings on cheap buck modules are peak numbers, and that a "3A" LM2596 or MP1584 board is comfortable at about **1.5–2 A continuous**. — [Microcontrollerslab buck module guide](https://microcontrollerslab.com/?p=44740)
- **Clone modules:** a Hackaday.io log found a fake "LM2596" module switching at **55 kHz instead of 150 kHz**. When powering a Pi, raising the output by about 100 mV reduced the low-voltage warnings. This is a single anecdotal report. — [Hackaday.io log](https://hackaday.io/project/168286/log/173866)
- ShillehTek warns that a **Pi 4 or 5 can pull 1.5–3 A under load**, that thin jumper wires brown out the Pi at boot, and that **powering through GPIO bypasses the USB-C polyfuse and input filter**. — [ShillehTek](https://shillehtek.com/blogs/shillehtek-product-manuals/shillehtek-lm2596-dc-dc-adjustable-step-down-power-supply-module-manual)
- **Raspberry Pi official figures for the Pi 4B:**
  - Recommended PSU: **5 V 3 A (15 W)**; recommended PSU current capacity 3.0 A.
  - Maximum total USB peripheral current: **1.2 A**.
  - Typical bare-board active current: **600 mA**.
  - **The Camera Module requires 250 mA.**
  - Undervoltage detection triggers when the supply drops **below 4.63 V (±5%)**.
  - Back-powering "bypasses the protection circuitry built into the Raspberry Pi."

  Source: [Raspberry Pi hardware documentation – Power supply](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- **Pololu D36V50F5:**
  - 5.5–50 V input, 5 V ±4% output
  - **Typical max continuous current 3.5–8 A** depending on Vin (5.5 A at 36 V)
  - 80–95% efficiency, ~500 kHz switching
  - Soft-start, over-current, short-circuit and over-temperature protection; reverse-voltage protection up to 40 V
  - 2–4 mA quiescent current
  - 25.4 × 25.4 × 9.5 mm; US$39.95; stock status listed as "Rationed"

  Source: [Pololu #4091](https://www.pololu.com/product/4091)

### Inferences
- **Load estimate for the 5 V rail:**
  - Pi 4 bare board: about 0.6 A typical, up to about 1.3 A under full CPU load (see Section 4 measurements).
  - Camera: +0.25 A.
  - BNO055: about 12 mA, negligible.
  - Total: about **0.9–1.6 A sustained**, with transient spikes (boot, CPU frequency changes, USB enumeration) toward 2 A or more.

  A genuine LM2596 module with a small heatsink can supply this. A clone, or an un-heatsinked module in a closed torso, may sag or overheat.
- **Heat:** at 12 V in and 5 V/1.5 A out with roughly 80–85% efficiency (typical for a 150 kHz non-synchronous buck at this ratio; not cited), the module dissipates about 1.3–1.9 W inside the robot. A synchronous 500 kHz regulator such as the D36V50F5 at about 90% dissipates about 0.8 W.
- **Voltage margin:**
  - The Pi flags undervoltage below about 4.63 V.
  - The LM2596 adjustable version has ±4% accuracy, and the trimpot drifts with vibration.
  - A typical USB-C cable drops about 0.1–0.2 V at 2 A.
  - If you keep the LM2596, set its output to **5.15–5.25 V measured at the Pi under load**, secure the trimpot with a dab of glue or nail polish, and use a short, thick cable.
  - Monitor with `vcgencmd get_throttled`; bit 0 means under-voltage now, and bit 16 means it has occurred.
- **Ripple:** cheap LM2596 modules commonly have tens of mV of ripple. Ripple matters less to the Pi than average-voltage dips do. Not cited; flagged as general experience.
- **Recommendation:** replace the LM2596 with the **Pololu D36V50F5** (or the D24V50F5, which accepts up to 38 V). Alternatives are a Matek 5 V/5 A BEC or a Hobbywing/Castle 5 V ≥5 A UBEC. The D36V50F5's 3.5–8 A rating gives >2x headroom. Soft-start and protection features are bonuses.
- **USB-C versus GPIO 5 V:**
  - **USB-C** keeps the Pi's polyfuse and input protection. Use a short USB-C pigtail soldered to the regulator, not a long cable.
  - **GPIO 5 V pins** (pins 2 and 4) avoid connector and cable resistance and are common in robots. However, they bypass protection, so a regulator fault or overvoltage goes straight to the SoC.
  - **Recommendation:** use USB-C with a short, heavy (≤20 cm) cable when using an off-the-shelf regulator. If you use GPIO, add a 5.6–6 V TVS/Zener clamp and an inline fuse (about 3 A).

### Gaps
- I could not extract the efficiency curve or thermal-resistance data from the TI LM2596 datasheet PDF; the efficiency figures above are approximate.
- No measured ripple data was found for typical LM2596 modules.
- I did not retrieve Matek or UBEC product pages, so I have no cited specs for them.

---

## 4. 3S 650 mAh run time for a Pi 4 at typical 3–6 W

### Takeaway
At a 3–7 W load (Pi 4, camera, IMU), a 3S 650 mAh pack (about 7.2 Wh nominal) gives roughly **40–90 min**. A realistic robot workload (camera streaming plus policy inference, about 5 W) gives roughly **55–60 min**. That outlasts the servo pack, so the logic battery is not the bottleneck. Input current is under 1 A, about 1.1C, which is trivial for a 75C pack.

### Cited Findings
- **Measured Pi 4 power:**
  - 575 mA (2.85 W) idle — [RasPi.TV](https://raspi.tv/2019/how-much-power-does-the-pi4b-use-power-measurements)
  - 540 mA (2.7 W) idle and **1280 mA (6.4 W) under 400% CPU stress**, with nothing on USB — [PiDramble](https://pidramble.com/node/27)
  - About 2.4 W idle and about 5.8 W under stress — [8086.support](https://8086.support/content/23/121/en/sitemap/A/en.html)
- **Camera module current:** 250 mA (≈1.25 W at 5 V). — [Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- **Tattu 3S 650 mAh pack:** 75C rating, about 59 g. — [GetFPV](https://www.getfpv.com/batteries/micro-quad-flight-batteries/tattu-650mah-3s1p-75c-11-1v-lipo-battery-pack-with-xt30-plug.html)

### Inferences
- **Energy budget:** 0.65 Ah × 11.1 V = 7.2 Wh. Usable (about 80%) is 5.8 Wh. After regulator efficiency (85% for the LM2596, about 90% for the Pololu), about 4.9–5.2 Wh reaches the Pi.

  | 5 V load | LM2596 (85%) | D36V50F5 (90%) |
  |---|---|---|
  | 3 W | ≈ 98 min | ≈ 104 min |
  | 5 W | ≈ 59 min | ≈ 62 min |
  | 7 W | ≈ 42 min | ≈ 45 min |

- **Battery current:** at 7 W, about 0.75 A from the pack.
- **Deep-discharge risk:** the regulator keeps drawing current as the pack drops, so an unattended Pi will run this pack well below 3.0 V/cell. **Add a balance-lead alarm, or a shutdown script that reads the pack voltage through an ADC, on this pack too.** Never leave the Pi running from the LiPo unattended.

### Gaps
- No measurements were found of a Pi 4 running camera capture plus neural-network inference at the same time. The 5 W "typical robot" figure is an estimate.

---

## 5. Single battery versus two batteries; grounding

### Takeaway
**Keep the two-battery design.** It isolates the Pi from the 9–10 A (up to about 24 A) servo spikes that would otherwise sag a shared 11 V rail. A good 5 V regulator would ride through such sags, but two packs remove the risk entirely, at a cost of about 60 g. However, the **grounds must be common**: TTL serial is referenced to ground. Do this with a **single deliberate ground connection**, and make sure heavy servo return current cannot flow through USB or signal wires.

### Cited Findings
- **BusLinker V3.0:**
  - Accepts 5–14 V on a DC jack (5.5 × 2.1 mm) or a screw terminal.
  - Servo power does not come from USB; the USB-C port is for communication.
  - No maximum current is specified.

  Source: [OpenELAB BusLinker V3.0](https://openelab.io/a/s/products/buslinker-v3-servo-board)
- **Different BusLinker version:** the older V2.4 board is listed at 6–8.4 V input. Do not confuse the two versions; a V2.4 board on 3S (12.6 V) would be over-voltage. — [OpenELAB BusLinker V3.0 listing / search summary](https://openelab.io/a/s/products/buslinker-v3-servo-board)
- **Back-powering:** back-powering the Pi through a hub "bypasses the protection circuitry". — [Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)

### Inferences
- **Single-battery option:**
  - **Pros:** one pack to charge and monitor; about 60 g lighter; one switch; frees space.
  - **Cons:** servo peaks cause about 0.3–0.5 V sag (Section 2), which the buck regulator absorbs easily since it only needs Vin > about 6–7 V. The bigger risks are:
    - stall or fall events (24 A, where a fault could collapse the rail);
    - LC spikes at plug-in;
    - servo regenerative back-EMF reaching the 5 V regulator input;
    - a dead or browned-out Pi while the servos keep holding torque.
  - A single 3S 2200 pack feeding a 5 V 5 A regulator is used in many hobby robots and generally works. If you consolidate, the minimum is a **≥470 µF input capacitor on the regulator plus a Schottky diode or low-Vf ideal diode** in its feed.
- **Two batteries (current design):**
  - **Pros:** complete isolation of logic from servo transients; the Pi can stay up and log data when servos fault; the servo pack can be swapped without rebooting.
  - **Cons:** about 60 g extra (650 pack); two packs to charge and monitor; the two packs deplete at different rates.
- **Grounding:**
  - The BusLinker's USB connection to the Pi already ties servo-battery negative to Pi ground through the USB cable's GND wire. That is a valid common ground, but it means that if any other ground path exists (chassis, a second wire), servo return currents may split through the USB cable.
  - **Recommendation:** keep the two packs' negatives tied at **one point**, ideally the BusLinker or servo power distribution ground. Make sure **no servo current returns through the Pi**: do not power servos from the Pi, and do not run a second ground wire between the boards besides the USB/UART.
  - If you replace the BusLinker's USB link with direct Pi UART (GPIO 14/15 plus a half-duplex circuit), you **must** add one GND wire between the Pi and the servo bus ground.
- **Power-on order:** power the servos first, then boot the Pi, or let the Pi start with torque disabled. On shutdown, disable servo torque before cutting logic power, so the servos do not jerk while the Pi is down.

### Gaps
- No Hiwonder documentation was found on BusLinker V3.0 isolation, ground topology, or the recommended grounding with an external SBC.

---

## 6. The 100 µF capacitor; bulk capacitance and placement; BusLinker and lead current limits; XT30/XT60; wire gauge

### Takeaway
- **Capacitor:** 100 µF is on the small side but not useless. Its main job is **damping LC voltage spikes at plug-in and during servo current steps**, not supplying 10 A peaks; the battery does that. Use a **470–1000 µF, ≥25 V, low-ESR electrolytic** at the point where power enters the servo bus.
- **Biggest unknown:** the **BusLinker's power path has no published current rating**. Route servo power around it through a small distribution point.
- **Connectors:** XT60 (30 A continuous) is ample. XT30 (15 A continuous) is ample for the logic pack.
- **Wire:** use 16–18 AWG from the battery to the distribution point.

### Cited Findings
- **Pololu's LC-spike application note:**
  - Lead inductance plus low-ESR ceramic capacitors cause ringing at plug-in. A 12 V supply on 36-inch leads spiked to **almost 40 V**, and a >13 V battery on short leads spiked to about **25 V**.
  - Remedies: add a bulk electrolytic in parallel (their example uses **100 µF electrolytic in parallel with 10 µF ceramic**), keep leads short and thick, add series resistance, or use a MOSFET electronic switch. The MOSFET switch cut the peak from about 40 V to under 20 V.

  Source: [Pololu – Understanding destructive LC voltage spikes](https://www.pololu.com/docs/0J16/all)
- **HTD-35H maximum working voltage:** 12.6 V. — [OpenELAB](https://openelab.io/products/htd-35h-bus-servo)
- **XT60:** 30 A continuous / 60 A burst for one minute (with 12 AWG, temperature rise under 60 °C). — [Holybro connector and wire rating](https://docs.holybro.com/power-module-and-pdb/power-module/connector-and-wire-rating); [Electrokit XT60](https://www.electrokit.com/en/stromkontakt-2-pol-xt60-30a-hane)
  - **Conflict:** ReadyMadeRC lists 60 A continuous / 100 A peak, which is likely a marketing figure. — [ReadyMadeRC XT60](https://www.readymaderc.com/products/details/86830-xt60-power-connectors-5-pair)
- **XT30:** **15 A continuous / 30 A peak.** — [Probots XT30PW](https://probots.co.in/amass-xt30pw-f-female-2pin-high-power-polarised-connector-15a.html); [Gogo:Tronics XT30](https://sparks.gogo.co.nz/catalog/Hardware-270/XT30-Connector-Pair---30-Amps-728.html)
- **BusLinker V3.0:** the spec lists only 5–14 V input via the DC jack or terminal, with **no maximum current**. — [OpenELAB BusLinker V3.0](https://openelab.io/a/s/products/buslinker-v3-servo-board)
- **Stall current wiring:** OpenELAB advises sizing wiring for the 3 A per-servo stall current. — [OpenELAB HTD-35H](https://openelab.io/products/htd-35h-bus-servo)

### Inferences
- **Why the capacitor is not for supplying peaks:**
  - A 10 A, 10 ms peak needs 100 mC of charge.
  - A 1000 µF capacitor sagging 0.5 V supplies only 0.5 mC.
  - Peaks must come from the battery, through low-resistance wiring.
  - What the capacitor does is **lower supply impedance at high frequency, absorb regenerative energy** (servos braking or back-driven during falls push current back onto the bus), and **damp plug-in ringing**.
- **The plug-in spike is the real hazard:**
  - A full 3S pack is 12.6 V, already at the servo's maximum rating.
  - Per Pololu's measurements, an undamped plug-in can ring to about 2x Vin (≈25 V).
  - A 100 µF electrolytic damps this somewhat. A larger, ≥25 V-rated electrolytic placed **right at the bus input terminal** (with short leads) damps it better.
  - A **16 V-rated capacitor is not adequate** for this; use 25 V or higher.
  - **Optional extras:**
    - a 15–16 V TVS diode (for example, SMBJ13A or SMBJ15A) across the bus;
    - an anti-spark XT60 connector;
    - a MOSFET switch rather than plugging the XT60 in live.
- **Placement:** one 470–1000 µF bulk capacitor at the power entry/distribution point, plus optionally 100–220 µF at the far end of each leg's daisy chain. That puts capacitance near the ankle and knee servos with the longest leads.
- **BusLinker power path:**
  - The DC jack (5.5 × 2.1 mm) is typically rated for only a few amps.
  - The screw terminal and PCB traces are unrated.
  - 24 A of stall current, or repeated 10 A peaks, through an unrated hobby PCB is a risk of trace heating and voltage drop.
  - **Recommendation:** make a small power distribution point (a 2-rail PCB, a Wago block, or a solder bus) fed directly from the XT60. Run a separate pair of 18–20 AWG leads from it to each leg's servo chain, preferably using a Hiwonder bus-servo hub/expansion board or a custom splitter. Use the BusLinker only for the signal and as one more branch, or validate it by measuring its terminal temperature at 10 A.
- **Servo leads and daisy chains:**
  - Hiwonder's stock 3-pin bus cables use thin wire and small connectors. Their gauge and rating were not found (see Gaps).
  - In a daisy chain, the first link carries the current of every downstream servo. For four servos on one leg: about 0.5–1.5 A walking, but up to 12 A in a stall.
  - **Recommendation:** **at most 4 servos per chain (one chain per leg), fed from the distribution point.** Keep the cables short. Consider upgrading the hip-to-distribution link to 20 AWG.
- **Wire gauge:**
  - 16 AWG silicone wire handles about 20+ A for short chassis runs, and 18 AWG about 15 A.
  - For 10 A peaks and 3–4 A average over less than 20 cm, **18 AWG is adequate and 16 AWG gives margin.** Use 22–24 AWG only for runs carrying a single servo.
  - These are general wiring-practice values; Holybro cites only 12 AWG with XT60 at 30 A.
- **XT60** on a 10 A system is about 3x oversized and can stay; it is robust and common. An XT30 (15 A continuous) would also work for the servo pack and saves a few grams. **XT30 on the logic pack** (<1 A) is fine.

### Gaps
- No official current rating was found for the BusLinker V3.0 terminal or traces, or for Hiwonder's 3-pin servo cable and connector.
- No Hiwonder recommendation for bulk capacitance on the bus was found.

---

## 7. Power switch and emergency stop for a small robot

### Takeaway
Use a three-layer setup:
- **(a)** A physical main switch on each battery's positive lead, rated ≥ peak current. For the servo pack, prefer a **MOSFET-based electronic switch or an anti-spark connector**, which also tames plug-in spikes.
- **(b)** A software e-stop that sends torque-off to all servos.
- **(c)** A low-voltage auto-sit or shutdown.

The physical switch must cut **servo power independently of the Pi**.

### Cited Findings
- Pololu measured that a **MOSFET-based electronic switch cut a 40 V plug-in spike to under 20 V** compared with a mechanical switch, and noted that **mechanical switch bounce adds repeated spikes**. — [Pololu LC spikes note](https://www.pololu.com/docs/0J16/all)
- The HTD-35H includes stall and over-temperature protection. That is a last-ditch servo-side safety, not a substitute for an e-stop. — [RobotShop](https://ca.robotshop.com/products/hiwonder-hiwonder-htd-35h-high-voltage-serial-bus-servo-35kg-torque-with-three-connectors-and-data-feedback)

### Inferences
- **Servo rail (recommended):**
  - Use an electronic power switch with a pushbutton, rated ≥15–20 A at ≥20 V (Pololu "Big MOSFET Slide Switch" or "Pushbutton Power Switch HP" families; not fetched in this session, so verify the ratings). Alternatively, use an XT60 "loop key" or anti-spark XT90-S style connector.
  - Mount a big, easily hit button on the back or top of the torso.
  - A mechanical rocker switch rated ≥20 A DC also works, but it bounces and is heavier. Many cheap rockers are rated for AC only, so check the **DC** rating.
- **Logic rail:** a small slide switch or the Pololu regulator's EN pin, plus a clean `sudo shutdown` path. Avoid hard power-off of the SD card while it is writing.
- **E-stop priority:**
  1. Hardware kill of servo power (the robot goes limp, which is safest for people but may damage the robot if it falls).
  2. Software torque-off broadcast to all servos, triggered by a gamepad button, a watchdog on lost policy heartbeat, or IMU tilt > about 45°.
  3. A low-voltage auto-sit.

  For RL testing, add a **watchdog**: if the control loop misses for more than about 100 ms, broadcast torque-off.
- **During early testing,** use a gantry or harness (a tether to an overhead bar). This reduces falls, which in turn reduces battery-impact and fire risk.

### Gaps
- No Pololu power-switch product pages were fetched, so switch current ratings are unverified here.
- No Hiwonder documentation was retrieved on a broadcast torque-off command or on behavior when bus communication is lost (whether servos hold their last position or go limp).

---

## Summary of recommendations (for quick reference; each is derived from the sections above)

1. **3S 2200 servo pack:** adequate. Use a ≥25C reputable pack (about 165–190 g). Expect about 25–35 min of walking. Cut off at 3.4–3.5 V/cell under load (≈10.2–10.5 V).
2. **Replace the LM2596** with a Pololu D36V50F5 or D24V50F5 (or a 5 V ≥5 A UBEC). If you keep the LM2596, set 5.2 V at the Pi under load, heatsink it, lock the trimpot, and monitor `vcgencmd get_throttled`.
3. **Keep two batteries,** with one common ground point and no servo current through the USB or Pi.
4. **Capacitor:** upgrade 100 µF to **470–1000 µF ≥25 V low-ESR** at the bus entry point. Optionally add a 15 V TVS and an anti-spark or MOSFET switch.
5. **Do not route all servo current through the BusLinker.** Add a power distribution point with 16–18 AWG feeds, one chain per leg, and at most 4 servos per chain.
6. **Monitoring:** software servo Vin readback, plus balance-lead buzzers on **both** packs. The 650 pack is easy to over-discharge.
7. **Switching:** add a main servo power switch (a MOSFET type is preferred), a software torque-off e-stop with watchdog, and a harness during early RL trials.
8. **Measure before trusting the numbers:** single-servo stall current, 8-servo standing current, and bus voltage at a 10 A peak on a near-empty pack.
