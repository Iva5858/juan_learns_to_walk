"""The Raspberry Pi's own health: temperature, undervoltage and throttling, CPU, memory, clocks and
Wi-Fi signal, from /sys, /proc and vcgencmd. Used by the dashboard and the self-test."""
import os
import subprocess
import time


def temperature():
    """SoC temperature [deg C], or None off a Pi."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000, 1)
    except (OSError, ValueError):
        return None


def throttle_flags():
    """vcgencmd get_throttled as an int (bit 0: undervoltage now, bit 2: throttled now, bit 16:
    undervoltage since boot, bit 18: throttled since boot), or None off a Pi."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1).stdout
        return int(out.strip().split("=")[1], 16)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def power():
    flags = throttle_flags()
    if flags is None:
        return {}
    return {"undervolt": bool(flags & 0x1), "throttled": bool(flags & 0x4), "undervolt_before": bool(flags & 0x10000)}


class SysStats:
    """CPU (total, per core, this program), memory, clocks, GPU memory and Wi-Fi signal, from /proc and
    vcgencmd. Cheap: read once a second (vcgencmd every 5 s)."""

    def __init__(self):
        self._cpu, self._self, self._n, self._vc = None, None, 0, {}

    @staticmethod
    def _cpu_times():
        out = []
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu"):
                    v = list(map(int, line.split()[1:]))
                    out.append((sum(v), v[3] + v[4]))  # total, idle + iowait
        return out

    def read(self):
        s = {}
        try:
            now = self._cpu_times()
            if self._cpu:
                busy = [100 * (1 - (i1 - i0) / max(t1 - t0, 1)) for (t0, i0), (t1, i1) in zip(self._cpu, now)]
                s["cpu"], s["cores"] = busy[0], busy[1:]
            self._cpu = now
            with open("/proc/self/stat") as f:  # this dashboard (all its threads), % of one core
                v = f.read().split(")")[1].split()
            ticks, wall = int(v[11]) + int(v[12]), time.monotonic()
            if self._self:
                s["self_cpu"] = 100 * (ticks - self._self[0]) / os.sysconf("SC_CLK_TCK") / max(wall - self._self[1], 1e-3)
            self._self = (ticks, wall)
            with open("/proc/loadavg") as f:
                s["load"] = float(f.read().split()[0])
            mem = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":")
                    mem[k] = int(v.split()[0])
            s["mem_used_mb"] = (mem["MemTotal"] - mem["MemAvailable"]) / 1024
            s["mem_total_mb"] = mem["MemTotal"] / 1024
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
                s["arm_mhz"] = int(f.read()) / 1000
        except (OSError, ValueError, KeyError, IndexError):
            pass
        try:
            with open("/proc/net/wireless") as f:
                lines = f.readlines()[2:]
            if lines:
                s["wifi_dbm"] = float(lines[0].split()[3].rstrip("."))
        except (OSError, ValueError, IndexError):
            pass
        if self._n % 5 == 0:  # vcgencmd: GPU (VideoCore) clock and memory split
            vc = {}
            for key, cmd in (("gpu_mhz", ["measure_clock", "core"]), ("gpu_mem_mb", ["get_mem", "gpu"])):
                try:
                    out = subprocess.run(["vcgencmd", *cmd], capture_output=True, text=True, timeout=1).stdout
                    val = out.strip().split("=")[1]
                    vc[key] = int(val) / 1e6 if key == "gpu_mhz" else int(val.rstrip("M"))
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    pass
            self._vc = vc
        self._n += 1
        return {**s, **self._vc}
