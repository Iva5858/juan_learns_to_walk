#!/usr/bin/env python3
"""Juan's robot dashboard: a web page to monitor, calibrate and drive the robot (robot/body.py does the
work; robot/actions.py lists everything the page can ask for).

Run on the Pi, from the project folder:   python3 -m robot.dashboard
Then open                                  http://<pi-name>.local:8000
Without the robot (any computer):          python3 -m robot.dashboard --fake     (http://localhost:8000)

    GET  /api/state             everything, once
    GET  /api/events            the state, pushed (Server-Sent Events): "fast" 10 times a second (joints,
                                IMU, walking, mode), "full" once a second (also health, settings, lists)
    GET  /api/actions           every action: name, label, group, parameters, confirmation
    POST /api/<action>          do it; parameters as JSON (or the query string); reply {"error": ...}
    GET  /camera.mjpg, /camera-stabilized.mjpg      the camera
    GET  /                      the page (robot/web/); /classic: the previous page (robot/dashboard.html)
The walking policy stands still if nothing has called /api/heartbeat (or an action, or /state) for
2 s, and stops after 6 s. (GET /<action>?... and /state still work for the current page.)
"""
import argparse
import json
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from robot import actions
from robot.body import PI_TEMP_LIMIT, SERVO_TEMP_LIMIT, Robot
from robot.joints import BY_ID

PORT = 8000
HERE = os.path.dirname(os.path.realpath(__file__))
WEB = os.path.join(HERE, "web")
CLASSIC = os.path.join(HERE, "dashboard.html")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}
FAST_S, FULL_S = 0.1, 1.0
robot = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser stopped waiting (slow request, page reloaded): what was asked still happened

    def file(self, path):
        path = os.path.realpath(path)
        if not (path == CLASSIC or path.startswith(WEB + os.sep)) or not os.path.isfile(path):
            return self.reply(404)
        with open(path, "rb") as f:
            self.reply(200, f.read(), TYPES.get(os.path.splitext(path)[1], "application/octet-stream"))

    def json(self, code, data):
        self.reply(code, json.dumps(data).encode())

    def action(self, name, raw):
        robot.walker.heartbeat()  # whoever acts is there
        self.json(*actions.run(robot, name, raw))

    def camera(self, stabilized=False):
        if stabilized and robot.camera.stabilizer is None:
            return self.reply(409, b"Camera stabilization requires calibration and OpenCV", "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for frame in robot.camera.stream(stabilized=stabilized):
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame))
                self.wfile.write(frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def events(self):
        """The state as Server-Sent Events until the page goes away. Not a heartbeat: a stream can stay
        open for a while after Wi-Fi drops, so only the page's own calls count as being there."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        last_full = 0.0
        try:
            while True:
                now = time.monotonic()
                full = now - last_full >= FULL_S
                last_full = now if full else last_full
                data = json.dumps(robot.state(fast=not full))
                self.wfile.write(f"event: {'full' if full else 'fast'}\ndata: {data}\n\n".encode())
                self.wfile.flush()
                time.sleep(FAST_S)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        self.close_connection = True

    def do_GET(self):
        url = urlparse(self.path)
        p = url.path
        if p in ("/", "/classic") or p.startswith("/web/"):
            return self.file(CLASSIC if p == "/classic" else os.path.join(WEB, "index.html" if p == "/" else p[len("/web/"):]))
        if p in ("/state", "/api/state"):
            robot.walker.heartbeat()  # the page is still there
            return self.json(200, robot.state())
        if p == "/api/events":
            return self.events()
        if p == "/api/actions":
            return self.json(200, [a.describe() for a in actions.ACTIONS.values()])
        if p == "/camera.mjpg":
            return self.camera()
        if p == "/camera-stabilized.mjpg":
            return self.camera(stabilized=True)
        # the current page's GET /<action>?...
        name = p.strip("/")
        if name in actions.ACTIONS:
            return self.action(name, {k: v[0] for k, v in parse_qs(url.query).items()})
        self.reply(404)

    def do_POST(self):
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return self.reply(404)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            raw = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if not isinstance(raw, dict):
                raise ValueError
        except ValueError:
            return self.json(400, {"error": "parameters must be a JSON object"})
        raw.update({k: v[0] for k, v in parse_qs(url.query).items()})
        self.action(url.path[len("/api/"):], raw)

    def log_message(self, *args):
        pass


class DualStackServer(ThreadingHTTPServer):
    """IPv4 and IPv6, so http://name.local:8000 works either way."""
    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def serve(port):
    try:
        return DualStackServer(("::", port), Handler)
    except OSError:
        try:
            return ThreadingHTTPServer(("0.0.0.0", port), Handler)
        except OSError:
            raise SystemExit(f"Port {port} is busy: stop any other servo script first (Ctrl+C in its window).") from None


def main():
    global robot
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="stand-in hardware (robot/fake.py): no robot needed")
    p.add_argument("--port", type=int, default=PORT)
    a = p.parse_args()
    server = serve(a.port)
    if a.fake:
        from robot import fake
        fake_dir = os.path.join(HERE, "logs", "fake")   # its calibration and walk logs
        os.makedirs(fake_dir, exist_ok=True)
        cal_path = os.path.join(fake_dir, "calibration.json")
        robot = Robot(fake.hardware(cal_path), cal_path)
        robot.walker.log_dir = fake_dir
    else:
        robot = Robot()
    for bus in robot.buses:
        print(f"Bus {bus.port}: servos {', '.join(map(str, bus.ids))}")
    missing = sorted(set(BY_ID) - set(robot.bus_of))
    if missing:
        print(f"Not found: servos {', '.join(map(str, missing))}")
    print(f"Calibration: {'zero set' if robot.cal.zero_set else 'NOT calibrated yet (set zero first)'}"
          + (f" ({robot.cal_path}: the fake robot's own)" if a.fake else ""))
    print(f"Protection:  servos relax above {SERVO_TEMP_LIMIT} C, everything relaxes if the Pi reaches {PI_TEMP_LIMIT} C")
    host = "localhost" if a.fake else socket.gethostname().removesuffix(".local") + ".local"
    print(f"Dashboard:   http://{host}:{a.port}   (Ctrl+C stops and holds position)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        robot.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
