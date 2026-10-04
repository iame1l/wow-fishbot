"""Screen capture backends for the fish bot.

On Wayland the bot can only reach the screen through the XDG desktop portal.
``pyscreenshot`` uses the portal's ``Screenshot`` method, which takes a full
screen shot, encodes it as a PNG and writes it to disk on *every* call. That is
where the 3-4 seconds per grab come from, and it is slow enough to miss most
fish bites.

This module instead opens a screen cast (portal ``ScreenCast`` + pipewire) once
at startup and then reads live frames from it through gstreamer. After that a
grab is just a crop out of the newest frame, i.e. a few milliseconds.

The screen cast delivers frames in the device pixels of the captured output,
while the rest of the bot (and pyautogui) work in the coordinate space of a
portal screenshot. When the two spaces do not line up 1:1 a single portal
screenshot is taken and used to calibrate the scale/offset between them, and
the result is cached, so the rest of the bot keeps working in the same
coordinates as before without paying for it on every grab.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import pyscreenshot as ImageGrab

try:  # jeepney is only needed for the fast screen cast path
    from jeepney import DBusAddress, MatchRule, new_method_call
    from jeepney.bus_messages import message_bus
    from jeepney.io.blocking import Proxy, open_dbus_connection
except ImportError:  # pragma: no cover - the fallback keeps working without it
    open_dbus_connection = None


class CaptureError(RuntimeError):
    """Raised when a capture backend cannot be started."""


# --------------------------------------------------------------------------- #
# Slow but dependable fallback: one portal screenshot per grab
# --------------------------------------------------------------------------- #
CAPTURE_BACKENDS = ["freedesktop_dbus", "pil", "grim", "gnome_dbus"]


class PortalCapture:
    """Full portal screenshot per grab (what the bot used to do)."""

    name = "portal screenshot"

    def __init__(self, debug=False):
        self.debug = debug
        self._backend = None
        self.full = None
        for name in CAPTURE_BACKENDS:
            try:
                im = ImageGrab.grab(backend=name)
                arr = np.asarray(im)
                if arr.max() > 0:  # reject all-black frames
                    self._backend = name
                    self.full = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    print(f"Capture backend: {name} ({im.size[0]}x{im.size[1]})")
                    break
            except Exception as err:
                if debug:
                    print(f"Capture backend {name} unavailable: {err}")
        if self._backend is None:
            raise CaptureError("No working screenshot backend found")

    def grab(self, region):
        bbox = (region["x1"], region["y1"], region["x2"], region["y2"])
        im = ImageGrab.grab(bbox=bbox, backend=self._backend).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Fast path: portal ScreenCast -> pipewire -> gstreamer -> raw frames
# --------------------------------------------------------------------------- #
PORTAL_PATH = "/org/freedesktop/portal/desktop"
PORTAL_BUS = "org.freedesktop.portal.Desktop"
REQUESTS_IFACE = "org.freedesktop.portal.Request"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"

RAW_FORMAT = "BGRx"  # 4 bytes per pixel: no stride padding surprises
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".screencast_cache.json")
THUMB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".screencast_thumb.png")
# Coarse candidates start close enough that the refinement below can bridge the
# remaining gap between candidates.
SCALE_GRID = [round(0.30 * 1.15 ** i, 4) for i in range(16)]  # 0.30 .. 2.62


def _unwrap(value):
    """Portal returns a{sv} values variant wrapped as (signature, value)."""
    sig_chars = set("ybnqiuxtdsogahv{}()")
    while (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and value[0]
        and all(c in sig_chars for c in value[0])
    ):
        value = value[1]
    return value


def _ncc(a, b):
    """Normalized cross correlation of two equally sized grayscale images."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return -1.0
    return float((a * b).sum() / denom)


def _gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


class _Portal:
    """Minimal blocking client for the portal request/response dance."""

    def __init__(self):
        if open_dbus_connection is None:
            raise CaptureError("jeepney is not installed")
        self.conn = open_dbus_connection(bus="SESSION", enable_fds=True)
        self.sender = self.conn.unique_name[1:].replace(".", "_")
        self._tokens = itertools.count(1)
        address = DBusAddress(PORTAL_PATH, bus_name=PORTAL_BUS)
        self._screencast = address.with_interface(SCREENCAST_IFACE)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def call(self, method, signature, options, session=None, timeout=180):
        """Call a ScreenCast request method, return (response_code, results)."""
        token = f"fishbot{next(self._tokens)}"
        handle = f"{PORTAL_PATH}/request/{self.sender}/{token}"
        options = dict(options, handle_token=("s", token))
        rule = MatchRule(type="signal", interface=REQUESTS_IFACE, path=handle)
        Proxy(message_bus, self.conn).AddMatch(rule)
        with self.conn.filter(rule) as responses:
            if session is None:
                args = (options,)
            elif method == "Start":
                args = (session, "", options)
            else:
                args = (session, options)
            reply = self.conn.send_and_get_reply(
                new_method_call(self._screencast, method, signature, args)
            )
            if str(reply.body[0]) != handle:
                raise CaptureError(f"unexpected request handle {reply.body[0]}")
            signal = self.conn.recv_until_filtered(responses, timeout=timeout)
        return signal.body

    def pipewire_fd(self, session):
        reply = self.conn.send_and_get_reply(
            new_method_call(self._screencast, "OpenPipeWireRemote", "oa{sv}", (session, {}))
        )
        return reply.body[0]


def portal_screen_size():
    """Size of the coordinate space pyautogui (and a portal screenshot) uses."""
    try:
        import pyautogui

        size = pyautogui.size()
        return int(size[0]), int(size[1])
    except Exception:
        return None


def kscreen_outputs():
    """Geometry of every enabled output (device pixels + logical position)."""
    try:
        out = subprocess.run(
            ["kscreen-doctor", "-j"], capture_output=True, text=True, timeout=15
        )
        data = json.loads(out.stdout)
    except Exception:
        return None
    outputs = []
    for item in data.get("outputs", []):
        size = item.get("size") or {}
        if not item.get("enabled") or not size.get("width"):
            continue
        pos = item.get("pos") or {}
        outputs.append(
            {
                "x": int(pos.get("x", 0)),
                "y": int(pos.get("y", 0)),
                "w": int(size["width"]),   # device pixels (the mode)
                "h": int(size["height"]),
                "scale": float(item.get("scale") or 1.0),
            }
        )
    return outputs or None


class ScreenCastCapture:
    """Persistent pipewire screen cast, cropped to any region on demand."""

    name = "screencast"

    def __init__(
        self, reference_provider, fps=4, roi=None, debug=False, force_scale=None
    ):
        self.debug = debug
        self.fps = max(1, int(fps))
        self.roi = roi
        self.force_scale = force_scale
        self._reference_provider = reference_provider
        self.reference = None  # portal screenshot, only needed for calibration
        self.portal_size = None  # size of the coordinate space we map into
        self.scale = 1.0  # portal = stream * scale + offset
        self.offset = (0.0, 0.0)
        self._portal = None
        self._session = None
        self._fd = None
        self._node = None
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stream_size = None
        self.restore_token = None

    # -- start up ----------------------------------------------------------- #

    def start(self):
        cache = self._load_cache()
        self._open_session(cache)
        self._spawn_pipeline()
        self._wait_for_frame(timeout=15)
        mapping = self._geometry_mapping()
        if mapping:
            self.scale, self.offset = mapping
            self.portal_size = portal_screen_size()
            print(
                "Screen cast mapping from screen geometry: "
                f"scale={self.scale:.4f} offset=({self.offset[0]:.0f},{self.offset[1]:.0f})"
            )
        elif cache and self._cache_matches(cache):
            self.scale = float(cache["scale"])
            self.offset = (float(cache["offset"][0]), float(cache["offset"][1]))
            cached_size = cache.get("portal_size") or portal_screen_size()
            self.portal_size = tuple(int(v) for v in cached_size)
            print(
                f"Reusing cached calibration (scale={self.scale:.3f}, "
                f"offset={self.offset[0]:.0f},{self.offset[1]:.0f})"
            )
            if not self._verify_cache():
                print("Cached calibration does not match the screen, re-calibrating...")
                self.reference = self._reference_provider()
                self._calibrate()
        else:
            self.reference = self._reference_provider()
            self.portal_size = (self.reference.shape[1], self.reference.shape[0])
            self._calibrate()
        self._apply_force_scale()
        self._save_cache()
        return self

    def _apply_force_scale(self):
        """Override the auto-calibrated scale when one is configured."""
        if self.force_scale is None:
            return
        forced = float(self.force_scale)
        if forced == self.scale:
            return
        print(f"Forcing screen cast scale {self.scale:.4f} -> {forced:.4f}")
        self.scale = forced

    def _geometry_mapping(self):
        """Map stream pixels onto portal pixels from the screen geometry alone.

        This does not look at the picture, so an animated game screen cannot
        throw the alignment off (the picture based fallback below can).
        """
        portal = portal_screen_size()
        outputs = kscreen_outputs()
        if not portal or not outputs:
            return None
        logical_w = max(o["x"] + o["w"] / o["scale"] for o in outputs)
        logical_h = max(o["y"] + o["h"] / o["scale"] for o in outputs)
        if logical_w <= 0 or logical_h <= 0:
            return None
        pscale = portal[0] / logical_w  # portal pixels per logical pixel
        if abs(portal[1] / logical_h - pscale) > 0.03:
            return None
        sw, sh = self._stream_size
        candidates = []
        for o in outputs:  # stream shows one output, in its device pixels
            if abs(o["w"] - sw) <= 2 and abs(o["h"] - sh) <= 2:
                candidates.append(
                    (pscale / o["scale"], (o["x"] * pscale, o["y"] * pscale))
                )
        if abs(logical_w - sw) <= 2 and abs(logical_h - sh) <= 2:
            candidates.append((pscale, (0.0, 0.0)))  # stream shows the whole desktop
        unique = []
        for scale, offset in candidates:
            if not any(
                abs(scale - s2) < 1e-3
                and abs(offset[0] - o2[0]) < 0.5
                and abs(offset[1] - o2[1]) < 0.5
                for s2, o2 in unique
            ):
                unique.append((scale, offset))
        if not unique:
            return None
        if len(unique) == 1 or self.roi is None:
            return unique[0]
        for scale, offset in unique:  # several outputs have the same size: use the ROI
            x1 = (self.roi["x1"] - offset[0]) / scale
            y1 = (self.roi["y1"] - offset[1]) / scale
            x2 = (self.roi["x2"] - offset[0]) / scale
            y2 = (self.roi["y2"] - offset[1]) / scale
            if 0 <= x1 and 0 <= y1 and x2 <= sw and y2 <= sh:
                return scale, offset
        return unique[0]

    def _verify_cache(self):
        """Compare the cached region of interest against a live stream frame.

        This catches a stale calibration (for example when the screen cast
        picked another monitor that happens to have the same resolution).
        """
        try:
            thumb = cv2.imread(THUMB_FILE, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return True
        if thumb is None or self.roi is None:
            return True
        frame = self._wait_for_frame(timeout=5)
        region = self._roi_region()
        crop = self._crop_stream(frame, region, self.scale, self.offset, thumb.shape[::-1])
        if crop is None:
            return False
        score = _ncc(_gray(crop), thumb)
        if self.debug:
            print(f"Cache verification: score={score:.3f}")
        return score > 0.6

    def _open_session(self, cache):
        self._portal = _Portal()
        _, results = self._portal.call(
            "CreateSession", "a{sv}", {"session_handle_token": ("s", "fishbot")}
        )
        self._session = _unwrap(results["session_handle"])

        options = {"types": ("u", 1), "multiple": ("b", False), "cursor_mode": ("u", 1)}
        if cache and cache.get("restore_token"):
            options["restore_token"] = ("s", cache["restore_token"])
        code, results = self._portal.call(
            "SelectSources", "oa{sv}", options, session=self._session
        )
        if code != 0 and "restore_token" in options:
            options.pop("restore_token")  # stale token: ask with a fresh picker
            code, results = self._portal.call(
                "SelectSources", "oa{sv}", options, session=self._session
            )
        if code != 0:
            raise CaptureError(f"SelectSources failed with code {code}")

        if not (cache and cache.get("restore_token")):
            print("Note: if a screen cast permission dialog appears, pick the monitor with the game")
        code, results = self._portal.call("Start", "osa{sv}", {}, session=self._session)
        if code != 0:
            raise CaptureError(f"ScreenCast Start failed with code {code}")
        streams = _unwrap(results["streams"])
        self._node = int(_unwrap(streams[0][0]))
        props = dict(streams[0][1])
        self._stream_size = tuple(int(v) for v in _unwrap(props["size"]))
        self.restore_token = _unwrap(results.get("restore_token"))
        self._fd = int(self._portal.pipewire_fd(self._session).to_raw_fd())
        if self.debug:
            print(f"ScreenCast node={self._node} size={self._stream_size} props={props}")

    def _spawn_pipeline(self):
        if self._stream_size is None:
            raise CaptureError("stream size unknown")
        cmd = [
            "gst-launch-1.0",
            "-q",
            "pipewiresrc",
            f"fd={self._fd}",
            f"path={self._node}",
            "do-timestamp=false",
            "keepalive-time=200",
            "!",
            "videoconvert",
            "!",
            "videorate",
            f"max-rate={self.fps}",
            "!",
            f"video/x-raw,framerate={self.fps}/1,format={RAW_FORMAT}",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=(self._fd,),
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        width, height = self._stream_size
        frame_bytes = width * height * 4  # BGRx
        read = self._proc.stdout.read
        while not self._stop.is_set():
            buf = read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 4)
            with self._lock:
                self._frame = frame
                self._seq += 1

    def _wait_for_frame(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame
            time.sleep(0.02)
        raise CaptureError("no frames received from the screen cast")

    # -- calibration -------------------------------------------------------- #

    def _load_cache(self):
        try:
            with open(TOKEN_FILE) as fh:
                return json.load(fh)
        except Exception:
            return None

    def _cache_matches(self, cache):
        if list(self._stream_size) != list(cache.get("stream_size", [])):
            return False
        if self.restore_token != cache.get("restore_token"):
            return False
        current = portal_screen_size()
        cached = cache.get("portal_size")
        if current and cached and list(current) != list(cached):
            return False
        if not (current or cached):
            return False  # without a known coordinate space the cache is useless
        return True

    def _save_cache(self):
        try:
            with open(TOKEN_FILE, "w") as fh:
                json.dump(
                    {
                        "restore_token": self.restore_token,
                        "stream_size": list(self._stream_size),
                        "portal_size": list(portal_screen_size() or ()),
                        "scale": self.scale,
                        "offset": [self.offset[0], self.offset[1]],
                    },
                    fh,
                )
            if self.reference is not None and self.roi is not None:
                region = self._roi_region()
                ref_crop = self.reference[
                    region["y1"]:region["y2"], region["x1"]:region["x2"]
                ]
                if ref_crop.size:
                    k = min(1.0, 320.0 / ref_crop.shape[1])
                    thumb = cv2.resize(
                        _gray(ref_crop),
                        (
                            max(int(ref_crop.shape[1] * k), 16),
                            max(int(ref_crop.shape[0] * k), 16),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imwrite(THUMB_FILE, thumb)
        except Exception as err:
            if self.debug:
                print(f"Could not cache calibration: {err}")

    def _calibrate(self):
        """Locate the region of interest inside the screen cast stream."""
        if self.roi is None:
            raise CaptureError("calibration needs a region of interest")
        frame = self._wait_for_frame(timeout=10)
        region = self._roi_region()
        ref_crop = _gray(self.reference[region["y1"]:region["y2"], region["x1"]:region["x2"]])
        if ref_crop.size == 0:
            raise CaptureError("empty region of interest")
        stream_gray = _gray(frame)

        # 1) Coarse sweep: at which scale does the game area show up at all?
        hits = []
        for scale in SCALE_GRID:
            hit = self._find_roi(stream_gray, ref_crop, region, scale, width=160)
            if hit:
                hits.append(hit)
        hits.sort(key=lambda h: h[0], reverse=True)
        if not hits or hits[0][0] < 0.3:
            raise CaptureError(
                "could not find the game area in the screen cast - "
                "did you pick the monitor with the game in the dialog?"
            )
        if self.debug:
            print(f"Coarse hits: {[(round(h[0], 3), round(h[1], 3)) for h in hits[:3]]}")

        # 2) Refine the best coarse candidates, then keep the strongest one.
        best = None
        for _score, seed_scale, seed_offset in hits[:3]:
            for step in range(-8, 9):
                scale = seed_scale * (1.0 + step * 0.01)
                hit = self._find_roi(
                    stream_gray, ref_crop, region, scale, width=800,
                    window=seed_offset, margin=120,
                )
                if hit and (best is None or hit[0] > best[0]):
                    best = hit
        if best is None:
            raise CaptureError("screen cast alignment failed")
        self.scale, self.offset = best[1], best[2]

        # 3) Verify at full resolution: the mapping has to really line up.
        score = self._roi_score(frame, region, self.scale, self.offset, ref_crop, None)
        if score < 0.5:
            raise CaptureError(f"screen cast alignment too weak (score={score:.2f})")
        if self.debug:
            print(
                f"Calibrated: score={score:.3f} scale={self.scale:.4f} "
                f"offset=({self.offset[0]:.1f},{self.offset[1]:.1f})"
            )

    def _find_roi(self, stream_gray, ref_crop, region, scale, width, window=None, margin=120):
        """Search for the reference region inside the stream at one candidate scale."""
        ik = width / ref_crop.shape[1]
        tpl = cv2.resize(
            ref_crop,
            (max(int(ref_crop.shape[1] * ik), 16), max(int(ref_crop.shape[0] * ik), 16)),
            interpolation=cv2.INTER_AREA,
        )
        img = cv2.resize(
            stream_gray, None, fx=ik * scale, fy=ik * scale, interpolation=cv2.INTER_AREA
        )
        if img.shape[0] < tpl.shape[0] or img.shape[1] < tpl.shape[1]:
            return None
        x0 = y0 = 0
        if window is not None:
            px = (region["x1"] - window[0]) * ik
            py = (region["y1"] - window[1]) * ik
            m = margin * ik
            x0 = int(max(px - m, 0))
            y0 = int(max(py - m, 0))
            x1 = int(min(px + tpl.shape[1] + m, img.shape[1]))
            y1 = int(min(py + tpl.shape[0] + m, img.shape[0]))
            if x1 - x0 < tpl.shape[1] or y1 - y0 < tpl.shape[0]:
                return None
            img = img[y0:y1, x0:x1]
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if not np.isfinite(score):
            return None
        ox = region["x1"] - (loc[0] + x0) / ik
        oy = region["y1"] - (loc[1] + y0) / ik
        return float(score), scale, (ox, oy)

    def _roi_region(self):
        width, height = self.portal_size
        return {
            "x1": max(int(self.roi["x1"]), 0),
            "y1": max(int(self.roi["y1"]), 0),
            "x2": min(int(self.roi["x2"]), width),
            "y2": min(int(self.roi["y2"]), height),
        }

    def _roi_score(self, frame, region, scale, offset, ref_cmp, out_size):
        crop = self._crop_stream(frame, region, scale, offset, out_size)
        if crop is None:
            return -1.0
        return _ncc(_gray(crop), _gray(ref_cmp))

    # -- grabbing ----------------------------------------------------------- #

    def _crop_stream(self, frame, region, scale, offset, out_size=None):
        """Crop a portal space region from a stream frame, resized back to portal size."""
        x1 = int(round((region["x1"] - offset[0]) / scale))
        y1 = int(round((region["y1"] - offset[1]) / scale))
        x2 = int(round((region["x2"] - offset[0]) / scale))
        y2 = int(round((region["y2"] - offset[1]) / scale))
        h, w = frame.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        crop = frame[y1:y2, x1:x2, :3]
        if out_size is None:
            out_size = (region["x2"] - region["x1"], region["y2"] - region["y1"])
        return cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)

    def grab(self, region):
        frame = self._frame
        if frame is None:
            frame = self._wait_for_frame(timeout=5)
        img = self._crop_stream(frame, region, self.scale, self.offset)
        if img is None:
            raise CaptureError("requested region is outside the screen cast")
        return np.ascontiguousarray(img)

    def close(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
        if self._portal is not None:
            self._portal.close()
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass


def create_capture(
    roi=None, fps=4, debug=False, use_screencast=True, force_scale=None
):
    """Return the fastest capture backend that works on this machine."""
    holder = {}

    def reference_provider():
        if "portal" not in holder:
            holder["portal"] = PortalCapture(debug=debug)
        return holder["portal"].full

    if use_screencast:
        capture = ScreenCastCapture(
            reference_provider, fps=fps, roi=roi, debug=debug, force_scale=force_scale
        )
        try:
            capture.start()
            print(
                "Fast screen capture ready "
                f"(scale={capture.scale:.3f}, {capture.fps} fps stream)"
            )
            return capture
        except Exception as err:
            print(f"Screen cast unavailable ({err}); using slow portal captures")
            capture.close()
    if "portal" not in holder:
        holder["portal"] = PortalCapture(debug=debug)
    return holder["portal"]
