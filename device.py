"""
ADB-Steuerung fuer MuMu Player (Emulator auf dem PC), das Xiaomi oder jedes
andere Android-Geraet.

MuMu Player: nichts einzurichten. Der Bot findet MuMus eigenes adb.exe und
verbindet sich automatisch mit 127.0.0.1:16384 (MuMu 12 / MuMu 5) bzw. 7555.

Voraussetzungen auf dem Xiaomi (MIUI / HyperOS):
  Einstellungen > Ueber das Telefon > 7x auf "OS-Version"/"MIUI-Version" tippen
  Einstellungen > Weitere Einstellungen > Entwickleroptionen:
     - USB-Debugging                              AN
     - USB-Debugging (Sicherheitseinstellungen)   AN   <- sonst gehen keine Taps!
     - Installation ueber USB                     AN   (optional)
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

GAME_PACKAGE = "com.grandgames.carmatch"

# MuMu: erste Instanz 16384, weitere +32; aeltere MuMu-Versionen 7555
EMULATOR_PORTS = [16384, 16416, 16448, 16480, 7555, 5555, 62001, 21503]


def find_adb():
    """adb im PATH, im Bot-Ordner oder aus einer MuMu-Installation."""
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "adb.exe" if sys.platform == "win32" else "adb")
    if os.path.exists(local):
        return local
    found = shutil.which("adb")
    if found:
        return found
    if sys.platform == "win32":
        pats = []
        for drive in "CDEFG":
            for base in (r"Program Files", r"Program Files (x86)", ""):
                root = os.path.join(f"{drive}:\\", base) if base else f"{drive}:\\"
                pats += [os.path.join(root, "Netease", "**", "adb.exe"),
                         os.path.join(root, "MuMu*", "**", "adb.exe")]
        for p in pats:
            hits = sorted(glob.glob(p, recursive=True))
            if hits:
                # bevorzugt shell\adb.exe bzw. nx_main\adb.exe
                hits.sort(key=lambda h: (("shell" not in h.lower()) and ("nx_main" not in h.lower()), len(h)))
                return hits[0]
    return "adb"


class Device:
    def __init__(self, serial: str | None = None, adb: str | None = None):
        self.adb = adb or find_adb()
        self.serial = serial
        self._raw_ok = True
        self.size = None

    # -------------------------------------------------------------- intern
    def _cmd(self, *args):
        c = [self.adb]
        if self.serial:
            c += ["-s", self.serial]
        return c + list(args)

    def shell(self, cmd: str, timeout=15) -> str:
        r = subprocess.run(self._cmd("shell", cmd), capture_output=True, timeout=timeout)
        return r.stdout.decode("utf-8", "ignore")

    # -------------------------------------------------------------- Verbindung
    def _devices(self):
        out = subprocess.run([self.adb, "devices"], capture_output=True, text=True, timeout=20).stdout
        return [l.split()[0] for l in out.splitlines()[1:] if l.strip().endswith("device")]

    def connect(self):
        print(f"adb: {self.adb}")
        try:
            devs = self._devices()
        except FileNotFoundError:
            raise RuntimeError("adb nicht gefunden. MuMu Player installieren oder adb.exe "
                               "in den Bot-Ordner legen.")
        if self.serial and ":" in self.serial and self.serial not in devs:
            subprocess.run([self.adb, "connect", self.serial], capture_output=True, timeout=10)
            devs = self._devices()
        if not devs:
            # Emulator (MuMu) automatisch verbinden
            for port in EMULATOR_PORTS:
                r = subprocess.run([self.adb, "connect", f"127.0.0.1:{port}"],
                                   capture_output=True, text=True, timeout=10)
                if "connected" in r.stdout and "cannot" not in r.stdout:
                    time.sleep(1)
                    devs = self._devices()
                    if devs:
                        print(f"Emulator verbunden: 127.0.0.1:{port}")
                        break
        if not devs:
            raise RuntimeError(
                "Kein Android-Geraet gefunden. Ist MuMu Player gestartet? "
                "(Beim Handy: USB-Debugging pruefen und 'zulassen' bestaetigen.)")
        if self.serial is None:
            if len(devs) > 1:
                print(f"Mehrere Geraete: {devs} - nehme {devs[0]} (mit --serial waehlen)")
            self.serial = devs[0]
        img = self.screenshot()
        self.size = (img.shape[1], img.shape[0])
        return self.serial

    # -------------------------------------------------------------- Screenshot
    def screenshot(self) -> np.ndarray:
        """Schnell: rohe Pixel (ohne PNG-Kompression). Fallback: PNG."""
        if self._raw_ok:
            try:
                r = subprocess.run(self._cmd("exec-out", "screencap"), capture_output=True, timeout=10)
                data = r.stdout
                w = int.from_bytes(data[0:4], "little")
                h = int.from_bytes(data[4:8], "little")
                n = w * h * 4
                if w > 0 and h > 0 and len(data) >= n + 12:
                    off = len(data) - n           # Header 12 oder 16 Bytes
                    arr = np.frombuffer(data, np.uint8, n, off).reshape(h, w, 4)
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                self._raw_ok = False
            except Exception:
                self._raw_ok = False
        r = subprocess.run(self._cmd("exec-out", "screencap", "-p"), capture_output=True, timeout=15)
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Screenshot fehlgeschlagen")
        return img

    # -------------------------------------------------------------- Eingaben
    def tap(self, x, y):
        out = self.shell(f"input tap {int(x)} {int(y)}")
        if "INJECT_EVENTS" in out or "SecurityException" in out:
            raise RuntimeError(
                "Xiaomi blockiert Taps: In den Entwickleroptionen "
                "'USB-Debugging (Sicherheitseinstellungen)' einschalten.")

    def back(self):
        self.shell("input keyevent 4")

    # -------------------------------------------------------------- App
    def foreground(self) -> str:
        """Paket/Activity im Vordergrund, z.B. 'com.grandgames.carmatch/...'."""
        out = self.shell("dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'")
        m = re.search(r"([a-zA-Z0-9_.]+/[a-zA-Z0-9_.$]+)", out)
        return m.group(1) if m else ""

    def game_running(self, package=GAME_PACKAGE) -> bool:
        return package in self.foreground()

    def start_game(self, package=GAME_PACKAGE):
        self.shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")
        time.sleep(6)

    def stop_game(self, package=GAME_PACKAGE):
        self.shell(f"am force-stop {package}")
        time.sleep(1)

    def keep_awake(self):
        # Bildschirm an lassen, solange USB dran ist
        self.shell("svc power stayon usb")
