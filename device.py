"""
ADB-Steuerung fuer das Xiaomi (oder jedes andere Android-Geraet).

Voraussetzungen auf dem Xiaomi (MIUI / HyperOS):
  Einstellungen > Ueber das Telefon > 7x auf "OS-Version"/"MIUI-Version" tippen
  Einstellungen > Weitere Einstellungen > Entwickleroptionen:
     - USB-Debugging                              AN
     - USB-Debugging (Sicherheitseinstellungen)   AN   <- sonst gehen keine Taps!
     - Installation ueber USB                     AN   (optional)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time

import cv2
import numpy as np

GAME_PACKAGE = "com.grandgames.carmatch"


class Device:
    def __init__(self, serial: str | None = None, adb: str | None = None):
        self.adb = adb or shutil.which("adb") or "adb"
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
    def connect(self):
        out = subprocess.run([self.adb, "devices"], capture_output=True, text=True).stdout
        devs = [l.split()[0] for l in out.splitlines()[1:] if l.strip().endswith("device")]
        if not devs:
            raise RuntimeError(
                "Kein Android-Geraet gefunden. USB-Kabel/WLAN-Debugging pruefen "
                "und am Handy 'USB-Debugging zulassen' bestaetigen.")
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

    def game_running(self) -> bool:
        return GAME_PACKAGE in self.foreground()

    def start_game(self):
        self.shell(f"monkey -p {GAME_PACKAGE} -c android.intent.category.LAUNCHER 1")
        time.sleep(6)

    def stop_game(self):
        self.shell(f"am force-stop {GAME_PACKAGE}")
        time.sleep(1)

    def keep_awake(self):
        # Bildschirm an lassen, solange USB dran ist
        self.shell("svc power stayon usb")
