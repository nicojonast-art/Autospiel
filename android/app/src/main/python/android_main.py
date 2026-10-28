"""
Startpunkt des Bots in der Android-App (Chaquopy).

Die Java-Klasse de.nico.autospiel.Bridge liefert Bildschirmfotos, fuehrt Taps
ueber die Bedienungshilfe aus und zeigt Meldungen in der App an.
Erkennung und Loeser sind dieselben Dateien wie in der PC-Version:
  Yarn Loop  -> yarn_vision.py, yarn_solver.py, yarn_bot.py
  Car Match  -> vision.py, solver.py, bot.py
"""
import json
import time as _time
import traceback

import cv2
import numpy as np
from java import jclass

Bridge = jclass("de.nico.autospiel.Bridge")

GAMES = {
    "yarn": "com.combo.yarnflow",
    "car": "com.grandgames.carmatch",
}


class StopBot(Exception):
    """Wird geworfen, sobald in der App auf Stopp getippt wurde."""


class _Zeit:
    """Ersatz fuer das time-Modul der Bots: Warten bricht bei Stopp sofort ab."""

    time = staticmethod(_time.time)
    strftime = staticmethod(_time.strftime)
    monotonic = staticmethod(_time.monotonic)

    @staticmethod
    def sleep(seconds):
        end = _time.time() + max(0.0, float(seconds))
        while True:
            if not Bridge.isRunning():
                raise StopBot()
            rest = end - _time.time()
            if rest <= 0:
                return
            _time.sleep(min(0.2, rest))


def _log(msg):
    Bridge.log(str(msg))


class AndroidDevice:
    """Gleiche Schnittstelle wie device.Device (ADB), aber direkt auf dem Handy."""

    serial = "android"
    size = None

    def __init__(self, package):
        self.package = package

    def screenshot(self):
        data = None
        for _ in range(4):
            if not Bridge.isRunning():
                raise StopBot()
            data = Bridge.grab()
            if data is not None:
                break
            _Zeit.sleep(0.3)
        if data is None:
            raise RuntimeError("Kein Bild von der Bildschirmaufnahme")
        raw = np.frombuffer(data, dtype=np.int8).view(np.uint8)
        w = int.from_bytes(raw[0:4].tobytes(), "big")
        h = int.from_bytes(raw[4:8].tobytes(), "big")
        rgba = raw[8:8 + w * h * 4].reshape(h, w, 4)
        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)   # Kopie -> Java-Puffer wieder frei
        self.size = (w, h)
        return img

    def tap(self, x, y):
        if not Bridge.tap(int(x), int(y)):
            _log("Tippen fehlgeschlagen – ist die Bedienungshilfe an?")

    def back(self):
        Bridge.back()

    def foreground(self):
        return str(Bridge.foreground() or "")

    def game_running(self, package=None):
        return (package or self.package) in self.foreground()

    def start_game(self, package=None):
        Bridge.launchGame(package or self.package)
        _Zeit.sleep(5)

    def stop_game(self, package=None):
        # Ohne Root kann eine App nicht beendet werden: zurueck, dann neu oeffnen
        self.back()
        _Zeit.sleep(1)
        self.back()
        _Zeit.sleep(1)

    def keep_awake(self):
        pass  # erledigt die Stopp-Pille (FLAG_KEEP_SCREEN_ON)


def _make_bot(game, cfg, dev):
    if game == "car":
        import bot as B
        B.log = _log
        B.time = _Zeit
        c = dict(B.DEFAULTS)
        c.update({k: v for k, v in cfg.items() if k in c})
        c.update(debug=False, dry_run=False, max_levels=0)
        return B.Bot(dev, c)
    import yarn_bot as YB
    YB.time = _Zeit
    c = {"think_time": float(cfg.get("think_time", 1.0)), "dry_run": False, "max_levels": 0,
         "package": dev.package}
    return YB.YarnBot(dev, c, log=_log)


def run(config_json="{}"):
    try:
        cfg = json.loads(config_json or "{}")
    except ValueError:
        cfg = {}
    game = cfg.get("game", "yarn")
    if game not in GAMES:
        game = "yarn"
    dev = AndroidDevice(cfg.get("package") or GAMES[game])
    bot = _make_bot(game, cfg, dev)
    name = "Yarn Loop" if game == "yarn" else "Car Match"
    _log(f"Bot gestartet – {name} wird geoeffnet")

    errors = 0
    try:
        if not dev.game_running():
            dev.start_game()
        while Bridge.isRunning():
            try:
                bot.step()
                errors = 0
            except StopBot:
                break
            except Exception as e:  # noqa: BLE001
                errors += 1
                _log(f"Fehler: {e}")
                print(traceback.format_exc())
                if errors >= 25:
                    _log("Zu viele Fehler hintereinander – Bot stoppt")
                    break
                _Zeit.sleep(min(2 * errors, 10))
            Bridge.setLevels(int(getattr(bot, "levels_done", 0)))
    except StopBot:
        pass
    _log(f"Bot beendet – {getattr(bot, 'levels_done', 0)} Level geschafft")
