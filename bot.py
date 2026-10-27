#!/usr/bin/env python3
"""
Car Match Bot - spielt "Car Match - Traffic Puzzle" (Grand Games) automatisch
auf einem per ADB verbundenen Android-Handy (z.B. Xiaomi).

Start:   python bot.py
Hilfe:   python bot.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque

import cv2

import screens
import solver as S
import vision as V
from device import Device

BOOSTER_INDEX = {"undo": 0, "shuffle": 1, "super_undo": 2, "magnet": 3}

DEFAULTS = {
    "serial": None,              # adb-Seriennummer (bei mehreren Geraeten)
    "max_levels": 0,             # 0 = endlos
    "think_time": 2.5,           # Sekunden Rechenzeit pro Planung
    "after_tap": 0.45,           # Wartezeit nach einem Tap
    "stable_wait": 0.25,         # Abstand zweier Screenshots fuer Ruhe-Check
    "use_boosters": True,        # Booster nutzen, wenn sonst verloren
    "booster_order": ["super_undo", "magnet", "shuffle"],
    "allow_coins": False,        # Muenzen ausgeben (z.B. "Weiterspielen") - aus!
    "debug": False,              # Screenshots + Erkennung in ./debug speichern
    "dry_run": False,            # nur erkennen/planen, nicht tippen
    "restart_after_unknown": 90, # Sek. unbekannter Bildschirm -> Back / Neustart
    "analysis_width": 720,       # Screenshots fuer die Analyse auf diese Breite
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Bot:
    def __init__(self, dev: Device, cfg: dict):
        self.dev = dev
        self.cfg = cfg
        self.grid = None
        self.top = None
        self.plan_moves = deque()
        self.plan_complete = False
        self.levels_done = 0
        self.no_move_count = 0
        self.unknown_since = None
        self.last_popup = deque(maxlen=8)
        self.booster_blocked = set()
        self.game_frames = 0
        self.empty_frames = 0
        self.dbg_n = 0
        self.scale = 1.0
        self.offx = 0
        self.last_car_count = 0
        self.same_move = (None, 0)
        self.level_counted = False
        if cfg["debug"]:
            os.makedirs("debug", exist_ok=True)

    # ------------------------------------------------------------------ util
    def tap(self, x, y, why=""):
        """x, y in Analyse-Pixeln."""
        X, Y = int(x * self.scale) + self.offx, int(y * self.scale)
        log(f"  tippe ({X},{Y}) {why}")
        if not self.cfg["dry_run"]:
            self.dev.tap(X, Y)

    def save_debug(self, img, scr, tag=""):
        if not self.cfg["debug"]:
            return
        self.dbg_n = (self.dbg_n + 1) % 300
        out = V.draw_debug(img, scr)
        h, w = out.shape[:2]
        out = cv2.resize(out, (540, int(540 * h / w)))
        cv2.imwrite(f"debug/{self.dbg_n:03d}_{scr.kind}{tag}.jpg", out)

    def reset_level(self):
        self.grid = None
        self.top = None
        self.plan_moves.clear()
        self.plan_complete = False
        self.booster_blocked.clear()
        self.no_move_count = 0

    @staticmethod
    def signature(scr):
        if scr.kind != "game" or scr.board is None:
            return (scr.kind,)
        b = scr.board
        cells = tuple((c.kind, c.color, c.eyes) for row in b.cells for c in row)
        return (cells, tuple(scr.holder.colors))

    def capture(self):
        """Screenshot holen und (verkleinert, schneller) analysieren.
        Alle Koordinaten im Screen-Objekt sind danach Analyse-Pixel;
        self.scale rechnet sie in echte Bildschirmpixel um."""
        full = self.dev.screenshot()
        # Emulator im Querformat: Spiel steht hochkant in der Mitte mit
        # schwarzen Raendern -> nur den Spielbereich auswerten
        self.offx = 0
        if full.shape[1] > full.shape[0]:
            colmax = full.max(axis=2).max(axis=0)
            xs = (colmax > 20).nonzero()[0]
            if len(xs) and (xs[-1] - xs[0]) < full.shape[1] * 0.8:
                self.offx = int(xs[0])
                full = full[:, xs[0]:xs[-1] + 1]
        W = full.shape[1]
        aw = self.cfg.get("analysis_width", 720)
        if W > aw:
            self.scale = W / aw
            img = cv2.resize(full, (aw, int(full.shape[0] / self.scale)), interpolation=cv2.INTER_AREA)
        else:
            self.scale = 1.0
            img = full
        scr = V.analyze(img, grid=self.grid, top=self.top)
        return img, scr

    def capture_stable(self, tries=8):
        """Wartet, bis sich nichts mehr bewegt (fahrende Autos, Animationen)."""
        img, scr = self.capture()
        for _ in range(tries):
            time.sleep(self.cfg["stable_wait"])
            img2, scr2 = self.capture()
            if self.signature(scr2) == self.signature(scr):
                return img2, scr2
            img, scr = img2, scr2
        return img, scr

    # ------------------------------------------------------------------ main
    def run(self):
        log("Bot laeuft. Abbrechen mit Strg+C.")
        self.dev.keep_awake()
        while True:
            if self.cfg["max_levels"] and self.levels_done >= self.cfg["max_levels"]:
                log(f"{self.levels_done} Level geschafft - fertig.")
                return
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"Fehler: {e}")
                traceback.print_exc()
                time.sleep(2)

    def step(self):
        img, scr = self.capture_stable()
        if scr.kind == "game":
            self.unknown_since = None
            self.save_debug(img, scr)
            self.game_step(img, scr)
            return
        # Nicht im Spielfeld
        if self.game_frames > 0:
            self.game_frames = 0
            self.reset_level()
        self.save_debug(img, scr)
        if scr.kind in ("unknown", "dim") and not self.cfg["dry_run"]:
            fg = self.dev.foreground()
            if fg and "carmatch" not in fg:
                if self._looks_like_ad(fg):
                    self.handle_ad(img, scr)
                    return
                log(f"Spiel nicht im Vordergrund ({fg}) - starte es.")
                self.dev.start_game()
                return
        self.menu_step(img, scr)

    # ------------------------------------------------------------------ Spiel
    def game_step(self, img, scr):
        self.game_frames += 1
        b, hold = scr.board, scr.holder
        if b is None:
            log("Spielfeld nicht erkannt - warte")
            time.sleep(0.5)
            return
        cars = [c for row in b.cells for c in row if c.kind == V.CAR]
        eyes = {(c.r, c.c) for c in cars if c.eyes}
        filled = hold.filled

        # Neues Level ohne Zwischenbildschirm (Level 1-5): Autozahl springt hoch
        if self.grid is not None and len(cars) > self.last_car_count + 2 and not filled:
            log("Mehr Autos als vorher - neues Level, Raster neu bestimmen")
            self.reset_level()
            img, scr = self.capture()
            b, hold = scr.board, scr.holder
            if b is None:
                return
            cars = [c for row in b.cells for c in row if c.kind == V.CAR]
            eyes = {(c.r, c.c) for c in cars if c.eyes}
            filled = hold.filled
        self.last_car_count = len(cars)

        # Raster fuer dieses Level einrasten (stabiler als jedes Mal neu messen)
        if self.grid is None and len(cars) >= 3:
            self.grid = scr.info.get("grid")
            self.top = scr.info.get("top")
            log(f"Neues Level erkannt: {b.rows}x{b.cols} Raster, {len(cars)} Autos, "
                f"Zelle {b.px:.0f}x{b.py:.0f}px")

        if not cars:
            self.empty_frames += 1
            if not filled and self.empty_frames >= 2 and not self.level_counted:
                self.levels_done += 1
                log(f"Level geschafft! ({self.levels_done} in dieser Sitzung)")
                self.reset_level()
                self.level_counted = True
            time.sleep(0.6)
            return
        self.empty_frames = 0
        self.level_counted = False

        if len(filled) >= S.HOLDER_SIZE:
            log("Parkleiste voll - warte auf Popup")
            time.sleep(1.0)
            return

        if not eyes:
            self.no_move_count += 1
            if self.no_move_count < 4:
                time.sleep(0.4)
                return
            log("Keine Augen erkannt - nutze Modell")
        else:
            self.no_move_count = 0

        move = self.next_move(b, filled, eyes)
        if move is None:
            if self.cfg["use_boosters"] and self.use_booster(scr, filled):
                return
            # notfalls irgendein Auto mit Augen
            if eyes:
                move = sorted(eyes)[0]
            else:
                time.sleep(0.5)
                return
        r, c = move
        key = (r, c, tuple(filled), len(cars))
        if self.same_move[0] == key:
            self.same_move = (key, self.same_move[1] + 1)
            if self.same_move[1] >= 2:
                log("Gleicher Zug ohne Wirkung - Raster neu bestimmen")
                self.reset_level()
                self.same_move = (None, 0)
                return
        else:
            self.same_move = (key, 0)
        cell = b.cells[r][c]
        x, y = b.tap_point(r, c)
        self.tap(x, y, f"Auto ({r},{c}) {cell.color}  Leiste={filled}")
        time.sleep(self.cfg["after_tap"])

    def next_move(self, b, filled, eyes):
        # Vorhandenen kompletten Plan weiterverfolgen, solange er passt
        while self.plan_moves:
            r, c = self.plan_moves[0]
            ok = (0 <= r < b.rows and 0 <= c < b.cols and b.cells[r][c].kind == V.CAR
                  and (not eyes or (r, c) in eyes))
            if ok and self.plan_complete:
                self.plan_moves.popleft()
                return (r, c)
            break
        self.plan_moves.clear()
        t0 = time.time()
        p = S.plan(b, filled, eyes_hint=eyes, time_limit=self.cfg["think_time"])
        dt = time.time() - t0
        if not p.moves:
            log(f"Kein sicherer Zug gefunden ({dt:.1f}s)")
            return None
        if p.dead_end:
            log("Alle Zuege fuehren zum Verlust")
            return None
        state = "komplette Loesung" if p.complete else f"beste Teilloesung ({len(p.moves)} Zuege)"
        log(f"Plan: {state} in {dt:.1f}s")
        self.plan_complete = p.complete
        self.plan_moves = deque(p.moves[1:])
        return p.moves[0]

    def use_booster(self, scr, filled):
        if len(scr.boosters) < 4:
            return False
        for name in self.cfg["booster_order"]:
            if name in self.booster_blocked:
                continue
            if name == "super_undo" and len(filled) < 3:
                continue
            x, y = scr.boosters[BOOSTER_INDEX[name]]
            self.tap(x, y, f"Booster {name}")
            self.booster_blocked.add(name)   # pro Level hoechstens einmal
            time.sleep(1.5)
            img, s2 = self.capture()
            if s2.kind != "game":
                # Kauf-Dialog o.ae. -> schliessen
                for (bx, by, bw, bh, lab) in s2.buttons:
                    if lab == "close":
                        self.tap(bx, by, "Kaufdialog schliessen")
                        break
                else:
                    self.dev.back()
                time.sleep(1)
                continue
            self.plan_moves.clear()
            return True
        return False

    # ------------------------------------------------------------------ Menues
    def menu_step(self, img, scr):
        if scr.kind == "unknown" or scr.kind == "dim":
            if self.unknown_since is None:
                self.unknown_since = time.time()
        d = screens.decide(img, scr, allow_coins=self.cfg["allow_coins"])
        if d is None:
            waited = time.time() - (self.unknown_since or time.time())
            if scr.kind == "dim" and waited > 3:
                x, y = self.tutorial_target(img)
                self.tap(x, y, "Abgedunkelt (Tutorial?) - tippe")
                time.sleep(1.5)
            elif waited > self.cfg["restart_after_unknown"]:
                log("Lange unbekannter Bildschirm - Spiel neu starten")
                if not self.cfg["dry_run"]:
                    self.dev.stop_game()
                    self.dev.start_game()
                self.unknown_since = None
            elif waited > 20:
                log("Unbekannter Bildschirm - Zuruecktaste")
                if not self.cfg["dry_run"]:
                    self.dev.back()
                time.sleep(2)
            else:
                time.sleep(1)
            return
        self.unknown_since = None
        if d[0] == "tap":
            _, x, y, why = d
            key = (x // 40, y // 40)
            self.last_popup.append(key)
            if self.last_popup.count(key) >= 6:
                log("Gleicher Button immer wieder - vermutlich keine Leben. Warte 5 Min.")
                self.last_popup.clear()
                time.sleep(300)
                return
            if "Level" in why or "level" in why:
                self.reset_level()
            self.tap(x, y, f"[{scr.kind}] {why}")
            time.sleep(1.2)
        elif d[0] == "wait":
            _, secs, why = d
            log(why)
            time.sleep(secs)
        elif d[0] == "back":
            self.dev.back()
            time.sleep(1)

    @staticmethod
    def tutorial_target(img):
        """Im Tutorial ist nur das zu tippende Auto hell - genau ein autogrosser
        bunter Fleck. Sonst einfach in die Mitte tippen."""
        import numpy as np
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = V.car_mask(hsv).astype(np.uint8)
        m[: int(H * 0.2)] = 0
        m[int(H * 0.88):] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        cands = [cen[i] for i in range(1, n)
                 if W * 0.07 < st[i, 2] < W * 0.16 and W * 0.07 < st[i, 3] < W * 0.18]
        if len(cands) == 1:
            return int(cands[0][0]), int(cands[0][1] + W * 0.02)
        return W // 2, int(H * 0.45)

    # ------------------------------------------------------------------ Werbung
    AD_HINTS = ("ads", "adactivity", "applovin", "unity3d", "ironsource", "mbridge",
                "vungle", "fyber", "bytedance", "pangle", "inmobi", "chartboost",
                "mintegral", "facebook.ads", "adcolony", "admob", "moloco", "liftoff")

    def _looks_like_ad(self, fg):
        f = fg.lower()
        return any(h in f for h in self.AD_HINTS)

    def handle_ad(self, img, scr):
        log("Werbung erkannt - warte auf Schliessen-Knopf")
        H, W = img.shape[:2]
        start = time.time()
        while time.time() - start < 75:
            time.sleep(3)
            img = self.dev.screenshot()
            if "carmatch" in self.dev.foreground() and not self._looks_like_ad(self.dev.foreground()):
                return
            for (t, x, y, w, h) in screens.ocr_words(img):
                if t.strip() in ("x", "×", "close", "skip", "skip ad", "skip video", ">>", "»"):
                    log(f"  Werbung schliessen '{t}'")
                    self.dev.tap(x + w // 2, y + h // 2)   # volle Aufloesung
                    time.sleep(2)
                    break
            if time.time() - start > 35:
                self.dev.back()
        self.dev.back()


def analyze_file(path):
    img = cv2.imread(path)
    if img is None:
        print("Kann Bild nicht lesen:", path)
        return
    scr = V.analyze(img)
    print("Bildschirm:", scr.kind)
    if scr.holder:
        print("Parkleiste:", scr.holder.colors)
    if scr.board:
        b = scr.board
        print(f"Raster {b.rows}x{b.cols}, Zelle {b.px:.1f}x{b.py:.1f}px")
        for row in b.cells:
            line = []
            for c in row:
                if c.kind == V.CAR:
                    s = c.color[:2].upper() if c.eyes else c.color[:2]
                    line.append(f"[{s}]" if c.eyes else f" {s} ")
                else:
                    line.append({V.ROAD: "  . ", V.WALL: " ## ", V.UNKNOWN: "  ? "}[c.kind])
            print("  " + "".join(line))
        eyes = {(c.r, c.c) for row in b.cells for c in row if c.eyes}
        p = S.plan(b, scr.holder.filled, eyes_hint=eyes)
        print("Plan:", "komplett" if p.complete else "teilweise", p.moves[:12])
    out = os.path.splitext(path)[0] + "_erkannt.jpg"
    cv2.imwrite(out, V.draw_debug(img, scr))
    print("Debug-Bild:", out)


def main():
    ap = argparse.ArgumentParser(description="Car Match Bot (ADB)")
    ap.add_argument("--serial", help="ADB-Seriennummer / IP:Port")
    ap.add_argument("--levels", type=int, help="Anzahl Level, dann stoppen (0=endlos)")
    ap.add_argument("--debug", action="store_true", help="Debug-Bilder speichern")
    ap.add_argument("--dry-run", action="store_true", help="Nicht tippen, nur anzeigen")
    ap.add_argument("--no-boosters", action="store_true", help="Keine Booster benutzen")
    ap.add_argument("--allow-coins", action="store_true", help="Muenzen ausgeben erlauben")
    ap.add_argument("--think", type=float, help="Rechenzeit pro Zug (s)")
    ap.add_argument("--analyze", metavar="BILD", help="Nur ein Screenshot analysieren")
    ap.add_argument("--screenshot", metavar="DATEI", help="Screenshot vom Handy speichern")
    ap.add_argument("--config", default="config.json")
    a = ap.parse_args()

    if a.analyze:
        analyze_file(a.analyze)
        return

    cfg = dict(DEFAULTS)
    if os.path.exists(a.config):
        with open(a.config, encoding="utf-8") as f:
            cfg.update(json.load(f))
    if a.serial: cfg["serial"] = a.serial
    if a.levels is not None: cfg["max_levels"] = a.levels
    if a.debug: cfg["debug"] = True
    if a.dry_run: cfg["dry_run"] = True
    if a.no_boosters: cfg["use_boosters"] = False
    if a.allow_coins: cfg["allow_coins"] = True
    if a.think: cfg["think_time"] = a.think

    dev = Device(cfg["serial"])
    serial = dev.connect()
    log(f"Verbunden mit {serial}, Bildschirm {dev.size[0]}x{dev.size[1]}")
    if not screens.HAVE_OCR:
        log("Hinweis: Tesseract-OCR nicht gefunden - Popups werden nur per Farbe erkannt.")
    if a.screenshot:
        cv2.imwrite(a.screenshot, dev.screenshot())
        log(f"Screenshot gespeichert: {a.screenshot}")
        return
    if not dev.game_running():
        log("Starte Car Match ...")
        dev.start_game()
    try:
        Bot(dev, cfg).run()
    except KeyboardInterrupt:
        log("Beendet.")


if __name__ == "__main__":
    main()
