"""
Ablauf des Bots fuer Yarn Loop. Funktioniert mit jedem "Geraet", das
screenshot(), tap(x, y), back(), foreground() und start_game() kann
(ADB am PC oder die Android-App).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

import yarn_solver as S
import yarn_vision as Y

GAME_PACKAGE = "com.combo.yarnflow"

DEFAULTS = {
    "think_time": 2.0,
    "burst": 2,              # so viele Spulen hintereinander losschicken (Band-Kapazitaet beachten)
    "idle_timeout": 20.0,    # max. Wartezeit, bis das Band leer ist
    "default_count": 20,     # falls eine Zahl nicht lesbar ist
    "max_levels": 0,
    "dry_run": False,
    "package": GAME_PACKAGE,
}


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class YarnBot:
    def __init__(self, dev, cfg=None, log=None):
        self.dev = dev
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self.log = log or _log
        self.lattice = None
        self.palette = None
        self.levels_done = 0
        self.level_cells = 0
        self.unknown_since = None
        self.fail_count = 0
        self.last_taps = []

    # ------------------------------------------------------------------ Hilfen
    def tap(self, x, y, why=""):
        self.log(f"  tippe ({x},{y}) {why}")
        if not self.cfg["dry_run"]:
            self.dev.tap(int(x), int(y))

    def run(self, should_continue=lambda: True):
        """Endlosschleife (PC). Auf Android steuert android_main.py die Schleife."""
        self.log("Yarn-Loop-Bot laeuft. Abbrechen mit Strg+C.")
        try:
            fg = self.dev.foreground() or ""
            if self.cfg["package"] not in fg:
                self.log("Starte Yarn Loop ...")
                self.dev.start_game(self.cfg["package"])
        except Exception:
            pass
        while should_continue():
            if self.cfg["max_levels"] and self.levels_done >= self.cfg["max_levels"]:
                self.log(f"{self.levels_done} Level geschafft – fertig.")
                return
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                self.log(f"Fehler: {e}")
                time.sleep(2)

    def reset_level(self):
        self.lattice = None
        self.palette = None
        self.level_cells = 0

    def _motion_region(self, img, belt):
        if belt is None:
            return img[: img.shape[0] // 2]
        return img[belt.y0:belt.y1, belt.x0:belt.x1]

    def wait_idle(self):
        """Wartet, bis Foerderband und Bild sich nicht mehr bewegen."""
        t_end = time.time() + self.cfg["idle_timeout"]
        prev = None
        calm = 0
        img = self.dev.screenshot()
        while True:
            belt = Y.find_belt(img)
            reg = self._motion_region(img, belt)
            small = cv2.resize(reg, (120, 140), interpolation=cv2.INTER_AREA).astype(np.int16)
            if prev is not None and prev.shape == small.shape:
                # Bewegung = einzelne Bildpunkte aendern sich stark (fahrende Spule,
                # Faden). Kompressionsrauschen bleibt weit darunter.
                moving = int((np.abs(small - prev).max(axis=2) > 50).sum())
                calm = calm + 1 if moving < 4 else 0
                if calm >= 2:
                    return img
            prev = small
            if belt is None:
                # Popup/Menue: dort blinkt und glitzert immer etwas (Knopf-Glanz,
                # Mini-Foerderband) - nicht auf Stillstand warten
                t_end = min(t_end, time.time() + 1.5)
            if time.time() > t_end:
                return img
            time.sleep(0.35)
            img = self.dev.screenshot()

    # ------------------------------------------------------------------ Hauptschritt
    def step(self):
        img = self.wait_idle()
        scr = Y.analyze(img, self.lattice, self.palette)
        # Spulen rutschen evtl. noch nach -> Zahl unlesbar: kurz warten, neu lesen
        for _ in range(3):
            if scr.kind != "game":
                break
            unread = [c[0] for c in scr.columns if c and c[0].color != Y.UNKNOWN and c[0].count is None]
            unread += [sp for sp in scr.rack if sp is not None and sp.color != Y.UNKNOWN and sp.count is None]
            if not unread:
                break
            time.sleep(0.5)
            img = self.dev.screenshot()
            scr = Y.analyze(img, self.lattice, self.palette)
        if scr.kind == "game":
            self.unknown_since = None
            self.game_step(img, scr)
        else:
            self.menu_step(img, scr)

    def game_step(self, img, scr):
        n_cells = int((scr.grid >= 0).sum()) if scr.grid is not None else 0
        spools_visible = any(scr.columns) or any(scr.rack)

        # Levelwechsel erkennen: mehr Maschen als vorher -> neues Bild
        if self.lattice is not None and n_cells > self.level_cells + 3:
            self.log("Neues Level erkannt")
            self.reset_level()
            scr = Y.analyze(img)
            n_cells = int((scr.grid >= 0).sum()) if scr.grid is not None else 0
        if self.lattice is None and scr.lattice is not None and n_cells > 0:
            self.lattice = scr.lattice
            self.palette = [c for _, c in scr.palette]
            pal = ", ".join(f"{n}" for n, _ in scr.palette)
            self.log(f"Level: {scr.lattice.nx}x{scr.lattice.ny} Maschen, Farben: {pal}")
        self.level_cells = n_cells

        if n_cells == 0:
            if self.lattice is not None:
                self.levels_done += 1
                self.log(f"Level geschafft! ({self.levels_done})")
                self.reset_level()
            time.sleep(1.5)
            return

        grid, columns, rack, taps = self.model(scr)
        p = S.plan(grid, columns, rack, time_limit=self.cfg["think_time"])
        if not p.moves:
            # Keine bekannte Spule -> Ueberraschungsspule vorne antippen
            for i, col in enumerate(scr.columns):
                if col:
                    self.tap(col[0].x, col[0].y, "Ueberraschungs-Spule")
                    time.sleep(1.2)
                    return
            self.log("Keine Spule antippbar – warte")
            time.sleep(1.5)
            return
        state = "Loesung gefunden" if p.complete else ("alles riskant" if p.dead_end else "beste Fortsetzung")
        self.log(f"Plan ({state}): {len(p.moves)} Zuege, {n_cells} Maschen uebrig")

        # Die ersten Zuege ausfuehren (Spalten-Zuege duerfen gebuendelt werden)
        sent = 0
        used_cols = set()
        for mv in p.moves[: max(1, int(self.cfg["burst"]))]:
            if sent > 0 and mv.source != "col":
                break
            key = (mv.source, mv.index)
            if key not in taps:
                break
            if mv.source == "col" and (mv.index in used_cols or
                                       (mv.partner and mv.partner[1] in used_cols)):
                time.sleep(0.6)       # naechste Spule rutscht nach vorne
            x, y = taps[key]
            what = "Ablage" if mv.source == "rack" else f"Spalte {mv.index + 1}"
            if mv.partner is not None:
                what += f" + Partner ({self._cname(scr, mv.partner[2])} {mv.partner[3]})"
            self.tap(x, y, f"{what}: Farbe {self._cname(scr, mv.color)} ({mv.count})")
            if mv.source == "col":
                used_cols.add(mv.index)
                if mv.partner is not None:
                    used_cols.add(mv.partner[1])
            sent += 1
            time.sleep(0.45)
        time.sleep(1.0)

    def _cname(self, scr, idx):
        try:
            return scr.palette[idx][0]
        except Exception:
            return str(idx)

    def model(self, scr):
        """Bildschirm -> Loeser-Eingabe."""
        grid = scr.grid.copy()
        default = int(self.cfg["default_count"])

        def conv(sp, front):
            if sp is None:
                return None
            if sp.color is None or sp.color == Y.UNKNOWN or not str(sp.color).isdigit():
                color = S.UNKNOWN_COLOR
            else:
                color = int(sp.color)
            count = sp.count if sp.count is not None else default
            if not front and sp.count is None:
                count = default
            link = getattr(sp, "link", None)
            return (color, count, -1 if link is None else int(link))

        columns, taps = [], {}
        for i, col in enumerate(scr.columns):
            seq = []
            for k, sp in enumerate(col):
                c = conv(sp, k == 0)
                if c is not None:
                    seq.append(c)
            columns.append(seq)
            if col:
                taps[("col", i)] = (col[0].x, col[0].y)
        rack = []
        for j, sp in enumerate(scr.rack):
            rack.append(conv(sp, True))
            if sp is not None:
                taps[("rack", j)] = (sp.x, sp.y)
        return grid, columns, rack, taps

    # ------------------------------------------------------------------ Menues
    def menu_step(self, img, scr):
        """Popups. Sicherheitsregeln:
        - Zwei oder mehr gruene Knoepfe (z.B. "Spiel weiter 900" + "4,99 EUR")
          -> nie gruen tippen, sondern rotes X bzw. Zuruecktaste.
        - Gruener Knopf mit Muenzsymbol -> nie tippen.
        - Genau ein gruener Knopf ohne Muenze ("Versuche es noch einmal",
          "Level 6", "Weiter") -> tippen."""
        import screens
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        greens = [b for b in scr.buttons if b[4] == "green"]
        closes = sorted([b for b in scr.buttons if b[4] == "close"], key=lambda b: b[1])
        paid = [g for g in greens if screens.has_coin_icon(img, hsv, g[0], g[1], g[2], g[3])]
        free = [g for g in greens if g not in paid]
        if len(greens) >= 2 or paid:
            # nur ein X oberhalb aller gruenen Knoepfe ist das Schliessen-X des Fensters
            top_green = min(g[1] for g in greens)
            closes = [c for c in closes if c[1] < top_green - 0.05 * H]
            if closes:
                x, y, w, h, _ = closes[0]
                self.log("Angebot (Muenzen/Geld) – lehne ab")
                self.tap(x, y, "X")
            else:
                self.log("Angebot ohne X – Zuruecktaste")
                if not self.cfg["dry_run"]:
                    self.dev.back()
            time.sleep(2)
            return
        if free:
            x, y, w, h, _ = free[0]
            if scr.belt is None and y > H * 0.7:
                self.log("Startseite – Level starten")
            else:
                self.log("Gruener Knopf (z.B. Versuche es noch einmal / Weiter)")
                if scr.belt is not None:
                    self.fail_count += 1
            self.reset_level()
            self.tap(x, y)
            time.sleep(3)
            return
        if scr.kind == "dim":
            # kurz abwarten (z.B. Banner "Kein Platz mehr!"), dann oben antippen
            if self.unknown_since is None:
                self.unknown_since = time.time()
                time.sleep(1.5)
                return
            if time.time() - self.unknown_since > 3:
                self.tap(W // 2, int(H * 0.06), "Hinweis wegtippen")
                self.unknown_since = None
                time.sleep(1.5)
            else:
                time.sleep(1.0)
            return
        if closes:
            x, y, w, h, _ = closes[0]
            self.tap(x, y, "Fenster schliessen")
            time.sleep(1.5)
            return
        # unbekannt: Spiel im Vordergrund?
        if self.unknown_since is None:
            self.unknown_since = time.time()
        fg = ""
        try:
            fg = self.dev.foreground() or ""
        except Exception:
            pass
        waited = time.time() - self.unknown_since
        if fg and self.cfg["package"] not in fg and waited > 3:
            self.log(f"Yarn Loop nicht im Vordergrund ({fg}) – starte es")
            self.dev.start_game(self.cfg["package"])
            self.unknown_since = None
            return
        if waited > 25:
            self.log("Unbekannter Bildschirm (Werbung?) – Zuruecktaste")
            self.dev.back()
            self.unknown_since = time.time() - 10
            time.sleep(2)
            return
        time.sleep(1.0)
