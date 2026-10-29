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
    "think_time": 1.0,
    "burst": 4,              # so viele Spulen hintereinander losschicken (Band-Kapazitaet beachten)
    "idle_timeout": 20.0,    # max. Wartezeit, bis das Band leer ist
    "default_count": 20,     # falls eine Zahl nicht lesbar ist
    "max_levels": 0,
    "dry_run": False,
    "package": GAME_PACKAGE,
}


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def status(msg):
    """Kurzer Zwischenstand (auf dem Handy in der Stopp-Pille), kein Protokoll."""


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
        self.nospool_since = None
        self.fail_count = 0
        self.last_taps = []
        self.futile = set()          # Ablage-Spulen, die nachweislich nichts nehmen
        self.last_rack_move = None
        self.last_col_move = None
        self.blocked_cols = set()     # Spalten, deren Antippen nichts bewirkt (z.B. verbunden)
        self.bad_pairs = set()        # vermeintliche Doppelspulen, die gar keine sind
        self.last_pair_move = None
        self.prev_rack_n = None
        self.prev_cells = None

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
        # War das Bild (fast) leer, gilt das Level als geschafft - danach kommt
        # meist gleich das Gewinn-Fenster, das Bild selbst sieht man nicht leer
        start = getattr(self, "start_cells", 0)
        if self.lattice is not None and start and self.level_cells <= max(3, 0.15 * start):
            self.levels_done += 1
            self.log(f"Level geschafft! ({self.levels_done})")
        self.start_cells = 0
        self.futile = set()
        self.last_rack_move = None
        self.last_col_move = None
        self.blocked_cols = set()
        self.bad_pairs = set()
        self.last_pair_move = None
        self.prev_rack_n = None
        self.prev_cells = None
        self.lattice = None
        self.palette = None
        self.level_cells = 0

    def _motion_region(self, img, belt):
        if belt is None:
            return img[: img.shape[0] // 2]
        return img[belt.y0:belt.y1, belt.x0:belt.x1]

    def wait_idle(self):
        """Wartet, bis Foerderband und Bild sich nicht mehr bewegen.
        Faehrt eine Spule endlos im Kreis (nimmt nichts mehr), reicht es, wenn
        sich das Maschenbild selbst ca. 1,5 s nicht mehr veraendert."""
        t_end = time.time() + self.cfg["idle_timeout"]
        prev = prev_pic = None
        calm = calm_pic = 0
        img = self.dev.screenshot()
        while True:
            belt = Y.find_belt(img)
            reg = self._motion_region(img, belt)
            small = cv2.resize(reg, (120, 140), interpolation=cv2.INTER_AREA).astype(np.int16)
            pic = None
            if belt is not None:
                X0, Y0, X1, Y1 = Y.picture_region(belt)
                pic = cv2.resize(img[Y0:Y1, X0:X1], (100, 120), interpolation=cv2.INTER_AREA).astype(np.int16)
            if prev is not None and prev.shape == small.shape:
                # Bewegung = einzelne Bildpunkte aendern sich stark (fahrende Spule,
                # Faden). Kompressionsrauschen bleibt weit darunter.
                moving = int((np.abs(small - prev).max(axis=2) > 50).sum())
                calm = calm + 1 if moving < 4 else 0
                if calm >= 2:
                    return img
            if pic is not None and prev_pic is not None and prev_pic.shape == pic.shape:
                moving_pic = int((np.abs(pic - prev_pic).max(axis=2) > 50).sum())
                calm_pic = calm_pic + 1 if moving_pic < 3 else 0
                if calm_pic >= 4:
                    return img
            prev, prev_pic = small, pic
            if belt is None:
                # Popup/Menue: dort blinkt und glitzert immer etwas (Knopf-Glanz,
                # Mini-Foerderband) - nicht auf Stillstand warten
                t_end = min(t_end, time.time() + 1.5)
            if time.time() > t_end:
                return img
            status(f"warte, bis das Band still steht ... {time.time() - t_end + self.cfg['idle_timeout']:.0f} s")
            time.sleep(0.35)
            img = self.dev.screenshot()

    # ------------------------------------------------------------------ Hauptschritt
    def step(self):
        t0 = time.time()
        status("warte, bis das Band still steht ...")
        img = self.wait_idle()
        t1 = time.time()
        status(f"erkenne Spielfeld ... (gewartet {t1 - t0:.1f} s)")
        scr = Y.analyze(img, self.lattice, self.palette)
        self._timing = f"warten {t1 - t0:.1f} s, erkennen {time.time() - t1:.1f} s"
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
        if scr.kind == "game" and not (any(scr.columns) or any(scr.rack)):
            # Rahmen ohne Spulen: z.B. Startseite mit Bild-Vorschau (sieht aus
            # wie ein Band). Nach kurzer Zeit den Level-Knopf unten antippen -
            # bei schweren Levels ist er rot/pink statt gruen.
            if self.nospool_since is None:
                self.nospool_since = time.time()
            elif time.time() - self.nospool_since > 6:
                H, W = img.shape[:2]
                self.log("Keine Spulen sichtbar (Startseite?) – tippe Level-Knopf")
                self.tap(W // 2, int(H * 0.84), "Level starten")
                self.nospool_since = None
                time.sleep(3)
                return
        else:
            self.nospool_since = None
        if scr.kind == "game":
            self.unknown_since = None
            self._tapped_continue = False
            self.game_step(img, scr)
        else:
            self.menu_step(img, scr)

    def game_step(self, img, scr):
        n_cells = S.weight(scr.grid) if scr.grid is not None else 0
        spools_visible = any(scr.columns) or any(scr.rack)

        # Levelwechsel erkennen: mehr Maschen als vorher -> neues Bild
        if self.lattice is not None and n_cells > self.level_cells + 3:
            self.log("Neues Level erkannt")
            self.reset_level()
            scr = Y.analyze(img)
            n_cells = S.weight(scr.grid) if scr.grid is not None else 0
        if self.lattice is None and scr.lattice is not None and n_cells > 0:
            self.lattice = scr.lattice
            self.palette = [c for _, c in scr.palette]
            pal = ", ".join(f"{n}" for n, _ in scr.palette)
            self.log(f"Level: {scr.lattice.nx}x{scr.lattice.ny} Maschen, Farben: {pal}")
            self.start_cells = n_cells
            self._dump_grid(scr)
        self.level_cells = n_cells

        if n_cells == 0:
            if self.lattice is not None:
                self.reset_level()
            time.sleep(1.5)
            return

        grid, columns, rack, taps = self.model(scr)
        # Schutz vor Endlosschleifen: Kommt eine Ablage-Spule unveraendert
        # zurueck (gleiche Zahl, Bild nicht kleiner), nimmt sie in Wirklichkeit
        # nichts - dann liegen lassen, bis sich das Bild veraendert hat.
        if self.prev_cells is not None:
            if n_cells < self.prev_cells - 2:
                self.futile.clear()
            elif self.last_rack_move is not None and n_cells >= self.prev_cells - 1:
                c, n = self.last_rack_move
                if any(r is not None and r[0] == c and r[1] == n for r in rack):
                    if (c, n) not in self.futile:
                        self.log(f"Ablage-Spule {self._cname(scr, c)} {n} nimmt nichts – bleibt liegen")
                    self.futile.add((c, n))
        rack_n = sum(1 for r in rack if r is not None)
        if self.prev_cells is not None:
            if n_cells < self.prev_cells - 2 or (self.prev_rack_n is not None and rack_n != self.prev_rack_n):
                self.blocked_cols.clear()
            elif self.last_col_move is not None and n_cells >= self.prev_cells - 1:
                i, c, n = self.last_col_move
                if i < len(columns) and columns[i] and columns[i][0][0] == c and columns[i][0][1] == n:
                    if i not in self.blocked_cols:
                        self.log(f"Spalte {i + 1} reagiert nicht (verbunden?) – lasse sie aus")
                    self.blocked_cols.add(i)
            if self.last_pair_move is not None and n_cells >= self.prev_cells - 1:
                i, j, c, n = self.last_pair_move
                if i < len(columns) and columns[i] and columns[i][0][0] == c and columns[i][0][1] == n:
                    if (i, j) not in self.bad_pairs:
                        self.log(f"Spalte {i + 1}+{j + 1} sind wohl nicht verbunden – einzeln probieren")
                    self.bad_pairs.add((i, j))
        if self.prev_cells is not None and n_cells < self.prev_cells - 2:
            self.bad_pairs.clear()
        for i, j in self.bad_pairs:
            for k in (i, j):
                if k < len(columns) and columns[k] and len(columns[k][0]) > 2 and columns[k][0][2] >= 0:
                    c0 = columns[k][0]
                    columns[k] = [(c0[0], c0[1], -1)] + list(columns[k][1:])
        for i in self.blocked_cols:
            if i < len(columns) and columns[i]:
                c0 = columns[i][0]
                # erfundene Verbindung ohne Partner -> fuer den Loeser nicht antippbar
                columns[i] = [(c0[0], c0[1], 900 + i)] + list(columns[i][1:])
        if self.futile:
            rack = [(S.UNKNOWN_COLOR, r[1], -1) if r is not None and (r[0], r[1]) in self.futile else r
                    for r in rack]
        self.prev_cells = n_cells
        self.prev_rack_n = rack_n
        self.last_rack_move = None
        self.last_col_move = None
        self.last_pair_move = None
        self.log("Sehe: " + self._describe(scr, grid, columns, rack)
                 + f" ({getattr(self, '_timing', '')})")
        t_plan = time.time()
        status("rechne ...")
        p = S.plan(grid, columns, rack, time_limit=self.cfg["think_time"])
        if p.moves and not p.complete and sum(1 for r in rack if r is None) <= 3:
            # Keine sichere Loesung gefunden -> deutlich laenger suchen
            status("rechne laenger ...")
            p2 = S.plan(grid, columns, rack, time_limit=self.cfg["think_time"] * 5)
            if p2.moves and (p2.complete or p2.score > p.score):
                p = p2
        if not p.moves:
            # Keine bekannte Spule -> Ueberraschungsspule vorne antippen
            for i, col in enumerate(scr.columns):
                if col:
                    self.tap(col[0].x, col[0].y, "Ueberraschungs-Spule")
                    time.sleep(1.2)
                    return
            # Nur noch Ablage-Spulen: der Reihe nach probieren statt ewig zu warten
            slots = [sp for sp in scr.rack if sp is not None]
            if slots:
                self.fallback_i = getattr(self, "fallback_i", -1) + 1
                sp = slots[self.fallback_i % len(slots)]
                self.futile = set()
                self.tap(sp.x, sp.y, "Ablage probieren")
                time.sleep(1.2)
                return
            self.log("Keine Spule antippbar – warte")
            time.sleep(1.5)
            return
        state = "Loesung gefunden" if p.complete else ("alles riskant" if p.dead_end else "beste Fortsetzung")
        self.log(f"Plan ({state}): {len(p.moves)} Zuege, {n_cells} Maschen uebrig, "
                 f"{time.time() - t_plan:.1f} s gerechnet")

        # Die ersten Zuege ausfuehren (Spalten-Zuege duerfen gebuendelt werden)
        sent = 0
        used_cols = set()
        rack_sent = False
        # Nur so viele Spulen gleichzeitig losschicken, wie Ablage-Plaetze frei
        # sind - jede kann im schlimmsten Fall mit Rest in der Ablage landen.
        free = sum(1 for sp in scr.rack if sp is None)
        limit = max(1, min(int(self.cfg["burst"]), free))
        if p.dead_end or (not p.complete and free <= 3):
            limit = 1          # knapp/riskant: nach jeder Spule neu nachdenken
        for mv in p.moves[: limit * 3]:
            if sent >= limit:
                break
            key = (mv.source, mv.index)
            if mv.source != "col":
                # Ablage-Plaetze ruecken nach, sobald eine Ablage-Spule losfaehrt ->
                # pro Runde hoechstens eine Ablage-Spule; Ablage-Zuege auf Spulen,
                # die erst noch in die Ablage kommen, ueberspringen (naechste Runde)
                # statt abzubrechen - die Spalten-Zuege danach laufen trotzdem.
                if rack_sent or key not in taps:
                    if sent == 0:
                        break
                    continue
            elif key not in taps:
                break
            if mv.source == "col" and (mv.index in used_cols or
                                       (mv.partner and mv.partner[1] in used_cols)):
                time.sleep(0.6)       # naechste Spule rutscht nach vorne
            x, y = taps[key]
            if sent == 0 and mv.source == "rack" and mv.partner is None:
                self.last_rack_move = (mv.color, mv.count)
            if sent == 0 and mv.source == "col" and mv.partner is None:
                self.last_col_move = (mv.index, mv.color, mv.count)
            if sent == 0 and mv.source == "col" and mv.partner is not None:
                self.last_pair_move = (mv.index, mv.partner[1], mv.color, mv.count)
            what = "Ablage" if mv.source == "rack" else f"Spalte {mv.index + 1}"
            if mv.partner is not None:
                what += f" + Partner ({self._cname(scr, mv.partner[2])} {mv.partner[3]})"
            self.tap(x, y, f"{what}: Farbe {self._cname(scr, mv.color)} ({mv.count})")
            if mv.source == "col":
                used_cols.add(mv.index)
                if mv.partner is not None:
                    used_cols.add(mv.partner[1])
            if mv.source == "rack":
                rack_sent = True
            sent += 1
            time.sleep(0.35)
        time.sleep(0.3)

    def _dump_grid(self, scr):
        """Gitter zeilenweise ins Protokoll (fuer die Fehlersuche).
        Grossbuchstabe = Farbe (A = 1. Farbe), klein = Knoten, . = leer."""
        try:
            g = scr.grid
            legend = " ".join(f"{chr(65 + i)}={n}" for i, (n, _) in enumerate(scr.palette))
            lines = []
            for r in range(g.shape[0]):
                row = ""
                for v in g[r]:
                    v = int(v)
                    if v < 0:
                        row += "."
                    else:
                        ch = chr(65 + v % S.KNOT)
                        row += ch.lower() if v >= S.KNOT or (scr.info.get("knots") and False) else ch
                lines.append(f"G{r:02d} {row}")
            # Neueste Zeile steht oben -> rueckwaerts eintragen
            for ln in reversed(lines):
                self.log(ln)
            self.log(f"Gitter {g.shape[1]}x{g.shape[0]}: {legend} Knoten {scr.info.get('knots')}")
        except Exception as e:
            self.log(f"Gitter-Ausgabe fehlgeschlagen: {e}")

    def _describe(self, scr, grid, columns, rack):
        """Kurzbeschreibung fuer das Protokoll: was der Bot erkannt hat."""
        def sp(t):
            if t is None:
                return "-"
            c, n = t[0], t[1]
            name = "?" if c < 0 else self._cname(scr, c)
            link = "=" if len(t) > 2 and t[2] is not None and t[2] >= 0 else ""
            return f"{link}{name}{n}"
        counts = S.color_counts(grid)
        knots = scr.info.get("knots") if scr.info else None
        bild = " ".join(f"{self._cname(scr, k)}:{n}" for k, n in sorted(counts.items()))
        spalten = " | ".join(" ".join(sp(t) for t in col) for col in columns)
        ablage = " ".join(sp(t) for t in rack)
        kn = f" · Knoten {[k[0] for k in knots]}" if knots else ""
        return f"Bild {bild}{kn} · Spalten {spalten} · Ablage {ablage}"

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
            # Tutorial mit gruenem Pfeil (z.B. Gratis-Booster): genau dorthin tippen
            arrow = Y.find_tutorial_arrow(img)
            if arrow is not None:
                self.log("Tutorial-Pfeil – tippe dort")
                self.tap(arrow[0], arrow[1], "Tutorial")
                self.unknown_since = None
                time.sleep(1.5)
                return
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
        if 5 < waited and not getattr(self, "_tapped_continue", False):
            # z.B. "Wochen-Serie ... Tippen zum Fortfahren": irgendwo antippen.
            # Unten in der Mitte liegt sonst hoechstens der Level-Start-Knopf.
            self.log("Unbekannter Bildschirm – tippe zum Fortfahren")
            self.tap(W // 2, int(H * 0.79), "Fortfahren")
            self._tapped_continue = True
            time.sleep(2)
            return
        if waited > 25:
            self.log("Unbekannter Bildschirm (Werbung?) – Zuruecktaste")
            self.dev.back()
            self.unknown_since = time.time() - 10
            self._tapped_continue = False
            time.sleep(2)
            return
        time.sleep(1.0)
