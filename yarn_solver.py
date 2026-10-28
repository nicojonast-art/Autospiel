"""
Loeser fuer Yarn Loop.

Regeln (aus dem Video Bild fuer Bild nachgeprueft):
  * Eine Spule faehrt genau EINE Runde um das Bild:
      unten   von links nach rechts  (schaut nach oben in jede Spalte)
      rechts  von unten nach oben    (schaut nach links in jede Zeile)
      oben    von rechts nach links  (schaut nach unten in jede Spalte)
      links   von oben nach unten    (schaut nach rechts in jede Zeile)
  * An jeder Spalte/Zeile nimmt sie hoechstens EINE Masche: die vorderste
    (von ihrer Seite aus gesehen) - aber nur, wenn die Farbe passt.
  * Jede Masche kostet 1 von der Zahl auf der Spule. Bei 0 ist sie weg.
  * Bleibt etwas uebrig, kommt die Spule in die Ablage (5 Plaetze).
    Ist die Ablage voll -> "Kein Platz mehr!" -> verloren.
  * Antippbar: vorderste Spule jeder Spalte und jede Spule in der Ablage.

Gesucht wird eine Reihenfolge, in der die Ablage nie ueberlaeuft und das
Bild leer wird. Nach jeder Runde liest der Bot neu und plant neu.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

RACK_SIZE = 5
UNKNOWN_COLOR = -2


def lanes(ny, nx):
    """Reihenfolge der Bahnen einer Runde: (art, index)."""
    out = [("up", c) for c in range(nx)]                 # unten, links -> rechts
    out += [("left", r) for r in range(ny - 1, -1, -1)]  # rechts, unten -> oben
    out += [("down", c) for c in range(nx - 1, -1, -1)]  # oben, rechts -> links
    out += [("right", r) for r in range(ny)]             # links, oben -> unten
    return out


def front_cell(grid, kind, i):
    """Vorderste Masche einer Bahn (r, c) oder None."""
    ny, nx = grid.shape
    if kind == "up":
        col = grid[:, i]
        nz = np.nonzero(col >= 0)[0]
        return (int(nz[-1]), i) if len(nz) else None
    if kind == "down":
        col = grid[:, i]
        nz = np.nonzero(col >= 0)[0]
        return (int(nz[0]), i) if len(nz) else None
    if kind == "left":
        row = grid[i, :]
        nz = np.nonzero(row >= 0)[0]
        return (i, int(nz[-1])) if len(nz) else None
    row = grid[i, :]
    nz = np.nonzero(row >= 0)[0]
    return (i, int(nz[0])) if len(nz) else None


def run_lap(grid, color, count, lane_list=None):
    """Spielt eine Runde. Veraendert grid! Rueckgabe: (genommen, rest)."""
    if lane_list is None:
        lane_list = lanes(*grid.shape)
    taken = 0
    if count is None:
        count = 10 ** 6
    for kind, i in lane_list:
        if count <= 0:
            break
        cell = front_cell(grid, kind, i)
        if cell is None:
            continue
        r, c = cell
        if grid[r, c] == color:
            grid[r, c] = -1
            count -= 1
            taken += 1
    return taken, count


# ---------------------------------------------------------------------------
@dataclass
class Move:
    source: str          # "col" oder "rack"
    index: int           # Spaltennummer bzw. Ablageplatz
    color: int
    count: int | None


class State:
    __slots__ = ("grid", "cols", "rack", "path", "removed", "dead")

    def __init__(self, grid, cols, rack, path=(), removed=0, dead=False):
        self.grid = grid          # np.int16 array
        self.cols = cols          # tuple of tuples ((color, count), ...) - vorne zuerst
        self.rack = rack          # tuple Laenge RACK_SIZE, (color, count) oder None
        self.path = path
        self.removed = removed
        self.dead = dead

    def key(self):
        return (self.grid.tobytes(), self.cols, tuple(sorted(r for r in self.rack if r)))


def moves(st):
    out = []
    for i, col in enumerate(st.cols):
        if col:
            color, count = col[0]
            if color != UNKNOWN_COLOR:
                out.append(Move("col", i, color, count))
    seen = set()
    for j, sp in enumerate(st.rack):
        if sp and sp[0] != UNKNOWN_COLOR and sp not in seen:
            seen.add(sp)
            out.append(Move("rack", j, sp[0], sp[1]))
    return out


def apply(st, mv, lane_list):
    grid = st.grid.copy()
    cols = list(st.cols)
    rack = list(st.rack)
    if mv.source == "col":
        cols[mv.index] = cols[mv.index][1:]
    else:
        rack[mv.index] = None
    taken, rest = run_lap(grid, mv.color, mv.count, lane_list)
    dead = False
    if taken == 0 and mv.source == "rack":
        dead = True          # nutzlose Runde - Spule kaeme nur zurueck
    if rest and rest > 0:
        if None in rack:
            rack[rack.index(None)] = (mv.color, rest)
        else:
            dead = True
    return State(grid, tuple(cols), tuple(rack), st.path + (mv,), st.removed + taken, dead)


def cells_left(st):
    return int((st.grid >= 0).sum())


# Strafpunkte fuer belegte Ablageplaetze (0..5 belegt). Eine Masche zaehlt 10.
# Die Ablage soll moeglichst leer bleiben: schon der 3. Platz kostet so viel
# wie 30 Maschen, der 4./5. ist fast so schlimm wie verlieren.
RACK_PENALTY = (0.0, 40.0, 120.0, 300.0, 900.0, 2500.0)


def exposed_by_color(grid, lane_list):
    """Wie viele Bahnen zeigen gerade vorne eine Masche dieser Farbe?"""
    out = {}
    for kind, i in lane_list:
        cell = front_cell(grid, kind, i)
        if cell is not None:
            v = int(grid[cell])
            out[v] = out.get(v, 0) + 1
    return out


def evaluate(st, lane_list):
    if st.dead:
        return -1e9
    left = cells_left(st)
    if left == 0:
        return 1e9 - len(st.path)
    rack = [r for r in st.rack if r]
    score = st.removed * 10.0
    score -= RACK_PENALTY[min(len(rack), RACK_SIZE)]
    exposed = exposed_by_color(st.grid, lane_list)
    for color, count in rack:
        # Ablage-Spule, die gerade gar nichts nehmen kann, blockiert ihren Platz
        if exposed.get(color, 0) == 0:
            score -= 60.0
        # Reste, die in einer Runde aufgehen, sind gut - grosse Reste schlecht
        elif count is not None and count <= exposed.get(color, 0):
            score += 25.0
    # Wie viel koennten die jetzt antippbaren Spulen sofort nehmen?
    best_now = 0
    for mv in moves(st):
        g = st.grid.copy()
        t, rest = run_lap(g, mv.color, mv.count, lane_list)
        bonus = t + (15 if (rest is not None and rest <= 0) else 0)
        best_now = max(best_now, bonus)
    score += best_now * 2.0
    # Farben, fuer die keine Spule mehr bekannt ist, aber Maschen da sind
    colors_left = set(int(v) for v in np.unique(st.grid) if v >= 0)
    have = set(c for col in st.cols for c, _ in col) | set(r[0] for r in rack)
    unknown_spools = UNKNOWN_COLOR in have
    for c in colors_left:
        if c not in have and not unknown_spools:
            score -= 150.0
    return score


@dataclass
class Plan:
    moves: list
    complete: bool
    score: float
    dead_end: bool


def plan(grid, columns, rack, time_limit=2.0, beam_width=60, depth=14):
    """grid: np-Array (-1 leer, sonst Farb-Index);
    columns: Liste von Listen [(farbe, zahl), ...] vorderste zuerst;
    rack: Liste (Laenge 5) mit (farbe, zahl) oder None."""
    grid = np.asarray(grid, np.int16)
    lane_list = lanes(*grid.shape)
    rk = list(rack) + [None] * (RACK_SIZE - len(rack))
    st0 = State(grid, tuple(tuple(c) for c in columns), tuple(rk[:RACK_SIZE]))
    first = moves(st0)
    if not first:
        return Plan([], False, -1e9, True)
    deadline = time.time() + time_limit
    beam = [(evaluate(s, lane_list), s) for s in (apply(st0, m, lane_list) for m in first)]
    beam.sort(key=lambda t: -t[0])
    best = beam[0]
    seen = {s.key() for _, s in beam}
    for _ in range(depth - 1):
        if time.time() > deadline:
            break
        nxt = []
        for sc, st in beam:
            if st.dead or cells_left(st) == 0:
                continue
            for m in moves(st):
                s2 = apply(st, m, lane_list)
                k = s2.key()
                if k in seen:
                    continue
                seen.add(k)
                v = evaluate(s2, lane_list)
                nxt.append((v, s2))
                if v >= 1e8:
                    return Plan(list(s2.path), True, v, False)
            if time.time() > deadline:
                break
        if not nxt:
            break
        nxt.sort(key=lambda t: -t[0])
        beam = nxt[:beam_width]
        if beam[0][0] > best[0]:
            best = beam[0]
    sc, st = best
    return Plan(list(st.path), sc >= 1e8, sc, sc <= -1e8)
