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
# Fadenknoten: eine Zelle mit mehreren Faeden. Gitterwert = farbe + KNOT*(n-1).
# Jede Bahn nimmt pro Runde einen Faden davon; bei 0 ist die Zelle frei.
KNOT = 64


def _take(grid, cell):
    v = int(grid[cell])
    grid[cell] = v - KNOT if v >= KNOT else -1


def weight(grid):
    """Maschen insgesamt (Knoten zaehlen mit ihrer Fadenzahl)."""
    g = grid[grid >= 0].astype(np.int32)
    return int((g // KNOT + 1).sum())


def color_counts(grid):
    out = {}
    for v in grid[grid >= 0].ravel():
        v = int(v)
        out[v % KNOT] = out.get(v % KNOT, 0) + v // KNOT + 1
    return out


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
        if grid[r, c] % KNOT == color:
            _take(grid, (r, c))
            count -= 1
            taken += 1
    return taken, count


def run_lap_multi(grid, spools, lane_list=None):
    """Mehrere Spulen direkt hintereinander (Doppelspule): an jeder Bahn
    kommt zuerst die vordere, dann die hintere vorbei. Veraendert grid!
    spools: [(farbe, zahl), ...]. Rueckgabe: (genommen gesamt, [reste])."""
    if lane_list is None:
        lane_list = lanes(*grid.shape)
    rest = [10 ** 6 if n is None else n for _, n in spools]
    taken = 0
    for kind, i in lane_list:
        if not any(r > 0 for r in rest):
            break
        for k, (color, _) in enumerate(spools):
            if rest[k] <= 0:
                continue
            cell = front_cell(grid, kind, i)
            if cell is None:
                break
            if grid[cell] % KNOT == color:
                _take(grid, cell)
                rest[k] -= 1
                taken += 1
    return taken, rest


# ---------------------------------------------------------------------------
@dataclass
class Move:
    source: str          # "col" oder "rack"
    index: int           # Spaltennummer bzw. Ablageplatz
    color: int
    count: int | None
    partner: tuple | None = None   # Doppelspule: (source, index, color, count)

    def spools(self):
        out = [(self.color, self.count)]
        if self.partner is not None:
            out.append((self.partner[2], self.partner[3]))
        return out


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


def _norm(item):
    """(farbe, zahl) oder (farbe, zahl, verbindung) -> immer 3 Werte."""
    if item is None:
        return None
    if len(item) == 2:
        return (item[0], item[1], -1)
    lid = item[2]
    return (item[0], item[1], -1 if lid is None else int(lid))


def moves(st):
    out = []
    fronts = {i: col[0] for i, col in enumerate(st.cols) if col}
    for i, (color, count, lid) in fronts.items():
        if color == UNKNOWN_COLOR:
            # Ueberraschungs-Spule ("?"-Kasten): Farbe erst nach dem Antippen
            # bekannt. Vorsichtig gerechnet: nimmt nichts, belegt einen
            # Ablageplatz - die Suche oeffnet sie also nur, wenn es sich lohnt
            # bzw. nichts anderes mehr geht. Danach wird ohnehin neu geplant.
            if lid < 0:
                out.append(Move("col", i, color, count if count else 10))
            continue
        if lid >= 0:
            # Doppelspule: nur antippbar, wenn der Partner auch vorne steht
            js = [j for j, f in fronts.items() if j != i and f[2] == lid]
            if not js or js[0] < i:
                continue
            j = js[0]
            c2, n2, _ = fronts[j]
            if c2 == UNKNOWN_COLOR:
                continue
            out.append(Move("col", i, color, count, ("col", j, c2, n2)))
        else:
            out.append(Move("col", i, color, count))
    seen = set()
    for j, sp in enumerate(st.rack):
        if not sp or sp[0] == UNKNOWN_COLOR:
            continue
        color, count, lid = sp
        if lid >= 0:
            ks = [k for k, o in enumerate(st.rack) if o and k != j and o[2] == lid]
            if ks:
                if ks[0] < j:
                    continue
                o = st.rack[ks[0]]
                if o[0] == UNKNOWN_COLOR:
                    continue
                key = (sp, o)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Move("rack", j, color, count, ("rack", ks[0], o[0], o[1])))
                continue
        if (color, count) in seen:
            continue
        seen.add((color, count))
        out.append(Move("rack", j, color, count))
    return out


def apply(st, mv, lane_list):
    grid = st.grid.copy()
    cols = list(st.cols)
    rack = list(st.rack)
    parts = [(mv.source, mv.index)]
    if mv.partner is not None:
        parts.append((mv.partner[0], mv.partner[1]))
    lid = -1
    for src, idx in parts:
        if src == "col":
            lid = max(lid, cols[idx][0][2])
            cols[idx] = cols[idx][1:]
        else:
            lid = max(lid, rack[idx][2])
            rack[idx] = None
    sp = mv.spools()
    taken, rests = run_lap_multi(grid, sp, lane_list)
    dead = False
    if taken == 0 and mv.source == "rack":
        dead = True          # nutzlose Runde - Spule kaeme nur zurueck
    left = [(sp[k][0], rests[k]) for k in range(len(sp)) if rests[k] > 0]
    keep_link = -1
    if len(sp) == 2 and left:
        # Doppelspule: bleibt zusammen - auch die leere (0) belegt einen Platz
        left = [(sp[k][0], max(0, rests[k])) for k in range(2)]
        keep_link = lid if lid >= 0 else 999
    for color, rest in left:
        if None in rack:
            rack[rack.index(None)] = (color, rest, keep_link)
        else:
            dead = True
    return State(grid, tuple(cols), tuple(rack), st.path + (mv,), st.removed + taken, dead)


def cells_left(st):
    return weight(st.grid)


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
            v = int(grid[cell]) % KNOT
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
    remaining = color_counts(st.grid)
    for color, count, _lid in rack:
        # Farbe fertig, Spule bleibt fuer immer liegen -> Platz ist verloren
        if color >= 0 and remaining.get(color, 0) == 0:
            score -= 400.0
            continue
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
        t, rests = run_lap_multi(g, mv.spools(), lane_list)
        bonus = t + 15 * sum(1 for r in rests if r <= 0)
        best_now = max(best_now, bonus)
    score += best_now * 2.0
    # Farben, fuer die keine Spule mehr bekannt ist, aber Maschen da sind
    colors_left = set(color_counts(st.grid))
    have = set(c[0] for col in st.cols for c in col) | set(r[0] for r in rack)
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


def plan_beam(grid, columns, rack, time_limit=2.0, beam_width=60, depth=14):
    """grid: np-Array (-1 leer, sonst Farb-Index);
    columns: Liste von Listen [(farbe, zahl), ...] vorderste zuerst;
    rack: Liste (Laenge 5) mit (farbe, zahl) oder None."""
    grid = np.asarray(grid, np.int16)
    lane_list = lanes(*grid.shape)
    rk = [_norm(r) for r in rack] + [None] * (RACK_SIZE - len(rack))
    st0 = State(grid, tuple(tuple(_norm(c) for c in col) for col in columns), tuple(rk[:RACK_SIZE]))
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


# ---------------------------------------------------------------------------
# Makro-Suche: Ein Zug = eine Spule aus einer Spalte losschicken und danach
# alle Ablage-Spulen, die etwas nehmen koennen, gleich hinterher (Ablage leeren).
# Dadurch bleibt die Ablage moeglichst leer, und die Suche muss nur noch die
# Reihenfolge der Spalten entscheiden - die kann sie fast vollstaendig
# durchprobieren (bis alle sichtbaren Spulen verplant sind).

def _flush(st, lane_list):
    """Ablage-Spulen losschicken, solange eine etwas nimmt (die beste zuerst)."""
    while True:
        best = None
        for mv in moves(st):
            if mv.source != "rack":
                continue
            s2 = apply(st, mv, lane_list)
            if s2.dead:
                continue
            take = s2.removed - st.removed
            if take > 0 and (best is None or take > best[0]):
                best = (take, s2)
        if best is None:
            return st
        st = best[1]


def _leaf_score(st, lane_list):
    if st.dead:
        return -1e9
    left = cells_left(st)
    if left == 0:
        return 1e9 - len(st.path)
    rack = [r for r in st.rack if r]
    score = st.removed * 10.0 - RACK_PENALTY[min(len(rack), RACK_SIZE)]
    exposed = exposed_by_color(st.grid, lane_list)
    remaining = color_counts(st.grid)
    for color, count, _ in rack:
        if remaining.get(color, 0) == 0:
            score -= 400.0
        elif exposed.get(color, 0) == 0:
            score -= 60.0
    # Spalten-Spulen, deren Farbe gerade frei liegt, sind gute naechste Zuege
    for col in st.cols:
        if col and col[0][0] >= 0 and exposed.get(col[0][0], 0) > 0:
            score += 5.0 * min(exposed[col[0][0]], col[0][1] or 0)
    return score


def plan(grid, columns, rack, time_limit=2.0, beam_width=60, depth=14):
    grid = np.asarray(grid, np.int16)
    lane_list = lanes(*grid.shape)
    rk = [_norm(r) for r in rack] + [None] * (RACK_SIZE - len(rack))
    st0 = State(grid, tuple(tuple(_norm(c) for c in col) for col in columns), tuple(rk[:RACK_SIZE]))
    deadline = time.time() + time_limit * 0.75
    root = _flush(st0, lane_list)
    best_sc, best_st = _leaf_score(root, lane_list), root
    if cells_left(root) == 0:
        return Plan(list(root.path), True, best_sc, False)
    beam = [root]
    seen = {root.key()}
    width = 250
    while beam and time.time() < deadline:
        nxt = []
        for st in beam:
            for mv in moves(st):
                if mv.source != "col":
                    continue
                s2 = apply(st, mv, lane_list)
                if s2.dead:
                    continue
                s2 = _flush(s2, lane_list)
                if s2.dead:
                    continue
                k = s2.key()
                if k in seen:
                    continue
                seen.add(k)
                sc = _leaf_score(s2, lane_list)
                if sc >= 1e8:
                    return Plan(list(s2.path), True, sc, False)
                nxt.append((sc, s2))
            if time.time() > deadline:
                break
        if not nxt:
            break
        nxt.sort(key=lambda t: -t[0])
        beam = [s for _, s in nxt[:width]]
        # tiefer = besser: ein Blatt, das weiter kommt, schlaegt ein flaches
        sc, st = nxt[0]
        best_sc, best_st = sc, st
    if best_st.path:
        return Plan(list(best_st.path), False, best_sc, False)
    # Keine Spalten-Spule ohne Ueberlauf moeglich -> alte Suche als Notnagel
    return plan_beam(grid, columns, rack, time_limit=max(0.2, deadline - time.time() + time_limit * 0.25),
                     beam_width=beam_width, depth=depth)
