"""
Loeser fuer Car Match.

Regeln (aus Video + Recherche):
  * Ein Auto kann genommen werden, wenn mindestens eine Seite frei ist, d.h.
    ein Nachbarfeld (oben/unten/links/rechts) ist freie Strasse, die mit der
    Ausfahrt verbunden ist. Das Spiel zeigt das durch offene Augen an.
  * Genommene Autos landen in der Parkleiste (7 Plaetze). Drei gleiche
    Farben verschwinden sofort (Reihenfolge egal).
  * Ist die Leiste voll (7) ohne Dreier -> verloren.
  * Tunnel: Vor dem Tunnel steht ein Auto; wird es genommen, kommt das
    naechste (Farbe vorher unbekannt) heraus, bis der Zaehler 0 ist.

Suche: Strahlsuche (Beam Search) mit Bewertungsfunktion, danach - falls
noetig - Tiefensuche mit Zeitlimit, um eine komplette Loesung zu finden.
Nach jedem Zug wird der Bildschirm neu gelesen; ein kompletter Plan wird
weiterverfolgt, solange das Spielfeld zum Plan passt.

Intern: Bitmasken (Python-int) fuer Erreichbarkeit -> schnell genug fuer
Bretter mit ~100 Feldern und mehreren hunderttausend Knoten.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

ROAD, WALL, CAR, UNKNOWN = 0, 1, 2, 3
HOLDER_SIZE = 7

WALL_CODE, UNK_CODE, ROAD_CODE = -2, -1, 0
# Unbekannte Autos (aus Tunneln) bekommen Codes <= -10 (jedes einzigartig)


class Model:
    def __init__(self, rows, cols, cells, colors, tunnels):
        self.rows, self.cols = rows, cols
        self.cells = cells
        self.colors = colors
        self.tunnels = tunnels           # idx -> anzahl
        n = rows * cols
        self.nbmask = [0] * n
        self.nblist = [[] for _ in range(n)]
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        j = rr * cols + cc
                        self.nbmask[i] |= 1 << j
                        self.nblist[i].append(j)
        ex = 0
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if cells[i] == ROAD_CODE and (r == rows - 1 or c in (0, cols - 1)):
                    ex |= 1 << i
        if ex == 0:
            for i in range(max(0, (rows - 2) * cols), n):
                if cells[i] == ROAD_CODE:
                    ex |= 1 << i
        self.exit_mask = ex

    def idx(self, r, c):
        return r * self.cols + c

    def rc(self, i):
        return divmod(i, self.cols)

    # Erreichbare Strasse von der Ausfahrt aus
    def flood(self, cells, seed_mask, known=0):
        reach = known
        stack = [i for i in _bits(seed_mask) if cells[i] == ROAD_CODE and not (known >> i) & 1]
        for i in stack:
            reach |= 1 << i
        nbl = self.nblist
        while stack:
            i = stack.pop()
            for j in nbl[i]:
                if not (reach >> j) & 1 and cells[j] == ROAD_CODE:
                    reach |= 1 << j
                    stack.append(j)
        return reach


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def build_model(board):
    colors = ["road"]
    cells = []
    tunnels = {}
    for row in board.cells:
        for cell in row:
            if cell.kind == ROAD:
                cells.append(ROAD_CODE)
            elif cell.kind == WALL:
                cells.append(WALL_CODE)
            elif cell.kind == CAR and cell.color and cell.color != "?":
                if cell.color not in colors:
                    colors.append(cell.color)
                cells.append(colors.index(cell.color))
                if cell.tunnel:
                    tunnels[cell.r * board.cols + cell.c] = cell.tunnel
            else:
                cells.append(UNK_CODE)
    return Model(board.rows, board.cols, cells, colors, tunnels)


# ---------------------------------------------------------------------------
class State:
    __slots__ = ("cells", "holder", "tunnels", "reach", "cars", "unk", "removed", "path", "_tap")

    def __init__(self, cells, holder, tunnels, reach, cars, unk=0, removed=0, path=()):
        self.cells = cells        # tuple
        self.holder = holder      # tuple (sortiert)
        self.tunnels = tunnels    # tuple ((idx, n), ...)
        self.reach = reach        # int Bitmaske erreichbarer Strasse
        self.cars = cars          # int Bitmaske aller Autos (bekannt + unbekannt-enthuellt)
        self.unk = unk
        self.removed = removed
        self.path = path
        self._tap = None

    def key(self):
        return (self.cars, self.holder, self.tunnels, self.cells if self.unk else None)


def tappable(model, st):
    if st._tap is None:
        reach = st.reach
        nbm = model.nbmask
        st._tap = [i for i in _bits(st.cars) if nbm[i] & reach]
    return st._tap


def initial_state(model, holder_names):
    hold = []
    for name in holder_names:
        if name not in model.colors:
            model.colors.append(name)
        hold.append(model.colors.index(name))
    cells = tuple(model.cells)
    reach = model.flood(cells, model.exit_mask)
    cars = 0
    for i, v in enumerate(cells):
        if v > 0:
            cars |= 1 << i
    return State(cells, tuple(sorted(hold)), tuple(sorted(model.tunnels.items())), reach, cars)


def apply_move(model, st, i):
    cells = list(st.cells)
    color = cells[i]
    tun = st.tunnels
    unk = st.unk
    cars = st.cars
    reach = st.reach
    tdict = dict(tun) if tun else None
    if tdict and tdict.get(i, 0) > 0:
        unk += 1
        cells[i] = -10 - unk          # naechstes Auto, Farbe unbekannt
        tdict[i] -= 1
        if tdict[i] <= 0:
            del tdict[i]
        tun = tuple(sorted(tdict.items()))
    else:
        cells[i] = ROAD_CODE
        cars &= ~(1 << i)
        cells_t = tuple(cells)
        reach = model.flood(cells_t, 1 << i, reach)
    holder = st.holder + (color,)
    if color > 0 and holder.count(color) >= 3:
        holder = tuple(h for h in holder if h != color)
    else:
        holder = tuple(sorted(holder))
    return State(tuple(cells), holder, tun, reach, cars, unk, st.removed + 1, st.path + (i,))


def is_dead(st):
    return len(st.holder) >= HOLDER_SIZE


def cars_left(st):
    return bin(st.cars).count("1") + sum(n for _, n in st.tunnels)


def evaluate(model, st, color_total=None):
    if is_dead(st):
        return -1e9
    if st.cars == 0 and not st.tunnels and not st.holder:
        return 1e9 - len(st.path)
    h = st.holder
    n = len(h)
    score = st.removed * 10.0 - n * 7.0
    if n >= 4:
        score -= 18.0 * (n - 3) ** 2
    tap = tappable(model, st)
    cells = st.cells
    tap_cnt = Counter(cells[i] for i in tap)
    uncertain = bool(st.tunnels) or st.unk or UNK_CODE in cells
    board_cnt = None
    for col, k in Counter(h).items():
        if col <= -10:
            score -= 25.0
            continue
        need = 3 - k
        t = tap_cnt.get(col, 0)
        if t >= need:
            score += 6.0 * k
        elif t > 0:
            score += 2.0
        elif not uncertain:
            if board_cnt is None:
                board_cnt = Counter(cells[i] for i in _bits(st.cars))
            if board_cnt.get(col, 0) < need:
                score -= 400.0
    score += 1.5 * len(tap)
    return score


# ---------------------------------------------------------------------------
def beam_search(model, starts, width=200, deadline=None):
    beam = [(evaluate(model, s), s) for s in starts]
    beam.sort(key=lambda t: -t[0])
    best = beam[0]
    seen = {s.key() for _, s in beam}
    while beam and time.time() < deadline:
        nxt = []
        for _, st in beam:
            for i in tappable(model, st):
                s2 = apply_move(model, st, i)
                if is_dead(s2):
                    continue
                k = s2.key()
                if k in seen:
                    continue
                seen.add(k)
                sc = evaluate(model, s2)
                if sc >= 1e8:
                    return (sc, s2)
                nxt.append((sc, s2))
        if not nxt:
            break
        nxt.sort(key=lambda t: -t[0])
        beam = nxt[:width]
        if beam[0][0] > best[0]:
            best = beam[0]
    return best


def dfs_solve(model, starts, deadline, max_nodes=400000):
    visited = set()
    nodes = 0
    stack = sorted(starts, key=lambda s: evaluate(model, s))
    while stack:
        st = stack.pop()
        nodes += 1
        if nodes > max_nodes or ((nodes & 127) == 0 and time.time() > deadline):
            return None
        if st.cars == 0 and not st.tunnels and not st.holder:
            return st
        ch = []
        for i in tappable(model, st):
            s2 = apply_move(model, st, i)
            if is_dead(s2):
                continue
            k = s2.key()
            if k in visited:
                continue
            visited.add(k)
            ch.append((evaluate(model, s2), s2))
        ch.sort(key=lambda t: t[0])
        stack.extend(s for _, s in ch)
    return None


@dataclass
class Plan:
    moves: list
    complete: bool
    score: float
    dead_end: bool


def plan(board, holder_names, eyes_hint=None, time_limit=2.5, beam_width=200):
    """Hauptfunktion. eyes_hint = Autos mit offenen Augen (vom Spiel)."""
    model = build_model(board)
    st0 = initial_state(model, holder_names)
    moves0 = tappable(model, st0)
    if eyes_hint:
        eye_idx = sorted(model.idx(r, c) for (r, c) in eyes_hint
                         if 0 <= r < model.rows and 0 <= c < model.cols
                         and st0.cells[model.idx(r, c)] > 0)
        if eye_idx:
            moves0 = eye_idx          # Spiel sagt, diese sind frei
    if not moves0:
        return Plan([], False, -1e9, True)
    deadline = time.time() + time_limit
    firsts = []
    for i in moves0:
        s1 = apply_move(model, st0, i)
        if not is_dead(s1):
            firsts.append(s1)
    if not firsts:
        return Plan([], False, -1e9, True)
    for s1 in firsts:
        if evaluate(model, s1) >= 1e8:
            return Plan([model.rc(s1.path[0])], True, 1e9, False)

    uncertain = bool(model.tunnels) or UNK_CODE in model.cells
    # 1) Strahlsuche (gibt immer eine gute Teilloesung)
    sc, st = beam_search(model, firsts, width=beam_width,
                         deadline=time.time() + time_limit * (0.45 if not uncertain else 0.9))
    complete = sc >= 1e8
    # 2) Tiefensuche nach kompletter Loesung
    if not complete and not uncertain and time.time() < deadline:
        sol = dfs_solve(model, firsts, deadline)
        if sol is not None:
            st, sc, complete = sol, 1e9, True
    moves = [model.rc(i) for i in st.path]
    return Plan(moves, complete, sc, sc <= -1e8)


def simulate_holder(holder_names, color):
    h = list(holder_names) + [color]
    if h.count(color) >= 3:
        h = [x for x in h if x != color]
    return sorted(h)
