"""Tests an echten Yarn-Loop-Bildern (Xiaomi, 1080x2400).
Laufen bei GitHub mit denselben Bibliotheksversionen wie in der App."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import yarn_bot as YB  # noqa: E402
import yarn_solver as S  # noqa: E402
import yarn_vision as Y  # noqa: E402


def load(name):
    img = cv2.imread(os.path.join(HERE, name))
    assert img is not None, name
    return img


def counts(scr):
    out = {}
    for v in scr.grid.ravel():
        if v >= 0:
            out[int(v)] = out.get(int(v), 0) + 1
    return sorted(out.values())


def front(scr):
    return [(c[0].count if c else None) for c in scr.columns]


def test_level4_start():
    s = Y.analyze(load("yarn_056.jpg"))
    assert s.kind == "game"
    assert (s.lattice.nx, s.lattice.ny) == (20, 20)
    assert counts(s) == [80, 160, 160]
    assert front(s) == [20, 20, 20]


def test_level7_start():
    s = Y.analyze(load("yarn_249.jpg"))
    assert s.kind == "game"
    assert (s.lattice.nx, s.lattice.ny) == (20, 20)
    assert sum(counts(s)) == 400
    assert front(s) == [20, 30, 40]


def test_level9_lattice():
    # 20 Spalten (nicht 21) und 20 Reihen - die 3D-Seitenflaeche unten ist keine Reihe
    s = Y.analyze(load("yarn_lvl9.jpg"))
    assert s.kind == "game"
    assert (s.lattice.nx, s.lattice.ny) == (20, 20)
    c = counts(s)
    assert sum(c) == 400
    assert all(n % 10 == 0 for n in c)   # Spulen haben immer Zehnerzahlen


def test_rack_numbers():
    s = Y.analyze(load("yarn_280.jpg"))
    assert s.kind == "game"
    assert [sp.count if sp else None for sp in s.rack] == [10, 3, 6, None, None]


def test_offer_popup_is_declined():
    img = load("yarn_184.jpg")
    s = Y.analyze(img)
    assert s.kind == "popup"
    taps = []

    class Dev:
        def tap(self, x, y):
            taps.append((x, y))

        def back(self):
            taps.append("back")

    YB.time.sleep = lambda x: None
    bot = YB.YarnBot(Dev(), {"dry_run": False}, log=lambda m: None)
    bot.menu_step(img, s)
    # Rotes X oben rechts am Fenster - niemals ein gruener Kauf-Knopf
    assert taps and taps[0] != "back"
    x, y = taps[0]
    assert y < 700 and x > 900


def test_bist_du_sicher_popup():
    # "Bist du sicher?" nach "Kein Platz mehr!": X oben rechts, nie "Spiel weiter 900"
    img = load("yarn_sicher.jpg")
    s = Y.analyze(img)
    assert s.kind == "popup"
    closes = [b for b in s.buttons if b[4] == "close"]
    assert len(closes) == 1          # das Herz im Angebot ist kein X
    taps = []

    class Dev:
        def tap(self, x, y):
            taps.append((x, y))

        def back(self):
            taps.append("back")

    YB.time.sleep = lambda x: None
    bot = YB.YarnBot(Dev(), {"dry_run": False}, log=lambda m: None)
    bot.menu_step(img, s)
    assert taps and taps[0] != "back"
    x, y = taps[0]
    assert abs(x - 990) < 40 and abs(y - 518) < 40


def test_lap_simulation():
    g = np.full((3, 4), 1, np.int16)
    g[2, :] = 0
    taken, rest = S.run_lap(g, 0, 10)
    assert taken == 4 and rest == 6
    assert (g[2] == -1).all()


def test_planner_on_real_level():
    s = Y.analyze(load("yarn_056.jpg"))
    bot = YB.YarnBot(object(), {}, log=lambda m: None)
    grid, cols, rack, taps = bot.model(s)
    p = S.plan(grid, cols, rack, time_limit=3)
    assert p.moves and not p.dead_end
    assert ("col", p.moves[0].index) in taps


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print("OK  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
