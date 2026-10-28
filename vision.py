"""
Bilderkennung fuer "Car Match - Traffic Puzzle" (Grand Games).

Alles ist aufloesungsunabhaengig: Raster, Zellgroesse, Spielfeldgrenzen und
Parkleiste werden in jedem Screenshot neu aus den Farben gemessen.

Begriffe
--------
Zelle      : ein Feld im Parkplatz-Raster (ein Auto passt genau in eine Zelle)
ROAD       : freie Asphaltflaeche (dunkelblau) - hier koennen Autos fahren
WALL       : Gehweg / Bordstein / ausserhalb (lila / beige)
CAR        : Auto mit bekannter Farbe
UNKNOWN    : etwas, das blockiert, aber nicht sicher erkannt wurde
             (Kiste, Tunnel-Dach, Aufzug, verdecktes Auto ...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Farben
# --------------------------------------------------------------------------
# OpenCV-Hue (0..179). Bereiche sind grosszuegig; Autos sind stark gesaettigt.
COLOR_RANGES = [
    # name,    hue_lo, hue_hi
    ("red", 0, 7),
    ("orange", 8, 15),
    ("yellow", 16, 33),
    ("lime", 34, 44),
    ("green", 45, 80),
    ("cyan", 81, 92),
    ("blue", 93, 126),
    ("purple", 127, 150),
    ("pink", 151, 166),
    ("red", 167, 179),
]
_HUE_LUT = np.zeros(180, dtype=np.uint8)
COLOR_NAMES = []
for _name, _lo, _hi in COLOR_RANGES:
    if _name not in COLOR_NAMES:
        COLOR_NAMES.append(_name)
    _HUE_LUT[_lo:_hi + 1] = COLOR_NAMES.index(_name) + 1  # 0 = keine Farbe

# Zeichenfarben (BGR) fuer Debug-Bilder
DRAW_BGR = {
    "red": (40, 40, 230), "orange": (0, 140, 255), "yellow": (0, 220, 255),
    "lime": (0, 255, 180), "green": (40, 190, 40), "cyan": (230, 230, 0),
    "blue": (230, 120, 20), "purple": (200, 60, 160), "pink": (180, 100, 255),
    "white": (255, 255, 255), "black": (30, 30, 30), "?": (128, 128, 128),
}

ROAD, WALL, CAR, UNKNOWN = 0, 1, 2, 3


@dataclass
class Cell:
    r: int
    c: int
    kind: int = WALL
    color: Optional[str] = None
    eyes: bool = False          # Spiel zeigt offene Augen = antippbar
    cx: int = 0                 # Pixelmitte
    cy: int = 0
    car_frac: float = 0.0
    tunnel: int = 0             # >0: Tunnel/Garage darueber mit so vielen Autos


@dataclass
class Holder:
    x0: int
    x1: int
    y0: int
    y1: int
    slots: int = 7
    colors: list = field(default_factory=list)   # Farbe je Slot oder None

    @property
    def filled(self):
        return [c for c in self.colors if c]


@dataclass
class Board:
    rows: int
    cols: int
    px: float
    py: float
    x0: float          # Pixel-x der Mitte von Spalte 0
    y0: float          # Pixel-y der Mitte von Zeile 0
    cells: list        # cells[r][c]
    top: int
    bottom: int

    def cell(self, r, c):
        return self.cells[r][c]

    def tap_point(self, r, c):
        # Laut Spielern: eher unten/vorne aufs Auto tippen, sonst erwischt man
        # manchmal das Auto dahinter.
        x = self.x0 + c * self.px
        y = self.y0 + r * self.py + 0.18 * self.py
        return int(x), int(y)


@dataclass
class Screen:
    kind: str                      # "game", "home", "popup", "dim", "unknown"
    holder: Optional[Holder] = None
    board: Optional[Board] = None
    buttons: list = field(default_factory=list)   # [(x, y, w, h, label)]
    boosters: list = field(default_factory=list)  # [(x, y)]
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------
def _hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def car_mask(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    strong = (s > 120) & (v > 110)
    # helle, blasse Dach-/Scheibenflaechen der blauen Autos
    light_blue = (h >= 88) & (h <= 112) & (s > 40) & (v > 190)
    return strong | light_blue


def color_labels(hsv):
    """Pro Pixel: 0 = kein Auto, sonst Index+1 in COLOR_NAMES."""
    lab = _HUE_LUT[hsv[..., 0]]
    lab = np.where(car_mask(hsv), lab, 0).astype(np.uint8)
    return lab


def beige_mask(hsv):
    """Bordstein / Rahmen der Parkleiste (hell-rosa-beige)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 14) | (h >= 170)) & (s > 35) & (s < 125) & (v > 150)


def road_mask(hsv):
    """Dunkler Asphalt (dunkles Blau-Lila)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 100) & (h <= 135) & (v >= 55) & (v <= 135) & (s >= 50) & (s <= 150)


def sidewalk_mask(hsv):
    """Heller lila Gehweg."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 118) & (h <= 150) & (v > 135) & (s >= 30) & (s <= 120)


def _runs(bool_arr, min_len=1):
    """Zusammenhaengende True-Laeufe -> [(start, end_exclusive)]."""
    a = np.concatenate([[0], bool_arr.astype(np.int8), [0]])
    d = np.diff(a)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


# --------------------------------------------------------------------------
# Parkleiste (7 Slots unten)
# --------------------------------------------------------------------------
def find_holder(img, hsv=None) -> Optional[Holder]:
    H, W = img.shape[:2]
    if hsv is None:
        hsv = _hsv(img)
    bm = beige_mask(hsv)
    rm = road_mask(hsv) | ((hsv[..., 2] < 135) & (hsv[..., 2] > 40))
    y_lo, y_hi = int(H * 0.55), int(H * 0.92)
    rowfrac = bm[y_lo:y_hi].mean(axis=1)
    # Kandidaten-Zeilen fuer Ober-/Unterkante: lange beige Linien
    cand = rowfrac > 0.45
    lines = [(y_lo + s, y_lo + e) for s, e in _runs(cand)]
    if not lines:
        return None
    best = None
    # Unterkante: von unten nach oben die erste lange Linie, darueber dunkles Inneres
    for (b0, b1) in reversed(lines):
        # Seitenraender der Linie
        row = bm[(b0 + b1) // 2]
        xs = np.where(row)[0]
        if len(xs) < W * 0.4:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        # nach oben suchen: Inneres (dunkel) und eine weitere Linie bzw. Zellraender
        for (t0, t1) in reversed(lines):
            if t1 >= b0:
                continue
            gap = b0 - t1
            if gap < H * 0.025 or gap > H * 0.09:
                continue
            inner = rm[t1 + 2:b0 - 2, x0 + 10:x1 - 10]
            if inner.size == 0:
                continue
            best = (x0, x1, t0, b1)
            break
        if best is None:
            # Oberkante evtl. komplett von Autos verdeckt: Hoehe ueber Trennstriche bestimmen
            colfrac = bm[max(0, b0 - int(H * 0.08)):b0, x0:x1].mean(axis=0)
            dividers = _runs(colfrac > 0.5)
            if len(dividers) >= 4:
                hh = int(H * 0.052)
                best = (x0, x1, b0 - hh, b1)
        if best:
            break
    if best is None:
        return None
    x0, x1, y0, y1 = best
    # Trennstriche zaehlen -> Anzahl Slots
    band = bm[y0 + (y1 - y0) // 4: y1 - (y1 - y0) // 4, x0 + 8:x1 - 8]
    colfrac = band.mean(axis=0) if band.size else np.zeros(1)
    divs = _runs(colfrac > 0.5)
    slots = 7   # laut Spiel immer 7 Parkplaetze
    hold = Holder(x0, x1, y0, y1, slots)
    hold.colors = read_holder(img, hsv, hold)
    return hold


def read_holder(img, hsv, hold: Holder):
    lab = color_labels(hsv)
    sw = (hold.x1 - hold.x0) / hold.slots
    h = hold.y1 - hold.y0
    out = []
    for i in range(hold.slots):
        cx = hold.x0 + (i + 0.5) * sw
        xa, xb = int(cx - sw * 0.3), int(cx + sw * 0.3)
        ya, yb = int(hold.y0 + h * 0.15), int(hold.y1 - h * 0.1)
        patch = lab[ya:yb, xa:xb]
        out.append(_dominant(patch, min_frac=0.25))
    return out


def _dominant(lab_patch, min_frac=0.3):
    if lab_patch.size == 0:
        return None
    cnt = np.bincount(lab_patch.ravel(), minlength=len(COLOR_NAMES) + 1)
    total = lab_patch.size
    cnt[0] = 0
    k = int(cnt.argmax())
    if k == 0 or cnt[k] < total * min_frac:
        return None
    return COLOR_NAMES[k - 1]


# --------------------------------------------------------------------------
# Booster-Knoepfe (4 blaue runde Buttons unten)
# --------------------------------------------------------------------------
def find_boosters(img, hsv, below_y):
    H, W = img.shape[:2]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 95) & (h <= 120) & (s > 150) & (v > 170)).astype(np.uint8)
    m[:below_y] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        if a > (W * 0.05) ** 2 and 0.6 < w / max(hh, 1) < 1.6 and w < W * 0.3:
            out.append((int(cen[i][0]), int(cen[i][1])))
    out.sort()
    return out


# --------------------------------------------------------------------------
# Spielfeld-Oberkante (unter dem Stadt-Banner)
# --------------------------------------------------------------------------
def find_board_top(img, hsv, holder_y):
    H, W = img.shape[:2]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Wasserstreifen (tuerkis) unten am Banner, ueber volle Breite
    water = (h >= 82) & (h <= 104) & (s > 90) & (v > 170)
    frac = water[: int(H * 0.40)].mean(axis=1)
    rows = np.where(frac > 0.55)[0]
    if len(rows):
        return int(rows.max()) + 1
    # Fallback: Zeile, ab der Gehweg/Asphalt/Bordstein ueberwiegt
    bg = sidewalk_mask(hsv) | road_mask(hsv) | beige_mask(hsv)
    frac = bg[: holder_y].mean(axis=1)
    good = frac > 0.5
    for y in range(int(H * 0.08), holder_y):
        if good[y:y + int(H * 0.01)].all():
            return y
    return int(W * 0.41)


# --------------------------------------------------------------------------
# Raster bestimmen
# --------------------------------------------------------------------------
def _pitch_autocorr(profile, lo, hi):
    p = profile - profile.mean()
    n = len(p)
    best, bestv = None, -1e18
    for lag in range(int(lo), min(int(hi) + 1, n - 1)):
        val = float(np.dot(p[:-lag], p[lag:])) / (n - lag)
        if val > bestv:
            best, bestv = lag, val
    return best


def _refine_pitch(profile, p0):
    """Feinere Periode via Kammfilter ueber Sub-Pixel-Schritte."""
    best, bestv = p0, -1e18
    n = len(profile)
    for p in np.arange(p0 - 3, p0 + 3.01, 0.25):
        for ph in np.arange(0, p, 1.0):
            idx = np.arange(ph, n, p).astype(int)
            val = profile[idx].mean()
            if val > bestv:
                best, bestv = p, val
    return best


def _estimate_px(lab, top, bottom, W):
    """Spaltenabstand: Abstand Anfang-zu-Anfang aufeinanderfolgender
    Auto-Laeufe gleicher Breite in einer Bildzeile (sehr stabil, auch wenn
    Scheiben/Kuehler die Laeufe zerteilen)."""
    m = lab[top:bottom] > 0
    dists = []
    step = max(1, (bottom - top) // 250)
    for y in range(0, m.shape[0], step):
        runs = _runs(m[y], min_len=int(W * 0.045))
        for (a, b), (c, d) in zip(runs, runs[1:]):
            w1, w2 = b - a, d - c
            if 0.8 < w1 / max(w2, 1) < 1.25 and W * 0.08 < c - a < W * 0.30:
                dists.append(c - a)
    if len(dists) >= 8:
        hist = np.bincount(np.array(dists, int)).astype(float)
        hist = np.convolve(hist, np.ones(5), "same")
        return float(hist.argmax())
    widths = []
    for y in range(0, m.shape[0], step):
        widths += [b - a for a, b in _runs(m[y], min_len=int(W * 0.06)) if b - a < W * 0.3]
    if widths:
        return float(np.percentile(widths, 90)) / 0.92
    return None


def _comb_fit(values, p_lo, p_hi, tol_frac=0.12):
    """Findet (p, phase) so dass moeglichst viele Werte auf phase + k*p liegen."""
    v = np.asarray(values, float)
    best = (None, None, -1)
    for p in np.arange(p_lo, p_hi, 0.5):
        ph = v % p
        # Kreis-Histogramm
        nb = 48
        hist = np.bincount((ph / p * nb).astype(int) % nb, minlength=nb).astype(float)
        hist3 = hist + np.roll(hist, 1) + np.roll(hist, -1)
        b = int(hist3.argmax())
        center = (b + 0.5) / nb * p
        d = np.abs(((v - center + p / 2) % p) - p / 2)
        score = (d < tol_frac * p).sum() - 0.002 * p
        if score > best[2]:
            best = (p, center, score)
    return best


def _lsq_refine(values, p, phase, tol_frac=0.15):
    v = np.asarray(values, float)
    k = np.round((v - phase) / p)
    d = np.abs(v - (phase + k * p))
    sel = d < tol_frac * p
    if sel.sum() < 3 or len(np.unique(k[sel])) < 2:
        return p, phase
    A = np.vstack([k[sel], np.ones(sel.sum())]).T
    (p2, ph2), *_ = np.linalg.lstsq(A, v[sel], rcond=None)
    if abs(p2 - p) > 0.05 * p:
        return p, phase
    return float(p2), float(ph2 % p2)


PY_RATIO = 0.94   # Zellhoehe / Zellbreite (gemessen, durch die 3D-Perspektive)


def detect_grid(img, hsv, top, bottom, px_hint=None):
    """Liefert (px, py, x0, y0) oder None. x0/y0 = Pixelmitte einer Zelle.
    px_hint: erwartete Zellbreite (Slotbreite der Parkleiste)."""
    H, W = img.shape[:2]
    lab = color_labels(hsv)
    m = lab[top:bottom] > 0
    if m.sum() < (W * 0.06) ** 2:
        return None
    # ---------- Spalten ----------
    px0 = _estimate_px(lab, top, bottom, W)
    if px_hint:
        # Vielfache (Autos nur in jeder 2. Spalte) bzw. halbe Werte abfangen
        if px0 is None or not (0.6 * px_hint < px0 < 1.25 * px_hint):
            cands = [px0 / k for k in (2, 3) if px0] + [px0 * 2 if px0 else 0]
            ok = [c for c in cands if 0.6 * px_hint < c < 1.25 * px_hint]
            px0 = ok[0] if ok else px_hint
    if px0 is None:
        return None
    lefts, rights = [], []
    step = max(1, (bottom - top) // 250)
    for y in range(0, m.shape[0], step):
        for (a, b) in _runs(m[y], min_len=int(px0 * 0.3)):
            lefts.append(a)
            rights.append(b)
    best = None
    for p in np.arange(px0 * 0.96, px0 * 1.04, 0.25):
        sc = 0
        cen = []
        for vals in (lefts, rights):
            v = np.asarray(vals, float)
            nb = 40
            hist = np.bincount(((v % p) / p * nb).astype(int) % nb, minlength=nb).astype(float)
            hist3 = hist + np.roll(hist, 1) + np.roll(hist, -1)
            bi = int(hist3.argmax())
            c = (bi + 0.5) / nb * p
            d = np.abs(((v - c + p / 2) % p) - p / 2)
            sc += (d < 0.06 * p).sum()
            cen.append(c)
        if best is None or sc > best[0]:
            best = (sc, p, cen)
    _, px, (lph, rph) = best
    px, lph = _lsq_refine(lefts, px, lph, 0.08)
    d = (rph - lph) % px
    x0 = (lph + d / 2) % px

    # ---------- Zeilen ----------
    tops, bottoms = [], []
    ncols = int(W / px) + 2
    for k in range(-1, ncols):
        cx = x0 + k * px
        a, b = int(cx - px * 0.18), int(cx + px * 0.18)
        if a < 0 or b > W:
            continue
        strip = lab[top:bottom, a:b]
        cnts = np.stack([(strip == i).sum(axis=1) for i in range(len(COLOR_NAMES) + 1)], 1)
        seq = cnts.argmax(axis=1)
        frac = cnts.max(axis=1) / strip.shape[1]
        seq = np.where(frac > 0.4, seq, 0)
        seq = _smooth_labels(seq, int(px * 0.20))
        i, n = 0, len(seq)
        while i < n:
            if seq[i] == 0:
                i += 1
                continue
            j = i
            while j < n and seq[j] == seq[i]:
                j += 1
            if j - i > px * 0.55:
                if i > 2:
                    tops.append(top + i)
                if j < n - 2:
                    bottoms.append(top + j)
            i = j
    if len(tops) + len(bottoms) < 2:
        return None
    best = None
    for py in np.arange(px * (PY_RATIO - 0.04), px * (PY_RATIO + 0.04), 0.5):
        sc = 0
        res = []
        for vals in (tops, bottoms):
            if len(vals) == 0:
                res.append(None)
                continue
            v = np.asarray(vals, float)
            nb = 40
            hist = np.bincount(((v % py) / py * nb).astype(int) % nb, minlength=nb).astype(float)
            hist3 = hist + np.roll(hist, 1) + np.roll(hist, -1)
            bidx = int(hist3.argmax())
            center = (bidx + 0.5) / nb * py
            d = np.abs(((v - center + py / 2) % py) - py / 2)
            sc += (d < 0.10 * py).sum()
            res.append(center)
        if best is None or sc > best[0]:
            best = (sc, py, res)
    _, py, (ct, cb) = best
    # Wenige Zeilen -> Periode nicht bestimmbar, Erfahrungswert nehmen
    rows_seen = len(set(np.round(np.asarray(tops + bottoms) / py).astype(int)))
    if rows_seen < 4:
        py = px * PY_RATIO
        for vals, idx in ((tops, 0), (bottoms, 1)):
            if vals:
                v = np.asarray(vals, float)
                ang = (v % py) / py * 2 * math.pi
                c = (math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * math.pi) * py) % py
                if idx == 0:
                    ct = c
                else:
                    cb = c
    else:
        if ct is not None:
            py2, ct2 = _lsq_refine(tops, py, ct, 0.12)
            if abs(py2 / px - PY_RATIO) < 0.05:
                py, ct = py2, ct2
    # Zellmitte: Autodach-Kante + halbe Hoehe (Sprites ragen ca. 6% nach oben)
    if ct is not None:
        y0 = (ct + py * 0.52) % py
    else:
        y0 = (cb - py * 0.48) % py
    return float(px), float(py), float(x0 % px), float(y0)


def _smooth_labels(seq, gap):
    """Fuellt kurze Luecken (0 oder fremdes Label) zwischen gleichen Labels."""
    seq = seq.copy()
    n = len(seq)
    i = 0
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        # Lauf [i, j)
        if seq[i] != 0 or True:
            k = j
            # naechster Lauf mit gleichem Label innerhalb gap?
            if j < n and seq[i] != 0:
                m = j
                while m < n and m - j <= gap and seq[m] != seq[i]:
                    m += 1
                if m < n and m - j <= gap and seq[m] == seq[i]:
                    seq[j:m] = seq[i]
                    continue  # gleichen Lauf erneut verlaengern
        i = j
    return seq


# --------------------------------------------------------------------------
# Zellen klassifizieren
# --------------------------------------------------------------------------
def eyes_score(img_bgr, hsv, cx, cy, px, py):
    """Offene Augen auf der Windschutzscheibe = antippbares Auto.
    Gesucht werden dunkle Pupillen, die von Weiss umgeben sind."""
    H, W = hsv.shape[:2]
    ya, yb = int(cy - py * 0.40), int(cy + py * 0.42)
    xa, xb = int(cx - px * 0.42), int(cx + px * 0.42)
    ya, yb, xa, xb = max(0, ya), min(H, yb), max(0, xa), min(W, xb)
    p = hsv[ya:yb, xa:xb]
    if p.size == 0:
        return 0.0
    white = ((p[..., 1] < 55) & (p[..., 2] > 185)).astype(np.uint8)
    if white.mean() < 0.012:
        return 0.0
    dark = (p[..., 2] < 75).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    k = max(3, int(px * 0.05)) | 1
    kern = np.ones((k, k), np.uint8)
    pupils = 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (px * 0.055 <= w <= px * 0.45 and py * 0.055 <= h <= py * 0.30):
            continue
        if a < (px * 0.05) ** 2:
            continue
        blob = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        pad = k
        y1, y2 = max(0, y - pad), min(p.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(p.shape[1], x + w + pad)
        full = np.zeros((y2 - y1, x2 - x1), np.uint8)
        full[y - y1:y - y1 + h, x - x1:x - x1 + w] = blob
        ring = cv2.dilate(full, kern) - full
        wf = white[y1:y2, x1:x2][ring > 0].mean() if ring.any() else 0
        if wf > 0.15:
            pupils += 2 if w > px * 0.3 else 1   # beide Pupillen verschmolzen
    return min(1.0, pupils / 2.0)


def classify_cells(img, hsv, grid, top, bottom):
    px, py, x0, y0 = grid
    H, W = img.shape[:2]
    lab = color_labels(hsv)
    rm = road_mask(hsv)
    sm = sidewalk_mask(hsv) | beige_mask(hsv)
    # Zeilen/Spalten so waehlen, dass die Zellmitte im Spielfeld liegt
    c_first = int(math.ceil((px * 0.40 - x0) / px))
    c_last = int(math.floor((W - px * 0.40 - x0) / px))
    r_first = int(math.ceil((top + py * 0.30 - y0) / py))
    r_last = int(math.floor((bottom - py * 0.25 - y0) / py))
    cols = list(range(c_first, c_last + 1))
    rows = list(range(r_first, r_last + 1))
    cells = []
    for ri, r in enumerate(rows):
        line = []
        cy = y0 + r * py
        for ci, c in enumerate(cols):
            cx = x0 + c * px
            cell = Cell(ri, ci, cx=int(cx), cy=int(cy))
            xa, xb = int(cx - px * 0.30), int(cx + px * 0.30)
            ya, yb = int(cy - py * 0.25), int(cy + py * 0.30)
            xa, ya = max(0, xa), max(0, ya)
            pl = lab[ya:yb, xa:xb]
            frac = (pl > 0).mean() if pl.size else 0
            cell.car_frac = float(frac)
            if frac > 0.30:
                cell.kind = CAR
                cell.color = _dominant(pl, min_frac=0.18) or "?"
                if cell.color == "?":
                    cell.kind = UNKNOWN
                cell.eyes = eyes_score(img, hsv, cx, cy, px, py) >= 0.5
            else:
                # Hintergrund: Mitte der Zelle
                xa2, xb2 = int(cx - px * 0.2), int(cx + px * 0.2)
                ya2, yb2 = int(cy - py * 0.2), int(cy + py * 0.2)
                rf = rm[ya2:yb2, xa2:xb2].mean()
                sf = sm[ya2:yb2, xa2:xb2].mean()
                pv = hsv[ya2:yb2, xa2:xb2]
                whitef = ((pv[..., 1] < 40) & (pv[..., 2] > 205)).mean() if pv.size else 0
                blackf = (pv[..., 2] < 50).mean() if pv.size else 0
                is_white = whitef > 0.50 and blackf < 0.12
                is_black = blackf > 0.55 and whitef < 0.08
                if is_white or is_black:
                    # weisse / schwarze Autos (nicht gesaettigt); Tutorial-Text
                    # (weiss mit schwarzem Rand) wird dadurch ausgeschlossen
                    cell.kind = CAR
                    cell.color = "white" if is_white else "black"
                    cell.eyes = eyes_score(img, hsv, cx, cy, px, py) >= 0.5
                elif rf > 0.55:
                    cell.kind = ROAD
                elif sf > 0.45:
                    cell.kind = WALL
                elif frac > 0.12:
                    cell.kind = UNKNOWN
                else:
                    # unklarer Inhalt (Kiste, Tunnel, Effekt ...)
                    cell.kind = UNKNOWN if pv[..., 1].mean() > 100 else WALL
            line.append(cell)
        cells.append(line)
    # Pixelmitte von Zeile 0 / Spalte 0 im neuen Index
    bx0 = x0 + cols[0] * px if cols else x0
    by0 = y0 + rows[0] * py if rows else y0
    return Board(len(rows), len(cols), px, py, bx0, by0, cells, top, bottom)


# --------------------------------------------------------------------------
# Tunnel / Garagen (lila Bogen mit Zahl)
# --------------------------------------------------------------------------
def detect_tunnels(img, hsv, board: Board):
    """Tunnel/Garage: kraeftig violetter Bogen mit weisser Zahl oben und
    dunkler Oeffnung. Liefert [(r, c, anzahl, (dr, dc))] - (dr, dc) zeigt
    auf die Zelle, in der das naechste Auto erscheint."""
    found = []
    px, py = board.px, board.py
    for row in board.cells:
        for cell in row:
            if cell.eyes:
                continue
            if not (cell.kind == UNKNOWN or (cell.kind == CAR and cell.color == "purple")):
                continue
            xa, xb = int(cell.cx - px * 0.45), int(cell.cx + px * 0.45)
            ya, yb = int(cell.cy - py * 0.48), int(cell.cy + py * 0.48)
            p = hsv[max(0, ya):yb, max(0, xa):xb]
            if p.size == 0:
                continue
            violet = ((p[..., 0] >= 125) & (p[..., 0] <= 150) & (p[..., 1] > 120) & (p[..., 2] > 150)).mean()
            dark = p[..., 2] < 50
            if violet < 0.2 or dark.mean() < 0.03:
                continue
            ys, xs = np.where(dark)
            dy = ys.mean() / p.shape[0] - 0.5
            dx = xs.mean() / p.shape[1] - 0.5
            if abs(dy) >= abs(dx):
                d = (1, 0) if dy > 0 else (-1, 0)
            else:
                d = (0, 1) if dx > 0 else (0, -1)
            cnt = _read_number(img[max(0, ya - int(py * 0.2)):int(cell.cy + py * 0.1), max(0, xa):xb])
            found.append((cell.r, cell.c, cnt, d))
    return found


def _read_number(patch):
    """Zahl auf dem Tunnel: dunkle Ziffer auf hell-lila Band. Tesseract optional."""
    try:
        import pytesseract
    except Exception:
        return 1
    try:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        light = ((hsv[..., 1] < 90) & (hsv[..., 2] > 195)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(light)
        if n < 2:
            return 1
        i = 1 + int(np.argmax(st[1:, 4]))
        x, y, w, h, _ = st[i]
        band = hsv[y:y + h, x:x + w]
        digit = (band[..., 2] < 130).astype(np.uint8) * 255
        # Rand (Bogenkontur) abschneiden
        mx, my = max(1, w // 8), max(1, h // 10)
        digit[:, :mx] = 0
        digit[:, -mx:] = 0
        digit[-my:, :] = 0
        ys, xs = np.where(digit > 0)
        if len(xs) < 10:
            return 1
        crop = 255 - digit[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        crop = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        for psm in (7, 8, 10):
            txt = pytesseract.image_to_string(
                crop, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789")
            digits = "".join(ch for ch in txt if ch.isdigit())
            if digits:
                return int(digits)
    except Exception:
        pass
    return 1


# --------------------------------------------------------------------------
# Andere Bildschirme
# --------------------------------------------------------------------------
def find_green_buttons(img, hsv):
    """Grosse gruene Buttons (z.B. 'Level 7', 'Continue', 'Try again')."""
    H, W = img.shape[:2]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 45) & (h <= 75) & (s > 140) & (v > 150)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        if w > W * 0.25 and hh > H * 0.03 and w / hh > 1.6 and a > 0.6 * w * hh:
            out.append((int(x + w / 2), int(y + hh / 2), int(w), int(hh), "green"))
    out.sort(key=lambda b: -b[2] * b[3])
    return out


def find_close_buttons(img, hsv):
    """Rote runde X-Buttons (Popups schliessen)."""
    H, W = img.shape[:2]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h <= 6) | (h >= 172)) & (s > 150) & (v > 150)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(red)
    out = []
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        # rund: das weisse X nimmt viel Flaeche weg -> nur ~40 % rot
        if not (W * 0.06 < w < W * 0.16 and 0.85 < w / max(hh, 1) < 1.18):
            continue
        if a < 0.33 * w * hh:
            continue
        # weisses X in der Mitte: Weiss liegt auf den Diagonalen
        cx, cy = int(x + w / 2), int(y + hh / 2)
        r = int(min(w, hh) / 2)
        sub = hsv[cy - r: cy + r, cx - r: cx + r]
        if sub.shape[0] != 2 * r or sub.shape[1] != 2 * r or r < 8:
            continue
        white = (sub[..., 1] < 70) & (sub[..., 2] > 200)
        d = 2 * r
        yy, xx = np.mgrid[0:d, 0:d]
        diag = (np.abs(yy - xx) < d * 0.08) | (np.abs(yy + xx - d) < d * 0.08)
        inner = ((yy - r) ** 2 + (xx - r) ** 2) < (r * 0.55) ** 2
        dw = white[diag & inner].mean()
        ow = white[inner & ~diag].mean()
        # alle vier Arme des X muessen weiss sein (ein Herz "oo" hat das nicht)
        arms = [white[diag & inner & qy & qx].mean()
                for qy in (yy < r, yy >= r) for qx in (xx < r, xx >= r)]
        if dw > 0.6 and dw - ow > 0.3 and min(arms) > 0.8:
            out.append((cx, cy, int(w), int(hh), "close"))
    return out


def screen_brightness(hsv):
    return float(hsv[..., 2].mean())


def analyze(img, grid=None, top=None) -> Screen:
    """grid/top: fest eingerastetes Raster des aktuellen Levels (optional)."""
    H, W = img.shape[:2]
    hsv = _hsv(img)
    holder = find_holder(img, hsv)
    greens = find_green_buttons(img, hsv)
    closes = find_close_buttons(img, hsv)
    bright = screen_brightness(hsv)
    if holder is not None:
        boosters = find_boosters(img, hsv, holder.y1)
        if top is None:
            top = find_board_top(img, hsv, holder.y0)
        if grid is None:
            grid = detect_grid(img, hsv, top, holder.y0,
                               px_hint=(holder.x1 - holder.x0) / holder.slots)
        board = None
        if grid is not None:
            board = classify_cells(img, hsv, grid, top, holder.y0)
            for (r, c, cnt, (dr, dc)) in detect_tunnels(img, hsv, board):
                cell = board.cells[r][c]
                cell.kind, cell.color = WALL, None
                cell.tunnel = max(cnt, 1)
                rr, cc = r + dr, c + dc
                if 0 <= rr < board.rows and 0 <= cc < board.cols:
                    board.cells[rr][cc].tunnel = max(cnt, 1)   # Ausfahrt
        scr = Screen("game", holder=holder, board=board, boosters=boosters,
                     buttons=closes + greens)
        scr.info["grid"] = grid
        scr.info["top"] = top
        scr.info["brightness"] = bright
        return scr
    if closes:
        return Screen("popup", buttons=closes + greens, info={"brightness": bright})
    if greens:
        return Screen("home" if greens[0][1] > H * 0.6 else "popup",
                      buttons=greens, info={"brightness": bright})
    if bright < 70:
        return Screen("dim", info={"brightness": bright})
    return Screen("unknown", info={"brightness": bright})


# --------------------------------------------------------------------------
# Debug-Ausgabe
# --------------------------------------------------------------------------
def draw_debug(img, scr: Screen):
    out = img.copy()
    H, W = img.shape[:2]
    t = max(2, W // 400)
    if scr.holder:
        h = scr.holder
        cv2.rectangle(out, (h.x0, h.y0), (h.x1, h.y1), (0, 255, 255), t)
        sw = (h.x1 - h.x0) / h.slots
        for i, col in enumerate(h.colors):
            cx = int(h.x0 + (i + 0.5) * sw)
            cv2.circle(out, (cx, h.y1 + 25), 14, DRAW_BGR.get(col, (60, 60, 60)) if col else (60, 60, 60), -1)
    if scr.board:
        b = scr.board
        for row in b.cells:
            for cell in row:
                x, y = cell.cx, cell.cy
                hx, hy = int(b.px * 0.45), int(b.py * 0.45)
                if cell.kind == CAR:
                    cv2.rectangle(out, (x - hx, y - hy), (x + hx, y + hy), DRAW_BGR.get(cell.color, (128, 128, 128)), t)
                    if cell.eyes:
                        cv2.circle(out, (x, y), int(b.px * 0.15), (255, 255, 255), -1)
                        cv2.circle(out, (x, y), int(b.px * 0.15), (0, 0, 0), t)
                elif cell.kind == ROAD:
                    cv2.circle(out, (x, y), 8, (0, 0, 0), -1)
                elif cell.kind == WALL:
                    cv2.line(out, (x - 10, y - 10), (x + 10, y + 10), (255, 255, 255), t)
                    cv2.line(out, (x - 10, y + 10), (x + 10, y - 10), (255, 255, 255), t)
                else:
                    cv2.putText(out, "?", (x - 12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 255), 3)
                if cell.tunnel:
                    cv2.putText(out, f"T{cell.tunnel}", (x - 25, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        cv2.line(out, (0, b.top), (W, b.top), (255, 0, 255), t)
    for (x, y, w, h, lab) in scr.buttons:
        cv2.rectangle(out, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 0, 255), t)
        cv2.putText(out, lab, (x - w // 2, y - h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    for (x, y) in scr.boosters:
        cv2.circle(out, (x, y), 20, (0, 255, 0), t)
    cv2.putText(out, scr.kind, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return out
