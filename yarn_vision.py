"""
Bilderkennung fuer "Yarn Loop" (Combo Game, com.combo.yarnflow).

Bildschirmaufbau (gemessen an 1080x2400, alles wird ueber das Foerderband
skaliert, funktioniert also bei jeder Aufloesung):

   Level-Schild
   +------------------------------+   <- Foerderband (cremefarbene Raender)
   |   Strickbild aus Maschen     |
   +------------------------------+
   [  ][  ][  ][  ][  ]               <- Ablage (5 Plaetze)
        (20) (30) (40)                <- Spulen-Spalten, vorderste antippbar
        (..) (..) (..)
   ============ Booster-Leiste ============
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import yarn_digits

# ---------------------------------------------------------------------------
# Referenzmasse (Video vom Xiaomi, 1080 x 2400)
# ---------------------------------------------------------------------------
REF_BELT_W = 963.0          # Breite Foerderband aussen (58..1021)
REF = {
    "inner_x0": 117, "inner_x1": 847,    # Bildbereich relativ zu belt.x0
    "inner_y0": 121, "inner_y1": 1006,   # relativ zu belt.y0
    "rack_y": 141,                       # Ablage: Zahl-Mitte unter belt.y1
    "rack_body_y": 200,
    "rack_x0": 132, "rack_dx": 175,      # Ablage-Platz i: belt.x0 + 132 + 175*i
    "spool_y": 361,                      # vorderste Spule: Zahl-Mitte unter belt.y1
    "spool_body_y": 425,
    "spool_dy": 192,                     # Abstand der Spulen-Reihen
    "spool_dx": 175,                     # Abstand der Spalten
    "disc_w": 130, "disc_h": 90,         # Ausschnitt um die Zahl
}

BG_REF = np.array([76, 68, 76], np.int16)   # Hintergrund (BGR)
UNKNOWN = "?"


@dataclass
class Belt:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def s(self):
        return (self.x1 - self.x0) / REF_BELT_W

    def X(self, rel):
        return int(round(self.x0 + rel * self.s))

    def Yb(self, rel):            # relativ zur Unterkante
        return int(round(self.y1 + rel * self.s))

    def Yt(self, rel):            # relativ zur Oberkante
        return int(round(self.y0 + rel * self.s))


@dataclass
class Lattice:
    x0: float
    y0: float
    px: float
    py: float
    nx: int
    ny: int

    def center(self, r, c):
        return (self.x0 + (c + 0.5) * self.px, self.y0 + (r + 0.5) * self.py)


@dataclass
class Spool:
    color: Optional[str]          # Palettenname oder "?" (Ueberraschung)
    count: Optional[int]          # Zahl auf der Spule
    x: int
    y: int                        # Tipp-Punkt
    lab: Optional[tuple] = None
    where: str = ""               # "col0".. / "rack2"
    link: Optional[int] = None    # verbundene Spulen (Doppelspule) teilen dieselbe Nummer


@dataclass
class YarnScreen:
    kind: str                     # "game", "popup", "dim", "unknown"
    belt: Optional[Belt] = None
    lattice: Optional[Lattice] = None
    grid: Optional[np.ndarray] = None     # ny x nx, -1 = leer, sonst Paletten-Index
    palette: list = field(default_factory=list)      # [(name, lab)]
    columns: list = field(default_factory=list)      # [[Spool vorne, dahinter ...]]
    rack: list = field(default_factory=list)         # Spool oder None je Platz
    buttons: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ziffern
# ---------------------------------------------------------------------------
def _load_digits():
    raw = base64.b64decode("".join(yarn_digits.DATA))
    d = np.load(io.BytesIO(raw))
    return d["T"].astype(np.float32) / 255.0, d["AR"].astype(np.float32)


_T, _AR = _load_digits()
_GW, _GH = 20, 28


def read_number(disc_bgr):
    """Zahl auf einer Spule. disc_bgr ist auf Referenzgroesse (130x90) skaliert.
    Rueckgabe: (zahl oder None, sicherheit 0..1).
    Vordere Spulen: weisse Ziffern. Hintere Spulen zeigen ihre Zahl blass
    (halb durchsichtig) - dafuer gibt es zwei Ersatz-Masken."""
    hsv = cv2.cvtColor(disc_bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 238) & (hsv[..., 1] < 40)).astype(np.uint8)
    best = _read_mask(m)
    if best[0] is not None and best[1] >= 0.25:
        return best
    for m2 in _faint_masks(disc_bgr):
        r = _read_mask(m2)
        if r[0] is not None and (best[0] is None or r[1] > best[1]):
            best = r
    return best


def _faint_masks(disc_bgr):
    H, W = disc_bgr.shape[:2]
    lab = cv2.cvtColor(disc_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    core = np.zeros((H, W), bool)
    core[int(H * 0.15):int(H * 0.85), int(W * 0.12):int(W * 0.88)] = True
    ring = L[core]
    Lmed = float(np.median(ring))
    cmed = float(np.median(chroma[core]))
    out = []
    # 1) heller und blasser als die Scheibe (farbige / schwarze Spulen)
    m1 = core & (L > Lmed + 15) & (chroma < max(cmed * 0.8, 12) + 6)
    out.append(m1.astype(np.uint8))
    # 1b) nur heller (z.B. hellgruene Ziffern auf gruener Scheibe)
    out.append((core & (L > Lmed + 15)).astype(np.uint8))
    out.append((core & (L > Lmed + 25)).astype(np.uint8))
    # 2) innerhalb der dunklen Umrandung der Ziffern (weisse Spulen)
    dark = (core & (L < Lmed - 45)).astype(np.uint8)
    if dark.any():
        filled = dark.copy()
        ff = np.zeros((H + 2, W + 2), np.uint8)
        inv = (1 - filled).astype(np.uint8)
        cv2.floodFill(inv, ff, (0, 0), 0)
        holes = inv.astype(bool) & ~dark.astype(bool)
        out.append((holes & core).astype(np.uint8))
    return out


def _read_mask(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if H * 0.33 <= h <= H * 0.8 and a >= 40 and w <= W * 0.45 and abs((y + h / 2) - H / 2) < H * 0.25:
            comps.append((x, y, w, h, i))
    if not comps or len(comps) > 4:
        return None, 0.0
    comps.sort()
    # Spulen tragen hoechstens zweistellige Zahlen - ueberzaehlige Flecken
    # (Glanzlichter) weglassen und die beste Lesart nehmen
    best = (None, 0.0)
    subsets = [comps] if len(comps) <= 2 else [comps[i:i + 2] for i in range(len(comps) - 1)]
    for sub in subsets:
        r = _read_digits(lab, sub)
        if r[0] is not None and r[0] < 100 and (best[0] is None or r[1] > best[1]):
            best = r
    return best


def _read_digits(lab, comps):
    digits, worst = [], 1.0
    for x, y, w, h, i in comps:
        g = cv2.resize((lab[y:y + h, x:x + w] == i).astype(np.float32), (_GW, _GH),
                       interpolation=cv2.INTER_AREA)
        ar = w / h
        dist = np.abs(_T - g[None]).mean(axis=(1, 2)) + 0.3 * np.abs(_AR - ar)
        k = int(np.argmin(dist))
        srt = np.sort(dist)
        conf = float(np.clip((srt[1] - srt[0]) / 0.08, 0, 1)) * float(np.clip(1.4 - srt[0] / 0.2, 0, 1))
        worst = min(worst, conf)
        digits.append(str(k))
    try:
        val = int("".join(digits))
    except ValueError:
        return None, 0.0
    return val, worst


def _disc(img, belt, cx, cy):
    s = belt.s
    hw, hh = int(REF["disc_w"] / 2 * s), int(REF["disc_h"] / 2 * s)
    H, W = img.shape[:2]
    x0, x1 = max(0, cx - hw), min(W, cx + hw)
    y0, y1 = max(0, cy - hh), min(H, cy + hh)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (REF["disc_w"], REF["disc_h"]), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Foerderband
# ---------------------------------------------------------------------------
def _groups(idx):
    out = []
    for v in idx:
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return out


def find_belt(img, hsv=None) -> Optional[Belt]:
    H, W = img.shape[:2]
    if hsv is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cream = (h >= 15) & (h <= 36) & (s >= 25) & (s <= 125) & (v >= 175)
    rows = cream[: int(H * 0.75)].sum(axis=1)
    cols = cream[: int(H * 0.75)].sum(axis=0)
    ry = _groups(np.where(rows > W * 0.45)[0])
    cx = _groups(np.where(cols > H * 0.22)[0])
    if len(ry) < 2 or len(cx) < 2:
        return None
    x0, x1 = cx[0][0], cx[-1][1]
    y0, y1 = ry[0][0], ry[-1][1]
    if (x1 - x0) < W * 0.6 or (y1 - y0) < (x1 - x0) * 0.8:
        return None
    return Belt(int(x0), int(y0), int(x1), int(y1))


# ---------------------------------------------------------------------------
# Strickbild
# ---------------------------------------------------------------------------
def _bg_color(region):
    q = (region[::4, ::4] // 8).reshape(-1, 3)
    vals, cnt = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-cnt)
    for i in order[:4]:
        c = vals[i] * 8 + 4
        if 50 < int(c.max()) < 110 and int(c.max()) - int(c.min()) < 25:
            return c.astype(np.int16)
    return BG_REF


def _runs(b, minlen):
    out, st = [], None
    for i, val in enumerate(b):
        if val and st is None:
            st = i
        elif not val and st is not None:
            if i - st >= minlen:
                out.append((st, i))
            st = None
    if st is not None and len(b) - st >= minlen:
        out.append((st, len(b)))
    return out


def _autocorr(p, lo, hi):
    p = p - p.mean()
    best, bl = -1e18, lo
    for lag in range(int(lo), min(int(hi), len(p) - 2)):
        val = float(np.dot(p[:-lag], p[lag:])) / (len(p) - lag)
        if val > best:
            best, bl = val, lag
    return bl


def _comb_score(prof, w, n):
    p = w / n
    tol = max(1, int(round(p * 0.08)))
    hits, mids = [], []
    for k in range(1, n):
        pos = int(round(k * p))
        a, b = max(0, pos - tol), min(len(prof), pos + tol + 1)
        if a < b:
            hits.append(float(prof[a:b].max()))
    for k in range(n):
        pos = int(round((k + 0.5) * p))
        a, b = max(0, pos - tol), min(len(prof), pos + tol + 1)
        if a < b:
            mids.append(float(prof[a:b].mean()))
    if not hits:
        return -1e9
    return float(np.median(hits)) - (float(np.mean(mids)) if mids else 0.0)


def _pitch_from_gaps(reg, bg, pmin, pmax):
    """Spaltenabstand grob schaetzen: In jeder Bildzeile die Abstaende
    zwischen benachbarten dunklen Fugen messen (nur wo dazwischen Maschen
    sind). Der haeufigste Abstand ist eine Spaltenbreite - Vielfache kommen
    so gar nicht erst vor."""
    d = np.abs(reg - bg).sum(axis=2)
    dark = reg.max(axis=2) < 45
    content = d > 60
    gaps = []
    for y in range(0, reg.shape[0], 2):
        xs = np.flatnonzero(dark[y])
        if len(xs) < 2:
            continue
        brk = np.flatnonzero(np.diff(xs) > 1)
        starts = np.r_[xs[0], xs[brk + 1]]
        ends = np.r_[xs[brk], xs[-1]]
        ok = (ends - starts) < 10
        cents = ((starts + ends) / 2.0)[ok]
        for a, b in zip(cents[:-1], cents[1:]):
            g = b - a
            if pmin <= g <= pmax and content[y, int(a) + 3:int(b) - 2].mean() > 0.9:
                gaps.append(g)
    if len(gaps) < 20:
        return None
    gaps = np.asarray(gaps)
    hist, edges = np.histogram(gaps, bins=np.arange(int(pmin), int(pmax) + 2, 1))
    mode = edges[int(np.argmax(hist))]
    sel = gaps[(gaps > mode - 3) & (gaps < mode + 4)]
    return float(np.median(sel))


def _pitch_phase(prof, pmin, pmax, step=0.25):
    """Spaltenabstand und Lage der Naehte direkt aus dem Profil der dunklen
    Fugen - unabhaengig davon, ob am Rand einzelne Maschen fehlen.
    Gibt (px, phase) zurueck, phase = Lage einer Naht ab Bildanfang."""
    n = len(prof)
    if n < 3 * pmin:
        return None, None
    tol = 2
    pad = np.pad(prof, tol, mode="edge")
    pf = np.max(np.stack([pad[i:i + n] for i in range(2 * tol + 1)]), axis=0)
    results = []
    for px in np.arange(pmin, min(pmax, n / 2.0) + 1e-6, step):
        best_ph, best_sc = 0, -1e9
        for ph in range(int(px)):
            seams = np.arange(ph, n, px)
            seams = seams[seams <= n - 1]          # float-arange kann n erreichen
            mids = seams + px / 2
            mids = mids[mids <= n - 1]
            if len(seams) < 3 or len(mids) < 2:
                continue
            sc = float(np.median(pf[seams.astype(int)])) - float(np.mean(prof[mids.astype(int)]))
            if sc > best_sc:
                best_sc, best_ph = sc, ph
        results.append((px, best_ph, best_sc))
    if not results:
        return None, None
    top = max(r[2] for r in results)
    if top < 0.05:
        return None, None
    # Vielfache des echten Abstands passen auch -> kleinsten guten nehmen,
    # danach im Umkreis von 10 % den besten
    for px, ph, sc in results:
        if sc >= 0.6 * top:
            near = [r for r in results if px * 0.95 <= r[0] <= px * 1.1]
            bp = max(near, key=lambda r: r[2])
            return float(bp[0]), float(bp[1])
    return None, None


def _best_count_all(prof, w, pmin, pmax):
    """Spaltenzahl: jede moegliche Anzahl pruefen (Naehte auf dem Raster?).
    Ein Vielfaches des echten Abstands (z.B. 3 Spalten auf einmal) passt auch
    gut - deshalb die GROESSTE Spaltenzahl nehmen, die fast so gut passt wie
    die beste."""
    lo = max(1, int(w / pmax))
    hi = max(lo, int(np.ceil(w / pmin)))
    scores = {n: _comb_score(prof, w, n) for n in range(max(2, lo), hi + 1)}
    if not scores:
        return 1
    best = max(scores.values())
    if best <= 0:
        return max(1, int(round(w / ((pmin + pmax) / 2))))
    good = [n for n, sc in scores.items() if sc >= 0.5 * best and sc >= 0.12]
    if not good:
        return max(scores, key=scores.get)
    return max(good)


def _best_count(prof, w, px):
    """Spaltenzahl genau bestimmen: Die Autokorrelation liefert den Abstand nur
    auf ein paar Prozent genau - bei 20 Spalten reicht das schon fuer eine
    Spalte zu viel. Deshalb fuer nx-1, nx, nx+1 pruefen, wie gut die dunklen
    Naehte zwischen den Spalten auf das Raster w/nx fallen."""
    n0 = max(1, int(round(w / max(px, 1e-6))))
    best, best_n = -1e9, n0
    for n in range(max(1, n0 - 2), n0 + 3):
        if n < 2:
            continue
        p = w / n
        tol = max(1, int(round(p * 0.08)))
        hits = []
        for k in range(1, n):
            pos = int(round(k * p))
            a, b = max(0, pos - tol), min(len(prof), pos + tol + 1)
            if a >= b:
                continue
            hits.append(float(prof[a:b].max()))
        if not hits:
            continue
        # Mitten der Spalten sollen hell sein (keine Naht)
        mids = []
        for k in range(n):
            pos = int(round((k + 0.5) * p))
            a, b = max(0, pos - tol), min(len(prof), pos + tol + 1)
            if a < b:
                mids.append(float(prof[a:b].mean()))
        score = float(np.median(hits)) - (float(np.mean(mids)) if mids else 0.0)
        if score > best + 1e-6:
            best, best_n = score, n
    return best_n


def picture_region(belt):
    return (belt.X(REF["inner_x0"]), belt.Yt(REF["inner_y0"]),
            belt.X(REF["inner_x1"]), belt.Yt(REF["inner_y1"]))


def detect_lattice(img, belt) -> Optional[Lattice]:
    X0, Y0, X1, Y1 = picture_region(belt)
    reg = img[Y0:Y1, X0:X1].astype(np.int16)
    if reg.size == 0:
        return None
    bg = _bg_color(reg)
    d = np.abs(reg - bg).sum(axis=2)
    m = (d > 60).astype(np.uint8)
    s = belt.s
    cr = _runs(m.mean(axis=0) > 0.12, int(12 * s))
    rr = _runs(m.mean(axis=1) > 0.12, int(12 * s))
    if not cr or not rr:
        return None
    x0, x1 = cr[0][0], cr[-1][1]
    y0, y1 = rr[0][0], rr[-1][1]
    w, h = x1 - x0, y1 - y0
    if w < 15 * s or h < 15 * s:
        return None
    sub = reg[y0:y1, x0:x1]
    dark = (sub.max(axis=2) < 45).astype(np.float32)
    prof = dark.mean(axis=0)
    px_est = _pitch_from_gaps(reg, bg, 18 * s, 110 * s)
    px, phase = (None, None)
    if px_est is not None:
        px, phase = _pitch_phase(prof, px_est * 0.96, px_est * 1.04, step=0.05)
    if px is None:
        nx = _best_count_all(prof, w, 14 * s, 100 * s)
        px = w / nx
    else:
        # Rasterlinie links an der Bildkante ausrichten
        first = phase - px * np.floor((phase + 0.3 * px) / px)
        x0 = int(round(x0 + first))
        nx = max(1, int(round((x1 - x0) / px)))
        # Bei vollstaendigem Rand ist die Gesamtbreite am genauesten
        exact = (x1 - x0) / nx
        if abs(exact - px) < 0.03 * px:
            px = exact
    # Zeilenabstand: im Spiel immer ca. 0.945 * Spaltenabstand (3D-Ansicht)
    py = px * 0.945
    # Unter der untersten Reihe sieht man deren 3D-Seitenflaeche (knapp eine
    # Reihe hoch, gleiche Farbe wie die Masche darueber) - das ist KEINE eigene
    # Reihe. Nachgemessen: Seitenflaeche ~0.9 * py.
    ny = max(1, int(round((h - 0.9 * py) / py)))
    return Lattice(X0 + x0, Y0 + y0, px, py, nx, ny)


def _lab(bgr_pixels):
    p = np.asarray(bgr_pixels, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(p, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


BLACK_LAB = np.array([20.0, 128.0, 128.0], np.float32)


def _color_of(patch_bgr, bg):
    """Farbe einer Masche: (lab, ist_leer). Farbton (a, b) als Median - der
    haengt kaum von der Schattierung ab; Helligkeit als 85%-Quantil."""
    p = patch_bgr.reshape(-1, 3).astype(np.int16)
    if p.size == 0:
        return None, True
    d = np.abs(p - bg).sum(axis=1)
    bgfrac = float((d < 45).mean())
    if bgfrac > 0.5:
        return None, True
    v = p.max(axis=1)
    darkfrac = float((v < 50).mean())
    if darkfrac > 0.5:
        return BLACK_LAB.copy(), False
    # schwarze Maschen haben graue Glanzlichter -> am Median erkennen
    allab = _lab(p[d >= 45].astype(np.uint8)) if np.any(d >= 45) else None
    if allab is not None and float(np.median(allab[:, 0])) < 90 \
            and float(np.median(np.hypot(allab[:, 1] - 128, allab[:, 2] - 128))) < 18:
        return BLACK_LAB.copy(), False
    sel = p[(v >= 50) & (d >= 45)]
    if len(sel) < 8:
        return BLACK_LAB.copy(), False
    lab = _lab(sel.astype(np.uint8))
    out = np.median(lab, axis=0)
    out[0] = np.percentile(lab[:, 0], 85)
    return out, False


def read_grid(img, lat: Lattice, belt: Belt):
    X0, Y0, X1, Y1 = picture_region(belt)
    bg = _bg_color(img[Y0:Y1, X0:X1].astype(np.int16))
    grid_lab = np.zeros((lat.ny, lat.nx, 3), np.float32)
    filled = np.zeros((lat.ny, lat.nx), bool)
    # Oberseite der Masche (Licht von oben links); Raender zeigen sonst ihre
    # dunklere 3D-Seitenflaeche
    hx, hy = lat.px * 0.33, lat.py * 0.33
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F))
    tex = np.zeros((lat.ny, lat.nx), np.float32)
    for r in range(lat.ny):
        for c in range(lat.nx):
            cx, cy = lat.center(r, c)
            cy -= lat.py * 0.05
            y0, y1 = int(cy - hy), int(cy + hy) + 1
            x0, x1 = int(cx - hx), int(cx + hx) + 1
            patch = img[y0:y1, x0:x1]
            lab, empty = _color_of(patch, bg)
            if not empty:
                filled[r, c] = True
                grid_lab[r, c] = lab
                t = lap[y0:y1, x0:x1]
                tex[r, c] = float(t.mean()) if t.size else 0.0
    _drop_side_faces(grid_lab, filled, tex)
    return grid_lab, filled, bg


def _drop_side_faces(grid_lab, filled, tex):
    """Unter der jeweils untersten Masche einer Spalte sieht man ihre glatte,
    dunklere 3D-Seitenflaeche - fast eine Zelle hoch. Die ist KEINE Masche
    (sonst haelt der Bot sie z.B. fuer braune Maschen unter orangen).
    Erkennung: Zelle darunter leer, Zelle darueber gefuellt, gleicher Farbton
    wie darueber aber dunkler, und kaum Struktur (keine Schlaufen)."""
    ny, nx = filled.shape
    for c in range(nx):
        for r in range(ny - 1, 0, -1):
            if not filled[r, c] or not filled[r - 1, c]:
                continue
            if r + 1 < ny and filled[r + 1, c]:
                continue
            up, me = grid_lab[r - 1, c], grid_lab[r, c]
            hue = float(np.hypot(me[1] - up[1], me[2] - up[2]))
            smooth = tex[r, c] < 0.45 * max(tex[r - 1, c], 1e-3) or tex[r, c] < 2.0
            if smooth and hue < 28 and me[0] <= up[0] + 8:
                filled[r, c] = False
                grid_lab[r, c] = 0


# ---------------------------------------------------------------------------
# Farbpalette
# ---------------------------------------------------------------------------
def _name_for(lab):
    L, a, b = float(lab[0]) * 100 / 255, float(lab[1]) - 128, float(lab[2]) - 128
    chroma = (a * a + b * b) ** 0.5
    if L < 45 and chroma < 20:
        return "schwarz"
    if chroma < 14:
        return "weiss" if L > 88 else "grau"
    hue = (np.degrees(np.arctan2(b, a)) + 360) % 360
    if L < 60 and 30 < hue < 90 and chroma < 45:
        return "braun"
    if chroma < 30 and 40 < hue < 100:
        return "beige"
    # Farbwinkel in OpenCV-Lab (nachgemessen: Rot ~40, Orange ~59, Gelb ~85)
    names = [(22, "pink"), (50, "rot"), (72, "orange"), (105, "gelb"), (170, "gruen"),
             (230, "tuerkis"), (290, "blau"), (330, "lila"), (360, "pink")]
    for lim, nm in names:
        if hue < lim:
            return nm
    return "rot"


def color_distance(a, b):
    """Farbabstand. Bei bunten Farben zaehlt fast nur der Farbton (a, b) -
    Schattierung aendert vor allem die Helligkeit. Bei Weiss/Grau/Schwarz
    entscheidet die Helligkeit."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = a - b
    ca = float(np.hypot(a[1] - 128, a[2] - 128))
    cb = float(np.hypot(b[1] - 128, b[2] - 128))
    wl = 0.2 if min(ca, cb) > 22 else 0.5
    return float(np.sqrt((wl * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))


def build_palette(labs, thr=16.0, merge=20.0, min_size=3):
    """Gruppiert die Lab-Farben der Maschen zu einer Palette.
    Rueckgabe: (Zentren, Zuordnung je Masche)."""
    labs = np.asarray(labs, np.float32)
    if len(labs) == 0:
        return [], np.zeros(0, int)
    # 1) grob vorgruppieren (gerundete Farben, haeufigste zuerst)
    q = np.round(labs / 6.0)
    keys, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(-cnt)
    centers, weights = [], []
    for k in order:
        c = labs[inv == k].mean(axis=0)
        w = int(cnt[k])
        if centers:
            d = [color_distance(c, x) for x in centers]
            j = int(np.argmin(d))
            if d[j] < thr:
                tot = weights[j] + w
                centers[j] = (centers[j] * weights[j] + c * w) / tot
                weights[j] = tot
                continue
        centers.append(c)
        weights.append(w)
    # 2) aehnliche Zentren zusammenlegen
    changed = True
    while changed and len(centers) > 1:
        changed = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if color_distance(centers[i], centers[j]) < merge:
                    tot = weights[i] + weights[j]
                    centers[i] = (centers[i] * weights[i] + centers[j] * weights[j]) / tot
                    weights[i] = tot
                    del centers[j], weights[j]
                    changed = True
                    break
            if changed:
                break
    # 3) Zuordnung, Mini-Gruppen (Schatten/Fehler) auf naechste grosse Gruppe
    C = np.array(centers)
    assign = np.array([int(np.argmin([color_distance(l, c) for c in C])) for l in labs])
    sizes = np.bincount(assign, minlength=len(C))
    keep = [i for i in range(len(C)) if sizes[i] >= min_size]
    if keep and len(keep) < len(C):
        C2 = C[keep]
        d2 = np.array([[color_distance(l, c) for c in C2] for l in labs])
        new = np.argmin(d2, axis=1)
        small = ~np.isin(assign, keep)
        far = d2[np.arange(len(labs)), new] > 45
        # weit entfernte Einzelmaschen behalten ihre eigene Farbe
        remap = {old: i for i, old in enumerate(keep)}
        out = np.array([remap.get(a, -1) for a in assign])
        extra = []
        for idx in np.where(small)[0]:
            if far[idx]:
                key = int(assign[idx])
                if key not in remap:
                    remap[key] = len(keep) + len(extra)
                    extra.append(C[key])
                out[idx] = remap[key]
            else:
                out[idx] = new[idx]
        centers = list(C2) + extra
        assign = out
    else:
        centers = list(C)
    # Zentren aus den finalen Zuordnungen neu berechnen
    centers = [labs[assign == i].mean(axis=0) if np.any(assign == i) else c for i, c in enumerate(centers)]
    return centers, assign


def _merge_edge_shades(grid, centers, labs, cells, max_dist=50.0, protect_labs=()):
    """Maschen am unteren/rechten Rand zeigen ihre dunklere 3D-Seite. Bildet
    eine Farbgruppe fast nur aus solchen Randmaschen, gehoert sie zur
    naechstliegenden Hauptfarbe."""
    ny, nx = grid.shape
    centers = list(centers)
    changed = True
    protect = set()
    while changed and len(centers) > 1:
        changed = False
        # Farben, die eine sichtbare Spule hat, sind echt -> nie verschmelzen
        protect = set()
        for sl in protect_labs:
            dd = [color_distance(sl, c) for c in centers]
            if dd and min(dd) < 30:
                protect.add(int(np.argmin(dd)))
        for k in range(len(centers)):
            idx = np.argwhere(grid == k)
            if len(idx) == 0:
                continue
            edge = 0
            for r, c in idx:
                below = r + 1 >= ny or grid[r + 1, c] < 0
                right = c + 1 >= nx or grid[r, c + 1] < 0
                if below or right:
                    edge += 1
            if k in protect:
                continue
            # dunkelgraue Varianten (Glanzlichter auf schwarzen Maschen)
            dark = (35 < float(centers[k][0]) < 130
                    and float(np.hypot(centers[k][1] - 128, centers[k][2] - 128)) < 15)
            if edge / len(idx) < 0.8 and not (dark and len(idx) < 25):
                continue
            others = [(color_distance(centers[k], centers[j]), j) for j in range(len(centers)) if j != k]
            if dark and edge / len(idx) < 0.8:
                others = [(d_, j) for d_, j in others if float(centers[j][0]) < 40]
                if not others:
                    continue
            dist, j = min(others)
            limit = max_dist if edge / len(idx) >= 0.8 else 60.0
            if dist > limit:
                continue
            # k in j aufgehen lassen, Indizes nachruecken
            grid[grid == k] = j
            grid[grid > k] -= 1
            del centers[k]
            changed = True
            break
    return grid, centers


# ---------------------------------------------------------------------------
# Spulen
# ---------------------------------------------------------------------------
def _is_surprise(disc):
    hsv = cv2.cvtColor(disc, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 70).mean()
    vivid = ((hsv[..., 1] > 120) & (hsv[..., 2] > 120)).mean()
    return dark > 0.35 and vivid > 0.03


SPOOL_L_OFFSET = 18.0   # Spulen sind dunkler schattiert als die Maschen


def _spool_lab(body_bgr):
    lab = _lab(body_bgr.reshape(-1, 3))
    L = lab[:, 0]
    if float(np.median(L)) < 45:          # schwarze Spule
        return tuple(float(v) for v in BLACK_LAB)
    sel = lab[L >= np.percentile(L, 75)]
    m = np.median(sel, axis=0)
    m[0] = min(255.0, m[0] + SPOOL_L_OFFSET)
    return tuple(float(v) for v in m)


def _disc_side(img, belt, cx, disc_y):
    """Pixel links und rechts neben der Zahl auf der Spulenscheibe."""
    s = belt.s
    parts = []
    for sgn in (-1, 1):
        a, b = sorted((int(cx + sgn * 42 * s), int(cx + sgn * 56 * s)))
        parts.append(img[int(disc_y - 8 * s):int(disc_y + 8 * s), a:b].reshape(-1, 3))
    return np.vstack(parts)


def _spool_at(img, belt, cx, disc_y, body_y, bg, where, occl_y=None):
    s = belt.s
    H, W = img.shape[:2]
    bx0, bx1 = int(cx - 28 * s), int(cx + 28 * s)
    by0, by1 = int(body_y - 22 * s), int(body_y + 22 * s)
    occluded = occl_y is not None and by1 > occl_y
    if occluded:
        # Koerper hinter den Booster-Knoepfen: Farbe von der Scheibe nehmen
        body = _disc_side(img, belt, cx, disc_y).astype(np.int16)
    else:
        body = img[max(0, by0):min(H, by1), max(0, bx0):min(W, bx1)].astype(np.int16)
    if body.size == 0:
        return None
    d = np.abs(body.reshape(-1, 3) - bg).sum(axis=1)
    disc = _disc(img, belt, cx, disc_y)
    if disc is None:
        return None
    num, conf = read_number(disc)
    if (d > 60).mean() < 0.45:
        # graue Spulen sehen dem Hintergrund aehnlich - dann zaehlt die lesbare Zahl
        if num is None or conf < 0.4 or (d > 25).mean() < 0.6:
            return None
    if (num is None or conf < 0.8) and _is_surprise(disc):
        return Spool(UNKNOWN, None, int(cx), int(disc_y + 20 * s), None, where)
    if occluded:
        lab3 = np.median(_lab(body.astype(np.uint8)), axis=0)
        lab = tuple(float(v) for v in (BLACK_LAB if lab3[0] < 50 else lab3))
    else:
        lab = _spool_lab(body.astype(np.uint8))
    if num is not None and conf < 0.25:
        num = None
    return Spool(None, num, int(cx), int(disc_y + 20 * s), lab, where)


def _bottom_bar_any(img):
    """Booster-Leiste in beliebiger Farbe (je nach Level blau, pink, ...):
    erste Zeilen im unteren Fuenftel, die ueber die ganze Breite gleichfarbig
    und kraeftig gefaerbt sind."""
    H = img.shape[0]
    y0 = int(H * 0.8)
    reg = img[y0:].astype(np.int16)
    hsv = cv2.cvtColor(img[y0:], cv2.COLOR_BGR2HSV)
    med = np.median(reg, axis=1)
    close = np.abs(reg - med[:, None, :]).sum(axis=2) < 45
    frac = (close & (hsv[..., 1] > 90)).mean(axis=1)
    run = 0
    for i, g in enumerate(frac > 0.55):
        run = run + 1 if g else 0
        if run >= 4:
            return y0 + i - 3
    return H


def read_spools(img, belt, bg):
    s = belt.s
    H, W = img.shape[:2]
    # Unterkante: blaue Booster-Leiste
    hsv = cv2.cvtColor(img[int(H * 0.8):], cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] >= 100) & (hsv[..., 0] <= 125) & (hsv[..., 1] > 150) & (hsv[..., 2] > 110)).mean(axis=1)
    bar = np.where(blue > 0.7)[0]
    bottom = int(H * 0.8) + int(bar[0]) if len(bar) else _bottom_bar_any(img)

    # Ablage
    rack = []
    for i in range(5):
        cx = belt.X(REF["rack_x0"] + REF["rack_dx"] * i)
        sp = _spool_at(img, belt, cx, belt.Yb(REF["rack_y"]), belt.Yb(REF["rack_body_y"]), bg, f"rack{i}")
        rack.append(sp)

    # Spalten finden: senkrechte Streifen mit Spulen
    y0 = belt.Yb(REF["spool_y"] - 50)
    y1 = min(bottom, belt.Yb(REF["spool_y"] + 2 * REF["spool_dy"]))
    band = img[y0:y1].astype(np.int16)
    # > 35: graue Spulen heben sich nur schwach vom Hintergrund ab
    colf = (np.abs(band - bg).sum(axis=2) > 35).mean(axis=0)
    runs = _runs(colf > 0.25, int(90 * s))
    centers = [int((a + b) / 2) for a, b in runs if (b - a) < 220 * s]
    # Graue Spulen heben sich kaum vom Hintergrund ab -> zusaetzlich die
    # moeglichen Spaltenplaetze (symmetrisch zur Bandmitte, halbe Abstaende)
    # pruefen: dort, wo eine lesbare Spule steht, ist auch eine Spalte.
    mid = (belt.x0 + belt.x1) / 2.0
    half = REF["spool_dx"] * s / 2.0
    dy0 = belt.Yb(REF["spool_y"])
    by0 = belt.Yb(REF["spool_body_y"])
    for m in range(-4, 5):
        cx = int(round(mid + m * half))
        if any(abs(cx - c) < 60 * s for c in centers):
            continue
        sp = _spool_at(img, belt, cx, dy0, by0, bg, "probe", occl_y=bottom - 140 * s)
        if sp is not None and sp.count is not None:
            centers.append(cx)
    centers.sort()
    columns = []
    for k, cx in enumerate(centers):
        col = []
        for row in range(3):
            dy = belt.Yb(REF["spool_y"] + REF["spool_dy"] * row)
            by = belt.Yb(REF["spool_body_y"] + REF["spool_dy"] * row)
            if by + 20 * s > bottom:
                break
            sp = _spool_at(img, belt, cx, dy, by, bg, f"col{k}", occl_y=bottom - 140 * s)
            if sp is None:
                break
            if row > 0:
                sp.where = f"col{k}+{row}"
            col.append(sp)
        columns.append(col)
    _find_links(img, belt, columns, rack, bg, bottom - 140 * belt.s)
    _find_hidden_links(img, belt, columns)
    return columns, rack, bottom


def _find_hidden_links(img, belt, columns):
    """Stab, der von einer Spule schraeg nach unten zu einer (noch) nicht
    sichtbaren Spule der Nachbarspalte fuehrt: Diese Spule ist erst antippbar,
    wenn der Partner vorne steht. Markiert sie mit einer Verbindung ohne
    sichtbaren Partner (Nummer ab 500)."""
    H, W = img.shape[:2]
    s = belt.s
    dy = REF["spool_dy"] * s
    lid = 500
    for k, col in enumerate(columns):
        for r, sp in enumerate(col):
            if sp.link is not None or sp.lab is None:
                continue
            ref = np.asarray(sp.lab, np.float32)
            found = False
            for dc in (-1, 1):
                if found or not (0 <= k + dc < len(columns)) or not columns[k + dc]:
                    continue
                tx = columns[k + dc][0].x
                for dr in (1, 2, 3):
                    if r + dr < len(columns[k + dc]):
                        continue        # Ziel sichtbar -> normale Verbindung
                    ty = sp.y + dr * dy
                    ok = True
                    for t in (0.25, 0.35, 0.45):
                        x = int(sp.x + (tx - sp.x) * t)
                        y = int(sp.y + 12 * s + (ty - sp.y) * t)
                        if not (4 <= x < W - 4 and 4 <= y < H - 4):
                            ok = False
                            break
                        p = _lab(img[y - 4:y + 5, x - 4:x + 5])
                        close = np.mean([color_distance(q, ref) < 25 for q in p])
                        if close < 0.7:
                            ok = False
                            break
                    if ok:
                        sp.link = lid
                        lid += 1
                        found = True
                        break


def _bar_at(img, bg, pts, r=4):
    """Liegt an allen Punkten ein farbiger Verbindungsstab (kein Hintergrund)?"""
    H, W = img.shape[:2]
    for x, y in pts:
        x, y = int(round(x)), int(round(y))
        if not (r <= x < W - r and r <= y < H - r):
            return False
        p = img[y - r:y + r + 1, x - r:x + r + 1].astype(np.int16)
        if float((np.abs(p - bg).sum(axis=2) > 60).mean()) < 0.75:
            return False
    return True


def _bar_colors_match(img, pts, spa, spb):
    """Ein Verbindungsstab verlaeuft farblich von Spule A nach Spule B. Ein
    fremder (schraeger) Stab, der zufaellig dazwischen kreuzt, hat andere
    Farben - nur pruefbar, wenn A und B verschieden aussehen."""
    if spa.lab is None or spb.lab is None:
        return True
    la, lb = np.asarray(spa.lab, np.float32), np.asarray(spb.lab, np.float32)
    if color_distance(la, lb) < 40:
        return True
    def patch_lab(x, y, r=3):
        x, y = int(round(x)), int(round(y))
        p = img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
        return np.median(_lab(p), axis=0)
    pa, pb = patch_lab(*pts[0]), patch_lab(*pts[-1])
    return (color_distance(pa, la) < color_distance(pa, lb) + 10 and
            color_distance(pb, lb) < color_distance(pb, la) + 10)


def _find_links(img, belt, columns, rack, bg, max_y=1e9):
    """Doppelspulen: zwei Spulen sind mit einem farbigen Stab verbunden
    (waagerecht oder schraeg zur Nachbarspalte, in der Ablage waagerecht).
    Tippt man eine an, fahren beide zusammen los."""
    s = belt.s
    off = 12 * s                    # Stab sitzt knapp unter der Zahl
    lid = 0
    for k in range(len(columns) - 1):
        a, b = columns[k], columns[k + 1]
        for ra, spa in enumerate(a):
            for rb, spb in enumerate(b):
                if abs(ra - rb) > 2 or spa.link is not None or spb.link is not None:
                    continue
                ya, yb = spa.y + off, spb.y + off
                # spool.y ist der Tipp-Punkt (Zahl) - Stab beginnt am Scheibenrand.
                # Ueber zwei Reihen ist der Stab sehr steil: weiter aussen pruefen,
                # damit ein waagerechter Stab der mittleren Reihe nicht zaehlt.
                ts = (0.35, 0.5, 0.65) if abs(ra - rb) == 2 else (0.4, 0.5, 0.6)
                pts = [(spa.x + (spb.x - spa.x) * t, ya + (yb - ya) * t) for t in ts]
                if max(p[1] for p in pts) > max_y:
                    continue            # hinter den Booster-Knoepfen nicht beurteilbar
                if _bar_at(img, bg, pts) and _bar_colors_match(img, pts, spa, spb):
                    spa.link = spb.link = lid
                    lid += 1
    for j in range(len(rack) - 1):
        a, b = rack[j], rack[j + 1]
        if a is None or b is None or a.link is not None:
            continue
        ya, yb = a.y + off, b.y + off
        pts = [(a.x + (b.x - a.x) * t, ya + (yb - ya) * t) for t in (0.45, 0.5, 0.55)]
        if _bar_at(img, bg, pts):
            a.link = b.link = lid
            lid += 1


# ---------------------------------------------------------------------------
# Gesamtanalyse
# ---------------------------------------------------------------------------
KNOT = 64      # Gitterwert = farbe + KNOT * (zusaetzliche Faeden); siehe yarn_solver


def find_knots(img, lat: Lattice, belt: Belt):
    """Fadenknoten: Bloecke im Bild mit einer weissen Zahl (dunkel umrandet).
    Rueckgabe: [(zahl, (x0, y0, x1, y1) der Ziffern, lab der Knotenfarbe)]."""
    X0, Y0, X1, Y1 = picture_region(belt)
    reg = img[Y0:Y1, X0:X1]
    if reg.size == 0:
        return []
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 230) & (hsv[..., 1] < 45)).astype(np.uint8)
    L = cv2.cvtColor(reg, cv2.COLOR_BGR2LAB)[..., 0].astype(np.int16)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white)
    p = min(lat.px, lat.py)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (1.0 * p <= h <= 1.8 * p and w <= 0.95 * h and a >= 8):
            continue
        # Ziffern sind dunkel umrandet, weisse Maschen nicht
        y0, y1 = max(0, y - 2), min(reg.shape[0], y + h + 2)
        x0, x1 = max(0, x - 2), min(reg.shape[1], x + w + 2)
        ring = L[y0:y1, x0:x1][lab[y0:y1, x0:x1] != i]
        if ring.size == 0 or float(np.percentile(ring, 10)) > 115:
            continue
        if a > 0.72 * w * h:          # volle weisse Flaeche = Masche, keine Ziffer
            continue
        comps.append((x, y, w, h, i))
    comps.sort()
    used, out = set(), []
    for k, c in enumerate(comps):
        if k in used:
            continue
        grp = [c]
        for k2 in range(k + 1, len(comps)):
            d = comps[k2]
            last = grp[-1]
            if (k2 not in used and abs((d[1] + d[3] / 2) - (last[1] + last[3] / 2)) < 0.3 * last[3]
                    and 0 <= d[0] - (last[0] + last[2]) < 0.6 * last[3] and len(grp) < 2):
                grp.append(d)
                used.add(k2)
        val, conf = _read_digits(lab, grp)
        if val is None or not (1 <= val <= 99) or conf < 0.25:
            continue
        gx0 = min(g[0] for g in grp)
        gy0 = min(g[1] for g in grp)
        gx1 = max(g[0] + g[2] for g in grp)
        gy1 = max(g[1] + g[3] for g in grp)
        # nur innerhalb des Maschengitters (Zahlen an Band-Spulen ausschliessen)
        mx, my = (gx0 + gx1) / 2 + X0, (gy0 + gy1) / 2 + Y0
        if not (lat.x0 + 0.3 * lat.px <= mx <= lat.x0 + (lat.nx - 0.3) * lat.px
                and lat.y0 + 0.3 * lat.py <= my <= lat.y0 + (lat.ny - 0.3) * lat.py):
            continue
        # Knotenfarbe: Ring um die Zahl, ohne Weiss und Umrandung
        pad = int(0.45 * p) + 2
        ry0, ry1 = max(0, gy0 - pad), min(reg.shape[0], gy1 + pad)
        rx0, rx1 = max(0, gx0 - pad), min(reg.shape[1], gx1 + pad)
        patch = reg[ry0:ry1, rx0:rx1].reshape(-1, 3)
        hp = hsv[ry0:ry1, rx0:rx1].reshape(-1, 3)
        Lp = L[ry0:ry1, rx0:rx1].reshape(-1)
        keep = (Lp >= 90) & ~((hp[:, 2] > 230) & (hp[:, 1] < 45))
        if keep.sum() < 10:
            continue
        col = _lab(np.median(patch[keep], axis=0).astype(np.uint8))[0]
        out.append((val, (gx0 + X0, gy0 + Y0, gx1 + X0, gy1 + Y0), col))
    return out


def apply_knots(grid, lat: Lattice, knots, centers):
    """Knoten ins Gitter eintragen. Ein Knoten ist ein Balken/Block in seiner
    Farbe; die Zahl steht nur in der Mitte. Alle zusammenhaengenden Zellen
    dieser Farbe rund um die Zahl (hoechstens 5 Zellen weit) gehoeren zum
    Knoten; die Zahl wird auf sie verteilt (Wert = farbe + KNOT * (n - 1))."""
    if grid is None or not knots or not centers:
        return grid
    ny, nx = grid.shape
    owner = -np.ones((ny, nx), np.int32)
    dist = np.full((ny, nx), 99, np.int32)
    colors, frontier = [], []
    for j, (val, (x0, y0, x1, y1), lab) in enumerate(knots):
        d = [color_distance(lab, c) for c in centers]
        k = int(np.argmin(d))
        colors.append(k)
        mx, my = 0.35 * lat.px, 0.35 * lat.py
        for r in range(ny):
            for c in range(nx):
                cx, cy = lat.center(r, c)
                if x0 - mx <= cx <= x1 + mx and y0 - my <= cy <= y1 + my and dist[r, c] > 0:
                    owner[r, c], dist[r, c] = j, 0
                    frontier.append((r, c))
    # Mehrquellen-Breitensuche ueber Zellen in Knotenfarbe
    step = 0
    while frontier and step < 5:
        step += 1
        nxt = []
        for r, c in frontier:
            j = owner[r, c]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx and dist[rr, cc] > step and grid[rr, cc] == colors[j]:
                    owner[rr, cc], dist[rr, cc] = j, step
                    nxt.append((rr, cc))
        frontier = nxt
    for j, (val, _, _) in enumerate(knots):
        cells = list(zip(*np.nonzero(owner == j)))
        if not cells:
            continue
        base, extra = divmod(val, len(cells))
        for i, (r, c) in enumerate(cells):
            m = base + (1 if i < extra else 0)
            grid[r, c] = -1 if m <= 0 else colors[j] + KNOT * (min(m, 30) - 1)
    return grid


def analyze(img, lattice: Optional[Lattice] = None, palette_lock=None) -> YarnScreen:
    """lattice / palette_lock: zu Levelbeginn eingerastetes Raster und Farben."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    belt = find_belt(img, hsv)
    bright = float(hsv[..., 2].mean())
    try:
        import vision as V
        buttons = V.find_close_buttons(img, hsv) + V.find_green_buttons(img, hsv)
    except Exception:
        buttons = []
    if belt is None:
        kind = "popup" if buttons else ("dim" if bright < 60 else "unknown")
        return YarnScreen(kind, buttons=buttons, info={"brightness": bright})
    # Abgedunkeltes Spiel = Popup oder Tutorial darueber
    if bright < 60:
        return YarnScreen("popup" if buttons else "dim", belt=belt, buttons=buttons,
                          info={"brightness": bright})

    lat = lattice or detect_lattice(img, belt)
    scr = YarnScreen("game", belt=belt, lattice=lat, info={"brightness": bright})
    X0, Y0, X1, Y1 = picture_region(belt)
    bg = _bg_color(img[Y0:Y1, X0:X1].astype(np.int16))
    columns, rack, bottom = read_spools(img, belt, bg)
    scr.columns, scr.rack = columns, rack
    scr.info["bottom"] = bottom
    spool_labs = [np.array(sp.lab, np.float32) for col in columns for sp in col[:1]
                  if sp is not None and sp.lab is not None]
    spool_labs += [np.array(sp.lab, np.float32) for sp in rack if sp is not None and sp.lab is not None]

    labs, cells = [], []
    if lat is not None:
        grid_lab, filled, _ = read_grid(img, lat, belt)
        for r in range(lat.ny):
            for c in range(lat.nx):
                if filled[r, c]:
                    labs.append(grid_lab[r, c])
                    cells.append((r, c))
    grid = np.full((lat.ny, lat.nx), -1, np.int16) if lat is not None else None
    if palette_lock is not None and len(palette_lock):
        centers = [np.asarray(c, np.float32) for c in palette_lock]
        for (r, c), lab in zip(cells, labs):
            d = [color_distance(lab, cc) for cc in centers]
            k = int(np.argmin(d))
            if d[k] > 55:            # unbekannte Farbe -> neue Gruppe
                centers.append(np.asarray(lab, np.float32))
                k = len(centers) - 1
            grid[r, c] = k
    else:
        centers, assign = build_palette(np.array(labs, np.float32)) if labs else ([], [])
        for (r, c), k in zip(cells, assign):
            grid[r, c] = k
        if grid is not None and len(centers) > 1:
            grid, centers = _merge_edge_shades(grid, centers, labs, cells, protect_labs=spool_labs)
    if lat is not None and grid is not None:
        try:
            knots = find_knots(img, lat, belt)
        except Exception:
            knots = []
        if knots:
            grid = apply_knots(grid, lat, knots, centers)
            scr.info["knots"] = [(v, int(b[0]), int(b[1])) for v, b, _ in knots]
    palette = [(_name_for(c), c) for c in centers]

    # Spulenfarbe der naechsten Maschenfarbe zuordnen
    def match(sp):
        if sp is None or sp.color == UNKNOWN or sp.lab is None:
            return
        lab = np.array(sp.lab, np.float32)
        if palette:
            dists = [color_distance(lab, c) for _, c in palette]
            k = int(np.argmin(dists))
            sp.color = str(k) if dists[k] < 32 else "x" + _name_for(lab)
        else:
            sp.color = "x" + _name_for(lab)

    for col in columns:
        for sp in col:
            match(sp)
    for sp in rack:
        match(sp)
    scr.grid = grid
    scr.palette = palette
    return scr


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
def describe(scr: YarnScreen):
    lines = [f"Bildschirm: {scr.kind}"]
    if scr.kind != "game":
        return "\n".join(lines)
    pal = {str(i): n for i, (n, _) in enumerate(scr.palette)}
    if scr.lattice is not None:
        L = scr.lattice
        lines.append(f"Raster {L.nx}x{L.ny}, Zelle {L.px:.1f}x{L.py:.1f}")
        cnt = {}
        for v in scr.grid.ravel():
            if v >= 0:
                cnt[int(v)] = cnt.get(int(v), 0) + 1
        lines.append("Maschen: " + ", ".join(f"{pal[str(k)]}#{k}={n}" for k, n in sorted(cnt.items())))
    def sp_s(sp):
        if sp is None:
            return "--"
        return f"{pal.get(sp.color, sp.color)}:{sp.count}"
    lines.append("Ablage: " + " | ".join(sp_s(s) for s in scr.rack))
    for i, col in enumerate(scr.columns):
        lines.append(f"Spalte {i}: " + " , ".join(sp_s(s) for s in col))
    return "\n".join(lines)


def draw_debug(img, scr: YarnScreen):
    out = img.copy()
    if scr.belt:
        b = scr.belt
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (0, 255, 255), 3)
    if scr.lattice is not None and scr.grid is not None:
        L = scr.lattice
        for r in range(L.ny):
            for c in range(L.nx):
                x, y = L.center(r, c)
                k = int(scr.grid[r, c])
                if k < 0:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 0, 0), -1)
                else:
                    cv2.putText(out, str(k), (int(x - 6), int(y + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 255), 1)
    for col in scr.columns:
        for sp in col:
            cv2.circle(out, (sp.x, sp.y), 18, (0, 255, 0), 3)
            cv2.putText(out, f"{sp.color}:{sp.count}", (sp.x - 50, sp.y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
    for sp in scr.rack:
        if sp:
            cv2.circle(out, (sp.x, sp.y), 18, (255, 255, 0), 3)
            cv2.putText(out, f"{sp.color}:{sp.count}", (sp.x - 50, sp.y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 0), 2)
    cv2.putText(out, scr.kind, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return out


def find_tutorial_arrow(img):
    """Gruener Tutorial-Pfeil (z.B. "Tippe, um ein Zusatztablett ..." mit
    Pfeil auf den Gratis-Booster). Rueckgabe: Tipp-Punkt (x, y) oder None."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] > 140) & (hsv[..., 2] > 170)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.002 * W * H or not (0.5 < h / max(w, 1) < 2.2):
            continue
        if a > 0.85 * w * h or a < 0.3 * w * h:      # Knoepfe sind fast voll, Pfeile nicht
            continue
        rows = (lab[y:y + h, x:x + w] == i).sum(axis=1)
        widest = int(np.argmax(rows))
        down = widest > h * 0.45
        if best is None or a > best[0]:
            ty = y + h + 0.75 * h if down else y - 0.75 * h
            best = (a, (int(x + w / 2), int(min(H - 1, max(0, ty)))))
    return best[1] if best else None
