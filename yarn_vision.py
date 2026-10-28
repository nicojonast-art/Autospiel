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
    Rueckgabe: (zahl oder None, sicherheit 0..1)."""
    hsv = cv2.cvtColor(disc_bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 238) & (hsv[..., 1] < 40)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if H * 0.33 <= h <= H * 0.8 and a >= 40 and w <= W * 0.45 and abs((y + h / 2) - H / 2) < H * 0.25:
            comps.append((x, y, w, h, i))
    if not comps or len(comps) > 3:
        return None, 0.0
    comps.sort()
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
    px = _autocorr(prof, 14 * s, 60 * s)
    nx = _best_count(prof, w, px)
    px = w / nx
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
        return "weiss" if L > 75 else "grau"
    hue = (np.degrees(np.arctan2(b, a)) + 360) % 360
    if L < 60 and 30 < hue < 90 and chroma < 45:
        return "braun"
    if chroma < 30 and 40 < hue < 100:
        return "beige"
    names = [(20, "rot"), (45, "orange"), (100, "gelb"), (170, "gruen"), (230, "tuerkis"),
             (290, "blau"), (330, "lila"), (360, "pink")]
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


def _spool_at(img, belt, cx, disc_y, body_y, bg, where):
    s = belt.s
    H, W = img.shape[:2]
    # Ist ueberhaupt eine Spule da? -> Koerper muss sich vom Hintergrund abheben
    bx0, bx1 = int(cx - 28 * s), int(cx + 28 * s)
    by0, by1 = int(body_y - 22 * s), int(body_y + 22 * s)
    body = img[max(0, by0):min(H, by1), max(0, bx0):min(W, bx1)].astype(np.int16)
    if body.size == 0:
        return None
    d = np.abs(body.reshape(-1, 3) - bg).sum(axis=1)
    if (d > 60).mean() < 0.45:
        return None
    disc = _disc(img, belt, cx, disc_y)
    if disc is None:
        return None
    num, conf = read_number(disc)
    if num is None and _is_surprise(disc):
        return Spool(UNKNOWN, None, int(cx), int(disc_y + 20 * s), None, where)
    lab = _spool_lab(body.astype(np.uint8))
    if num is not None and conf < 0.25:
        num = None
    return Spool(None, num, int(cx), int(disc_y + 20 * s), lab, where)


def read_spools(img, belt, bg):
    s = belt.s
    H, W = img.shape[:2]
    # Unterkante: blaue Booster-Leiste
    hsv = cv2.cvtColor(img[int(H * 0.8):], cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] >= 100) & (hsv[..., 0] <= 125) & (hsv[..., 1] > 150) & (hsv[..., 2] > 110)).mean(axis=1)
    bar = np.where(blue > 0.7)[0]
    bottom = int(H * 0.8) + int(bar[0]) if len(bar) else H

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
    colf = (np.abs(band - bg).sum(axis=2) > 60).mean(axis=0)
    runs = _runs(colf > 0.25, int(90 * s))
    centers = [int((a + b) / 2) for a, b in runs if (b - a) < 220 * s]
    columns = []
    for k, cx in enumerate(centers):
        col = []
        for row in range(3):
            dy = belt.Yb(REF["spool_y"] + REF["spool_dy"] * row)
            by = belt.Yb(REF["spool_body_y"] + REF["spool_dy"] * row)
            if by + 20 * s > bottom:
                break
            sp = _spool_at(img, belt, cx, dy, by, bg, f"col{k}")
            if sp is None:
                break
            if row > 0:
                sp.where = f"col{k}+{row}"
            col.append(sp)
        columns.append(col)
    return columns, rack, bottom


# ---------------------------------------------------------------------------
# Gesamtanalyse
# ---------------------------------------------------------------------------
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
