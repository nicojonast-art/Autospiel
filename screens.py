"""
Popups, Menues, Werbung: Was soll angetippt werden?

Mit Tesseract-OCR (empfohlen) werden die Texte auf Buttons gelesen und nach
Stichworten entschieden. Ohne OCR wird auf Farb-Heuristiken zurueckgegriffen
(rotes X schliessen, gruenen Button ohne Muenz-Symbol antippen).
"""
from __future__ import annotations

import re

import cv2
import numpy as np

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAVE_OCR = True
except Exception:  # pragma: no cover
    HAVE_OCR = False

# Reihenfolge = Prioritaet
POSITIVE = [
    "try again", "retry", "restart", "play again", "replay",
    "continue", "next", "claim", "collect", "tap to continue", "tap to start",
    "start", "play", "ok", "okay", "great", "awesome", "nice", "got it",
    "level",
]
# Nach einem verlorenen Level: Angebot ablehnen / aufgeben
GIVE_UP = ["give up", "no thanks", "no, thanks", "quit", "leave", "end level", "lose"]
# Niemals antippen (kostet Geld / Muenzen)
FORBIDDEN = ["buy", "purchase", "$", "€", "usd", "eur", "price", "subscribe",
             "remove ads", "shop", "store", "install", "download", "get it"]
NO_LIVES = ["no lives", "out of lives", "no more lives", "refill", "more lives", "lives full in"]


def ocr_words(img):
    """Liefert Zeilen [(text_lower, x, y, w, h)] mit Bildschirmkoordinaten."""
    if not HAVE_OCR:
        return []
    H, W = img.shape[:2]
    scale = 1000.0 / W if W > 1000 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lines = {}
    for variant in (gray, 255 - gray):
        try:
            d = pytesseract.image_to_data(variant, output_type=pytesseract.Output.DICT,
                                          config="--psm 11")
        except Exception:
            return []
        for i, txt in enumerate(d["text"]):
            t = (txt or "").strip()
            if not t or float(d["conf"][i]) < 45:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i], variant is gray)
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            if key in lines:
                t0, x0, y0, x1, y1 = lines[key]
                lines[key] = (t0 + " " + t, min(x0, x), min(y0, y), max(x1, x + w), max(y1, y + h))
            else:
                lines[key] = (t, x, y, x + w, y + h)
    out = []
    for t, x0, y0, x1, y1 in lines.values():
        out.append((t.lower(), int(x0 / scale), int(y0 / scale), int((x1 - x0) / scale), int((y1 - y0) / scale)))
    return out


def has_coin_icon(img, hsv, x, y, w, h):
    """Goldene Muenze im Button -> kostet Muenzen. Erkennt einen runden,
    goldenen Fleck (Muenzsymbol neben dem Preis)."""
    H, W = img.shape[:2]
    xa, xb = max(0, x - w // 2), min(W, x + w // 2)
    ya, yb = max(0, y - h // 2), min(H, y + h // 2)
    p = hsv[ya:yb, xa:xb]
    if p.size == 0:
        return False
    gold = ((p[..., 0] >= 12) & (p[..., 0] <= 32) & (p[..., 1] > 130) & (p[..., 2] > 170)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(gold)
    for i in range(1, n):
        bx, by, bw, bh, a = st[i]
        if a > (0.15 * (yb - ya)) ** 2 and 0.6 < bw / max(bh, 1) < 1.6:
            return True
    return False


def contains(text, words):
    return any(w in text for w in words)


def decide(img, scr, allow_coins=False):
    """Entscheidung fuer Nicht-Spiel-Bildschirme.
    Rueckgabe: ("tap", x, y, grund) | ("wait", sekunden, grund) | ("back", grund) | None"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    words = ocr_words(img)
    alltext = " | ".join(w[0] for w in words)

    if words and contains(alltext, NO_LIVES):
        # Leben leer: X schliessen falls vorhanden, dann warten
        for (x, y, w, h, lab) in scr.buttons:
            if lab == "close":
                return ("tap", x, y, "Keine Leben - Popup schliessen und warten")
        return ("wait", 300, "Keine Leben mehr - warte 5 Minuten")

    if words:
        # 1) Aufgeben/Ablehnen nach verlorenem Level (statt Muenzen auszugeben)
        if not allow_coins:
            for (t, x, y, w, h) in words:
                if contains(t, GIVE_UP) and not contains(t, FORBIDDEN):
                    return ("tap", x + w // 2, y + h // 2, f"'{t}'")
        # 2) Positive Buttons
        for key in POSITIVE:
            for (t, x, y, w, h) in words:
                if re.search(r"\b" + re.escape(key) + r"\b", t) and not contains(t, FORBIDDEN):
                    cx, cy = x + w // 2, y + h // 2
                    if not allow_coins and has_coin_icon(img, hsv, cx, cy, max(w, 200), max(h * 2, 80)):
                        continue
                    if key == "level" and not re.search(r"level\s*\d+", t):
                        continue
                    return ("tap", cx, cy, f"'{t}'")

    # 3) Rotes X
    for (x, y, w, h, lab) in scr.buttons:
        if lab == "close":
            return ("tap", x, y, "Popup schliessen (X)")
    # 4) Gruener Button ohne Muenzen
    for (x, y, w, h, lab) in scr.buttons:
        if lab == "green":
            if not allow_coins and has_coin_icon(img, hsv, x, y, w, h):
                continue
            return ("tap", x, y, "gruener Button")
    return None
