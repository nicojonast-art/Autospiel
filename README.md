# 🧶 Yarn Loop Bot

Spielt **Yarn Loop** (Combo Game) auf deinem Handy automatisch – Level für Level.

## ⬇️ APK herunterladen

### 👉 [**YarnLoop-Bot.apk herunterladen**](https://github.com/nicojonast-art/Autospiel/releases/latest/download/YarnLoop-Bot.apk)

Diesen Link **auf dem Handy** öffnen → Datei öffnen → **Installieren**.
Beim ersten Mal fragt Android, ob der Browser Apps installieren darf → erlauben.
Alle Versionen: [Releases](https://github.com/nicojonast-art/Autospiel/releases) · Neue Versionen einfach über die alte installieren.

## 📱 Einrichten und spielen

### Einrichten (einmalig, die App zeigt alles mit ✓ an)
1. **Bedienungshilfe**: In der App auf „Einschalten“ → *Installierte Apps* → **Yarn Loop Bot** → an.
   *Xiaomi/Android 13+: Ist der Schalter ausgegraut?* App-Info von „Yarn Loop Bot“ öffnen →
   oben rechts **⋮** → **„Eingeschränkte Einstellungen zulassen“** → danach geht's.
2. **Spiel installiert**: Yarn Loop aus dem Play Store.
3. **Akku: keine Einschränkung** – sonst beendet Xiaomi den Bot im Hintergrund.
4. **Benachrichtigung** erlauben (zeigt Stopp-Knopf).

### Spielen
1. Oben ist **Yarn Loop** ausgewählt → **Start** tippen.
2. Android fragt nach der Bildschirmaufnahme → **„Gesamten Bildschirm“** wählen → **Starten**.
3. Die App öffnet das Spiel und spielt los. Oben links schwebt eine kleine Pille
   „Bot · 3 Level · Stopp“ – antippen = stoppen. Stoppen geht auch über die Benachrichtigung.

## 🧠 So spielt der Bot Yarn Loop

Regeln (aus einer Bildschirmaufnahme Bild für Bild nachgeprüft):

- Eine angetippte Spule fährt **einmal** ums Bild: unten → rechts hoch → oben → links runter.
- An jeder Spalte/Zeile nimmt sie die **äußerste Masche**, wenn die Farbe passt – höchstens eine pro Spalte/Zeile.
- Jede Masche kostet 1 von der Zahl. Rest → **Ablage** (5 Plätze). Ablage voll → „Kein Platz mehr!“.
- **Doppelspulen** (mit einem Stab verbunden, ab Level 11): fahren nur zusammen los – beide müssen vorne stehen.

Der Bot

1. wartet, bis das Förderband still steht,
2. liest Maschenraster, Farben, Spulen, Zahlen (eigene Ziffernerkennung) und Ablage,
3. spielt die Reihenfolgen der Spalten im Kopf durch – nach jeder Spule werden sofort alle
   Ablage-Spulen hinterhergeschickt, die etwas nehmen können (Ablage so leer wie möglich),
4. wählt die Reihenfolge, bei der die Ablage nie überläuft (Test mit lösbaren Zufallsleveln auf echten
   Bildern: 78 von 80 gewonnen),
5. tippt, wartet, liest neu – bis das Bild leer ist, dann nächstes Level.

**Sicherheit:** Fenster mit Münz- oder Geld-Angeboten („Spiel weiter 900“, „4,99 EUR“) werden
immer mit X / Zurück geschlossen. Nach einem Verlust tippt der Bot „Versuche es noch einmal“.

## 💻 Am PC (MuMu Player oder Handy per USB)

1. **`start.bat`** doppelklicken (installiert Python bei Bedarf automatisch).
2. Der Bot findet MuMus `adb.exe` selbst und verbindet sich mit dem Emulator.

```
python bot.py                # Yarn Loop (Standard)
python bot.py --spiel car    # Car Match (zweites Spiel, ebenfalls eingebaut)
python bot.py --dry-run      # nur anzeigen, nicht tippen
python bot.py --analyze bild.png   # einen Screenshot auswerten
```

Handy per USB: Entwickleroptionen → **USB-Debugging** und (Xiaomi) **USB-Debugging (Sicherheitseinstellungen)** einschalten.

## 🔧 Für Entwickler

| Datei | Inhalt |
|---|---|
| `yarn_vision.py` | Erkennung Yarn Loop: Band, Maschenraster, Farbpalette, Spulen, Zahlen |
| `yarn_digits.py` | Ziffernvorlagen (aus echten Screenshots gelernt) |
| `yarn_solver.py` | Runden-Simulation und Suche |
| `yarn_bot.py` | Ablauf, Popups, Levelwechsel |
| `vision.py`, `solver.py`, `bot.py` | Car Match (und PC-Startprogramm) |
| `device.py` | ADB (PC) |
| `android/` | Android-App (Java + Chaquopy-Python) |
| `.github/workflows/apk.yml` | baut bei jedem Push automatisch die APK und veröffentlicht sie unter *Releases* |
