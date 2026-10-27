# Car Match Bot (Xiaomi / Android)

> Automatischer Löser für **Car Match – Traffic Puzzle** (Grand Games) per ADB.
> Läuft auf dem PC (Windows / macOS / Linux) und steuert das Handy über USB oder WLAN.

```
git clone https://github.com/nicojonast-art/Autospiel.git
cd Autospiel
pip install -r requirements.txt
python bot.py
```

Spielt **Car Match – Traffic Puzzle** (Grand Games, `com.grandgames.carmatch`)
vollautomatisch: erkennt bei jedem Zug das Spielfeld, plant die Züge, tippt,
startet das nächste Level, schließt Popups und versucht es nach einem
verlorenen Level erneut.

## Wie er funktioniert

1. **Screenshot** per ADB (Rohdaten, schnell).
2. **Bilderkennung** (`vision.py`), komplett auflösungsunabhängig:
   - Parkleiste (7 Plätze) und deren Inhalt
   - Raster des Parkplatzes (Zellgröße/Position werden pro Level gemessen und eingerastet)
   - jedes Feld: Straße, Gehweg/Wand, Auto + Farbe, Unbekanntes
   - **offene Augen** auf der Windschutzscheibe = Auto ist antippbar (so zeigt es das Spiel selbst)
   - Tunnel/Garagen (lila Bogen) inkl. Zahl der Autos darin (mit Tesseract)
   - Booster-Knöpfe, rote X-Knöpfe, grüne Buttons, abgedunkelte Tutorials
3. **Löser** (`solver.py`): Ein Auto ist frei, wenn ein Nachbarfeld freie,
   mit der Ausfahrt verbundene Straße ist. Strahlsuche + Tiefensuche finden
   meist sofort eine **komplette Lösung** fürs ganze Level (Level 5 aus dem
   Video: 39 Autos, gelöst in 0,4 s). Unbekannte Autos (Tunnel) werden nach
   jedem Zug neu eingeplant.
4. **Steuerung** (`bot.py`): wartet nach jedem Tap, bis alles still steht,
   prüft ob der Plan noch passt, sonst neu planen. Wenn alle Züge verlieren
   würden, werden (einmal pro Level) Booster probiert.
5. **Menüs** (`screens.py`): „Level N“ antippen, Popups mit X schließen,
   „Try again / Continue / Claim …“ per Texterkennung. **Münzen werden nie
   ausgegeben** (außer mit `--allow-coins`). Keine Leben → 5 Minuten warten.
   Werbung → warten, „X/Skip/Close“ suchen, sonst Zurück-Taste.

## Einrichtung am Xiaomi (einmalig)

1. *Einstellungen → Über das Telefon* → 7× auf **OS-Version** (bzw. MIUI-Version) tippen.
2. *Einstellungen → Weitere Einstellungen → Entwickleroptionen*:
   - **USB-Debugging** → AN
   - **USB-Debugging (Sicherheitseinstellungen)** → AN
     *(wichtig! Ohne das funktionieren keine Taps. Xiaomi verlangt dafür
     meist ein angemeldetes Mi-Konto und eine eingelegte SIM.)*
3. Handy per USB an den PC, Dialog „USB-Debugging zulassen?“ bestätigen
   (Häkchen „immer zulassen“).

## Einrichtung am PC (einmalig)

1. **Python 3.10+** installieren (Windows: bei der Installation „Add to PATH“ anhaken).
2. **ADB** (Android Platform-Tools) installieren und in den PATH legen
   (oder `adb.exe` in diesen Ordner kopieren).
3. Im Ordner dieses Bots:
   ```
   pip install -r requirements.txt
   ```
4. *(Empfohlen)* **Tesseract-OCR** installieren – damit liest der Bot Texte
   auf Popups und die Zahl auf Tunneln. Ohne geht es auch, dann nur per Farbe.

## Starten

Spiel muss nicht offen sein – der Bot startet es selbst.

```
python bot.py                 # endlos spielen
python bot.py --levels 10     # 10 Level, dann Stopp
python bot.py --debug         # Erkennungsbilder in ./debug speichern
python bot.py --dry-run       # nur erkennen + planen, nicht tippen
python bot.py --no-boosters   # keine Booster benutzen
python bot.py --think 4       # mehr Rechenzeit pro Plan (Standard 2,5 s)
```

Windows: einfach `start.bat` doppelklicken.

Beenden: **Strg + C**.

### Ohne Kabel (WLAN-Debugging, Android 11+)

Entwickleroptionen → *Kabelloses Debugging* → *Gerät mit Kopplungscode koppeln*:
```
adb pair 192.168.x.x:PORT        (Kopplungscode eingeben)
adb connect 192.168.x.x:PORT2    (Port aus dem Hauptbildschirm "Kabelloses Debugging")
python bot.py --serial 192.168.x.x:PORT2
```

## Einstellungen (`config.json`)

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `think_time` | 2.5 | Sekunden Rechenzeit pro Planung |
| `after_tap` | 0.45 | Pause nach jedem Tap |
| `use_boosters` | true | Booster nur, wenn sonst verloren |
| `booster_order` | super_undo, magnet, shuffle | Reihenfolge der Booster |
| `allow_coins` | false | Münzen für „Weiterspielen“ usw. ausgeben |
| `max_levels` | 0 | 0 = endlos |
| `analysis_width` | 720 | Screenshots für die Analyse auf diese Breite verkleinern |

## Fehlersuche

- **„Kein Android-Gerät gefunden“** → `adb devices` muss das Handy als `device` zeigen.
- **Tippt nicht** → *USB-Debugging (Sicherheitseinstellungen)* fehlt.
- **Erkennt etwas falsch** → `python bot.py --debug` laufen lassen; im Ordner
  `debug/` liegen Bilder mit eingezeichneter Erkennung (Kästchen = Auto in der
  erkannten Farbe, weißer Punkt = offene Augen, × = Wand, • = Straße).
  Einzelnes Bild prüfen: `python bot.py --analyze bild.png`.
  Screenshot vom Handy holen: `python bot.py --screenshot bild.png`.
- Neue Hindernisse (Kisten, Aufzüge, …) werden als „unbekannt“ behandelt;
  der Bot verlässt sich dann auf die Augen-Anzeige des Spiels und plant nach
  jedem Zug neu. Schick mir Debug-Bilder davon, dann baue ich sie gezielt ein.
