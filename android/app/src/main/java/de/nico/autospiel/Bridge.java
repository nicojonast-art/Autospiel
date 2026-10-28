package de.nico.autospiel;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.os.Handler;
import android.os.Looper;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Verbindung zwischen dem Python-Bot (android_main.py) und Android.
 * Alle Methoden sind statisch, damit Python sie einfach per jclass aufrufen kann.
 */
public final class Bridge {

    public static final String YARN_PACKAGE = "com.combo.yarnflow";
    public static final String CAR_PACKAGE = "com.grandgames.carmatch";

    /** Paketname zum Spiel-Kuerzel aus den Einstellungen ("yarn" / "car"). */
    static String packageFor(String game) {
        return "car".equals(game) ? CAR_PACKAGE : YARN_PACKAGE;
    }

    /**
     * Tatsaechlich installiertes Paket des Spiels. Zuerst der bekannte
     * Paketname, sonst wird unter allen startbaren Apps nach dem Namen gesucht
     * (z.B. "Yarn Loop"). Rueckgabe null, wenn nichts gefunden wurde.
     */
    static String resolvePackage(Context ctx, String game) {
        PackageManager pm = ctx.getPackageManager();
        String known = packageFor(game);
        if (pm.getLaunchIntentForPackage(known) != null) {
            return known;
        }
        String wanted = "car".equals(game) ? "car match" : "yarn loop";
        String wantedPkg = "car".equals(game) ? "carmatch" : "yarn";
        Intent main = new Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER);
        try {
            for (ResolveInfo ri : pm.queryIntentActivities(main, 0)) {
                String pkg = ri.activityInfo.packageName;
                if (pkg.equals(ctx.getPackageName())) {
                    continue;
                }
                CharSequence label = ri.loadLabel(pm);
                String l = label == null ? "" : label.toString().toLowerCase(Locale.ROOT);
                if (l.contains(wanted) || pkg.toLowerCase(Locale.ROOT).contains(wantedPkg)) {
                    return pkg;
                }
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    /** Wird von der Oberflaeche beobachtet. */
    public interface Listener {
        void onBotChanged();
    }

    private static final int MAX_LOG = 120;
    private static final List<String> LOG = new ArrayList<>();
    private static final Handler MAIN = new Handler(Looper.getMainLooper());
    private static final SimpleDateFormat TIME = new SimpleDateFormat("HH:mm:ss", Locale.GERMANY);

    static volatile boolean running = false;
    static volatile BotService service;
    static volatile int levels = 0;
    static volatile String status = "";
    static volatile Listener listener;

    private Bridge() {
    }

    // ------------------------------------------------------------ fuer Python

    public static boolean isRunning() {
        return running;
    }

    /** Aktuelles Bild: 8 Byte Kopf (Breite, Hoehe) + RGBA-Pixel. */
    public static byte[] grab() {
        BotService s = service;
        return s == null ? null : s.grabFrame();
    }

    public static boolean tap(int x, int y) {
        TapService t = TapService.instance;
        if (t == null) {
            log("Bedienungshilfe ist aus – Tippen nicht moeglich");
            return false;
        }
        return t.tap(x, y);
    }

    public static void back() {
        TapService t = TapService.instance;
        if (t != null) {
            t.back();
        }
    }

    /** "paket/klasse" des Fensters im Vordergrund. */
    public static String foreground() {
        return TapService.lastWindow;
    }

    public static void launchGame(String pkg) {
        TapService t = TapService.instance;
        if (t != null) {
            t.launchGame(pkg);
        } else if (service != null) {
            service.launchGame(pkg);
        }
    }

    public static void log(String message) {
        String line = TIME.format(new Date()) + "  " + message;
        synchronized (LOG) {
            LOG.add(line);
            while (LOG.size() > MAX_LOG) {
                LOG.remove(0);
            }
        }
        status = message;
        notifyChanged();
    }

    public static void setLevels(int n) {
        if (n != levels) {
            levels = n;
            notifyChanged();
        }
    }

    // ------------------------------------------------------------ fuer die App

    static String logText() {
        StringBuilder sb = new StringBuilder();
        synchronized (LOG) {
            for (int i = LOG.size() - 1; i >= 0; i--) {
                sb.append(LOG.get(i));
                if (i > 0) {
                    sb.append('\n');
                }
            }
        }
        return sb.toString();
    }

    static void clearLog() {
        synchronized (LOG) {
            LOG.clear();
        }
        status = "";
    }

    static void notifyChanged() {
        MAIN.post(() -> {
            Listener l = listener;
            if (l != null) {
                l.onBotChanged();
            }
            TapService t = TapService.instance;
            if (t != null) {
                t.updateOverlay();
            }
        });
    }
}
