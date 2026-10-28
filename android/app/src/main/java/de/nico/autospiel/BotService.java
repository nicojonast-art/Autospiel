package de.nico.autospiel;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.PixelFormat;
import android.graphics.drawable.Icon;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.view.Display;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.nio.ByteBuffer;

/**
 * Vordergrunddienst: nimmt den Bildschirm auf und laesst den Python-Bot laufen.
 */
public class BotService extends Service {

    static final String ACTION_START = "de.nico.autospiel.START";
    static final String ACTION_STOP = "de.nico.autospiel.STOP";
    static final String EXTRA_CODE = "code";
    static final String EXTRA_DATA = "data";
    static final String EXTRA_CONFIG = "config";

    private static final String CHANNEL = "bot";
    private static final int NOTIFICATION_ID = 7;

    private final Handler main = new Handler(Looper.getMainLooper());
    private final Object frameLock = new Object();

    private MediaProjection projection;
    private VirtualDisplay display;
    private ImageReader reader;
    private HandlerThread imageThread;
    private Image latest;
    private byte[] frameBuffer;
    private Thread botThread;
    private boolean stopping = false;

    static void requestStop(Context ctx) {
        Intent stop = new Intent(ctx, BotService.class).setAction(ACTION_STOP);
        try {
            ctx.startService(stop);
        } catch (Exception e) {
            Bridge.running = false;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if (ACTION_STOP.equals(intent.getAction())) {
            Bridge.log("Gestoppt");
            shutdown();
            return START_NOT_STICKY;
        }
        if (Bridge.running) {
            return START_NOT_STICKY;
        }
        if (botThread != null && botThread.isAlive()) {
            Bridge.log("Der Bot beendet sich noch – bitte gleich nochmal starten");
            return START_NOT_STICKY;
        }
        stopping = false;

        startInForeground();

        int code = intent.getIntExtra(EXTRA_CODE, 0);
        Intent data;
        if (Build.VERSION.SDK_INT >= 33) {
            data = intent.getParcelableExtra(EXTRA_DATA, Intent.class);
        } else {
            data = intent.getParcelableExtra(EXTRA_DATA);
        }
        String config = intent.getStringExtra(EXTRA_CONFIG);
        if (config == null) {
            config = "{}";
        }

        if (data == null || !startCapture(code, data)) {
            Bridge.log("Bildschirmaufnahme nicht erlaubt");
            shutdown();
            return START_NOT_STICKY;
        }

        Bridge.service = this;
        Bridge.running = true;
        Bridge.levels = 0;
        Bridge.notifyChanged();
        TapService t = TapService.instance;
        if (t != null) {
            t.showOverlay();
        }
        startBot(config);
        return START_NOT_STICKY;
    }

    // ---------------------------------------------------------------- Benachrichtigung

    private void startInForeground() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        NotificationChannel ch = new NotificationChannel(CHANNEL,
                getString(R.string.channel_name), NotificationManager.IMPORTANCE_LOW);
        nm.createNotificationChannel(ch);

        PendingIntent open = PendingIntent.getActivity(this, 1,
                new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        PendingIntent stop = PendingIntent.getService(this, 2,
                new Intent(this, BotService.class).setAction(ACTION_STOP),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);

        Notification n = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_stat_car)
                .setContentTitle(getString(R.string.notif_title))
                .setContentText(getString(R.string.notif_text))
                .setContentIntent(open)
                .setOngoing(true)
                .addAction(new Notification.Action.Builder(
                        Icon.createWithResource(this, R.drawable.ic_stop), getString(R.string.stop), stop).build())
                .build();

        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, n);
        }
    }

    // ---------------------------------------------------------------- Bildschirmaufnahme

    private boolean startCapture(int code, Intent data) {
        try {
            MediaProjectionManager mpm = getSystemService(MediaProjectionManager.class);
            projection = mpm.getMediaProjection(code, data);
            if (projection == null) {
                return false;
            }
            imageThread = new HandlerThread("bildschirm");
            imageThread.start();
            Handler imageHandler = new Handler(imageThread.getLooper());

            projection.registerCallback(new MediaProjection.Callback() {
                @Override
                public void onStop() {
                    if (!stopping) {
                        Bridge.log("Bildschirmaufnahme wurde vom System beendet");
                        main.post(BotService.this::shutdown);
                    }
                }
            }, imageHandler);

            DisplayMetrics dm = new DisplayMetrics();
            DisplayManager dmgr = getSystemService(DisplayManager.class);
            Display screen = dmgr.getDisplay(Display.DEFAULT_DISPLAY);
            screen.getRealMetrics(dm);
            int w = dm.widthPixels;
            int h = dm.heightPixels;

            reader = ImageReader.newInstance(w, h, PixelFormat.RGBA_8888, 3);
            reader.setOnImageAvailableListener(r -> {
                Image img;
                try {
                    img = r.acquireLatestImage();
                } catch (Exception e) {
                    return;
                }
                if (img == null) {
                    return;
                }
                synchronized (frameLock) {
                    if (latest != null) {
                        latest.close();
                    }
                    latest = img;
                    frameLock.notifyAll();
                }
            }, imageHandler);

            display = projection.createVirtualDisplay("Autospiel", w, h, dm.densityDpi,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                    reader.getSurface(), null, imageHandler);
            Bridge.log("Bildschirmaufnahme laeuft (" + w + "x" + h + ")");
            return true;
        } catch (Exception e) {
            Bridge.log("Fehler bei der Bildschirmaufnahme: " + e.getMessage());
            return false;
        }
    }

    /** Neuestes Bild kopieren: 8 Byte Kopf (Breite, Hoehe, big-endian) + RGBA. */
    byte[] grabFrame() {
        synchronized (frameLock) {
            long until = System.currentTimeMillis() + 3000;
            while (latest == null && System.currentTimeMillis() < until && Bridge.running) {
                try {
                    frameLock.wait(200);
                } catch (InterruptedException e) {
                    return null;
                }
            }
            if (latest == null) {
                return null;
            }
            try {
                int w = latest.getWidth();
                int h = latest.getHeight();
                Image.Plane plane = latest.getPlanes()[0];
                ByteBuffer buf = plane.getBuffer();
                int rowStride = plane.getRowStride();
                int rowBytes = w * 4;
                int size = 8 + rowBytes * h;
                if (frameBuffer == null || frameBuffer.length != size) {
                    frameBuffer = new byte[size];
                }
                byte[] out = frameBuffer;
                out[0] = (byte) (w >>> 24);
                out[1] = (byte) (w >>> 16);
                out[2] = (byte) (w >>> 8);
                out[3] = (byte) w;
                out[4] = (byte) (h >>> 24);
                out[5] = (byte) (h >>> 16);
                out[6] = (byte) (h >>> 8);
                out[7] = (byte) h;
                if (rowStride == rowBytes) {
                    buf.position(0);
                    buf.get(out, 8, rowBytes * h);
                } else {
                    for (int y = 0; y < h; y++) {
                        buf.position(y * rowStride);
                        buf.get(out, 8 + y * rowBytes, rowBytes);
                    }
                }
                return out;
            } catch (Exception e) {
                Bridge.log("Bild konnte nicht gelesen werden: " + e.getMessage());
                return null;
            }
        }
    }

    // ---------------------------------------------------------------- Bot

    private void startBot(String config) {
        botThread = new Thread(() -> {
            try {
                if (!Python.isStarted()) {
                    Bridge.log("Starte Bot ...");
                    Python.start(new AndroidPlatform(getApplicationContext()));
                }
                PyObject module = Python.getInstance().getModule("android_main");
                module.callAttr("run", config);
            } catch (Throwable t) {
                String msg = String.valueOf(t.getMessage());
                if (msg.length() > 400) {
                    msg = msg.substring(msg.length() - 400);
                }
                Bridge.log("Bot abgestuerzt: " + msg);
            } finally {
                main.post(this::shutdown);
            }
        }, "autospiel-bot");
        botThread.start();
    }

    void launchGame(String pkg) {
        Intent launch = getPackageManager().getLaunchIntentForPackage(pkg);
        if (launch != null) {
            launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            try {
                startActivity(launch);
            } catch (Exception e) {
                Bridge.log("Spiel konnte nicht gestartet werden");
            }
        }
    }

    // ---------------------------------------------------------------- Ende

    private void shutdown() {
        if (stopping) {
            return;
        }
        stopping = true;
        Bridge.running = false;
        TapService t = TapService.instance;
        if (t != null) {
            t.hideOverlay();
        }
        try {
            if (display != null) {
                display.release();
            }
        } catch (Exception ignored) {
        }
        try {
            if (projection != null) {
                projection.stop();
            }
        } catch (Exception ignored) {
        }
        synchronized (frameLock) {
            if (latest != null) {
                latest.close();
                latest = null;
            }
            frameLock.notifyAll();
        }
        try {
            if (reader != null) {
                reader.close();
            }
        } catch (Exception ignored) {
        }
        if (imageThread != null) {
            imageThread.quitSafely();
        }
        if (Bridge.service == this) {
            Bridge.service = null;
        }
        Bridge.notifyChanged();
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
    }

    @Override
    public void onDestroy() {
        shutdown();
        super.onDestroy();
    }
}
