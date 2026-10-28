package de.nico.autospiel;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.content.Intent;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.os.Handler;
import android.os.Looper;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;
import android.widget.TextView;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Bedienungshilfe: fuehrt die Taps aus, merkt sich die App im Vordergrund
 * und zeigt waehrend des Spielens eine kleine Stopp-Pille ueber dem Spiel.
 */
public class TapService extends AccessibilityService {

    static volatile TapService instance;
    static volatile String lastWindow = "";

    private final Handler main = new Handler(Looper.getMainLooper());
    private View overlay;
    private TextView overlayText;

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
        Bridge.notifyChanged();
    }

    @Override
    public boolean onUnbind(Intent intent) {
        hideOverlay();
        instance = null;
        Bridge.notifyChanged();
        return super.onUnbind(intent);
    }

    @Override
    public void onDestroy() {
        hideOverlay();
        instance = null;
        super.onDestroy();
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (event == null || event.getEventType() != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            return;
        }
        CharSequence pkg = event.getPackageName();
        if (pkg == null) {
            return;
        }
        String p = pkg.toString();
        CharSequence cls = event.getClassName();
        String c = cls == null ? "" : cls.toString();
        // Systemleiste, Tastatur und unsere eigene Stopp-Pille ignorieren
        if (p.equals("com.android.systemui") || p.contains("inputmethod")) {
            return;
        }
        if (p.equals(getPackageName()) && !c.endsWith("MainActivity")) {
            return;
        }
        lastWindow = p + "/" + c;
    }

    @Override
    public void onInterrupt() {
    }

    // ---------------------------------------------------------------- Taps

    boolean tap(int x, int y) {
        if (x < 0 || y < 0) {
            return false;
        }
        Path path = new Path();
        path.moveTo(x, y);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(new GestureDescription.StrokeDescription(path, 0, 60))
                .build();
        final CountDownLatch done = new CountDownLatch(1);
        final boolean[] ok = {false};
        boolean sent = dispatchGesture(gesture, new GestureResultCallback() {
            @Override
            public void onCompleted(GestureDescription g) {
                ok[0] = true;
                done.countDown();
            }

            @Override
            public void onCancelled(GestureDescription g) {
                done.countDown();
            }
        }, null);
        if (!sent) {
            return false;
        }
        try {
            done.await(1500, TimeUnit.MILLISECONDS);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
        return ok[0];
    }

    void back() {
        performGlobalAction(GLOBAL_ACTION_BACK);
    }

    void launchGame(String pkg) {
        Intent launch = getPackageManager().getLaunchIntentForPackage(pkg);
        if (launch == null) {
            Bridge.log("Spiel ist nicht installiert: " + pkg);
            return;
        }
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED);
        try {
            startActivity(launch);
        } catch (Exception e) {
            Bridge.log("Spiel konnte nicht gestartet werden: " + e.getMessage());
        }
    }

    // ---------------------------------------------------------------- Stopp-Pille

    void showOverlay() {
        main.post(() -> {
            if (overlay != null) {
                updateOverlay();
                return;
            }
            WindowManager wm = (WindowManager) getSystemService(WINDOW_SERVICE);
            View v = LayoutInflater.from(this).inflate(R.layout.overlay_pill, null);
            overlayText = v.findViewById(R.id.overlay_text);
            v.setOnClickListener(view -> BotService.requestStop(this));
            WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                            | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                            | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                    PixelFormat.TRANSLUCENT);
            lp.gravity = Gravity.TOP | Gravity.START;
            lp.x = dp(10);
            lp.y = dp(6);
            try {
                wm.addView(v, lp);
                overlay = v;
                updateOverlay();
            } catch (Exception e) {
                Bridge.log("Stopp-Knopf konnte nicht angezeigt werden: " + e.getMessage());
            }
        });
    }

    void hideOverlay() {
        main.post(() -> {
            if (overlay == null) {
                return;
            }
            try {
                ((WindowManager) getSystemService(WINDOW_SERVICE)).removeView(overlay);
            } catch (Exception ignored) {
            }
            overlay = null;
            overlayText = null;
        });
    }

    void updateOverlay() {
        if (overlayText == null) {
            return;
        }
        int n = Bridge.levels;
        overlayText.setText(n == 1 ? getString(R.string.overlay_running_one)
                : getString(R.string.overlay_running, n));
    }

    private int dp(int v) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v,
                getResources().getDisplayMetrics());
    }
}
