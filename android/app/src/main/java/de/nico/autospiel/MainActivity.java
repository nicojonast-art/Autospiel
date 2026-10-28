package de.nico.autospiel;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.provider.Settings;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.button.MaterialButtonToggleGroup;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.android.material.materialswitch.MaterialSwitch;

import org.json.JSONObject;

public class MainActivity extends AppCompatActivity implements Bridge.Listener {

    private static final String PREFS = "einstellungen";

    private TextView statusText;
    private TextView statusDetail;
    private TextView levelCount;
    private View statusDot;
    private MaterialButton startButton;
    private TextView logView;

    private ImageView checkAccess;
    private ImageView checkGame;
    private ImageView checkBattery;
    private ImageView checkNotify;
    private MaterialButton btnAccess;
    private MaterialButton btnGame;
    private MaterialButton btnBattery;
    private MaterialButton btnNotify;
    private View rowNotify;

    private MaterialSwitch swBoosters;
    private MaterialSwitch swCoins;
    private MaterialButtonToggleGroup gameToggle;
    private TextView gameTitle;
    private String game = "yarn";

    private ActivityResultLauncher<Intent> captureLauncher;
    private ActivityResultLauncher<String> notifyLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText = findViewById(R.id.status_text);
        statusDetail = findViewById(R.id.status_detail);
        levelCount = findViewById(R.id.level_count);
        statusDot = findViewById(R.id.status_dot);
        startButton = findViewById(R.id.start_button);
        logView = findViewById(R.id.log_view);

        checkAccess = findViewById(R.id.check_access);
        checkGame = findViewById(R.id.check_game);
        checkBattery = findViewById(R.id.check_battery);
        checkNotify = findViewById(R.id.check_notify);
        btnAccess = findViewById(R.id.btn_access);
        btnGame = findViewById(R.id.btn_game);
        btnBattery = findViewById(R.id.btn_battery);
        btnNotify = findViewById(R.id.btn_notify);
        rowNotify = findViewById(R.id.row_notify);

        swBoosters = findViewById(R.id.sw_boosters);
        swCoins = findViewById(R.id.sw_coins);

        TextView version = findViewById(R.id.version_text);
        try {
            String v = getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
            version.setText(getString(R.string.version, v));
        } catch (PackageManager.NameNotFoundException e) {
            version.setText("");
        }

        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        game = prefs.getString("game", "yarn");
        gameToggle = findViewById(R.id.game_toggle);
        gameTitle = findViewById(R.id.step_game_title);
        gameToggle.check("car".equals(game) ? R.id.btn_game_car : R.id.btn_game_yarn);
        gameToggle.addOnButtonCheckedListener((group, id, checked) -> {
            if (!checked) {
                return;
            }
            game = id == R.id.btn_game_car ? "car" : "yarn";
            prefs.edit().putString("game", game).apply();
            refresh();
        });
        swBoosters.setChecked(prefs.getBoolean("boosters", true));
        swCoins.setChecked(prefs.getBoolean("coins", false));
        swBoosters.setOnCheckedChangeListener((b, on) -> prefs.edit().putBoolean("boosters", on).apply());
        swCoins.setOnCheckedChangeListener((b, on) -> prefs.edit().putBoolean("coins", on).apply());

        captureLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(), result -> {
                    if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                        startBot(result.getResultCode(), result.getData());
                    } else {
                        toast(R.string.capture_denied);
                    }
                });
        notifyLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(), granted -> refresh());

        startButton.setOnClickListener(v -> {
            if (Bridge.running) {
                BotService.requestStop(this);
            } else {
                onStartPressed();
            }
        });

        btnAccess.setOnClickListener(v -> openAccessibility());
        findViewById(R.id.btn_restricted).setOnClickListener(v -> openAppInfo());
        btnGame.setOnClickListener(v -> openPlayStore());
        btnBattery.setOnClickListener(v -> askBattery());
        btnNotify.setOnClickListener(v -> {
            if (Build.VERSION.SDK_INT >= 33) {
                notifyLauncher.launch(Manifest.permission.POST_NOTIFICATIONS);
            }
        });
        findViewById(R.id.btn_clear_log).setOnClickListener(v -> {
            Bridge.clearLog();
            refresh();
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        Bridge.listener = this;
        refresh();
    }

    @Override
    protected void onPause() {
        if (Bridge.listener == this) {
            Bridge.listener = null;
        }
        super.onPause();
    }

    @Override
    public void onBotChanged() {
        refresh();
    }

    // ------------------------------------------------------------ Starten

    private void onStartPressed() {
        if (!isAccessibilityOn(this)) {
            new MaterialAlertDialogBuilder(this)
                    .setTitle(R.string.need_access_title)
                    .setMessage(R.string.need_access_text)
                    .setPositiveButton(R.string.open_settings, (d, w) -> openAccessibility())
                    .setNegativeButton(R.string.cancel, null)
                    .show();
            return;
        }
        if (!isGameInstalled()) {
            new MaterialAlertDialogBuilder(this)
                    .setTitle(getString(R.string.need_game_title, gameName()))
                    .setMessage(getString(R.string.need_game_text, gameName()))
                    .setPositiveButton(R.string.play_store, (d, w) -> openPlayStore())
                    .setNegativeButton(R.string.cancel, null)
                    .show();
            return;
        }
        new MaterialAlertDialogBuilder(this)
                .setTitle(R.string.capture_title)
                .setMessage(R.string.capture_text)
                .setPositiveButton(R.string.go, (d, w) -> {
                    MediaProjectionManager mpm = getSystemService(MediaProjectionManager.class);
                    try {
                        captureLauncher.launch(mpm.createScreenCaptureIntent());
                    } catch (ActivityNotFoundException e) {
                        toast(R.string.capture_denied);
                    }
                })
                .setNegativeButton(R.string.cancel, null)
                .show();
    }

    private void startBot(int code, Intent data) {
        String config;
        try {
            JSONObject cfg = new JSONObject();
            cfg.put("game", game);
            cfg.put("use_boosters", swBoosters.isChecked());
            cfg.put("allow_coins", swCoins.isChecked());
            config = cfg.toString();
        } catch (Exception e) {
            config = "{}";
        }
        Bridge.clearLog();
        Intent svc = new Intent(this, BotService.class)
                .setAction(BotService.ACTION_START)
                .putExtra(BotService.EXTRA_CODE, code)
                .putExtra(BotService.EXTRA_DATA, data)
                .putExtra(BotService.EXTRA_CONFIG, config);
        ContextCompat.startForegroundService(this, svc);
        // Spiel direkt nach vorne holen
        startButton.postDelayed(this::launchGame, 900);
    }

    private void launchGame() {
        Intent launch = getPackageManager().getLaunchIntentForPackage(Bridge.packageFor(game));
        if (launch != null) {
            launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(launch);
        }
    }

    // ------------------------------------------------------------ Anzeige

    private void refresh() {
        boolean running = Bridge.running;
        boolean access = isAccessibilityOn(this);
        boolean game = isGameInstalled();
        boolean battery = isBatteryFree();
        boolean notify = Build.VERSION.SDK_INT < 33 || ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED;

        gameTitle.setText(getString(R.string.step_game, gameName()));
        for (int i = 0; i < gameToggle.getChildCount(); i++) {
            gameToggle.getChildAt(i).setEnabled(!running);
        }
        setCheck(checkAccess, btnAccess, access);
        setCheck(checkGame, btnGame, game);
        setCheck(checkBattery, btnBattery, battery);
        setCheck(checkNotify, btnNotify, notify);
        rowNotify.setVisibility(Build.VERSION.SDK_INT >= 33 ? View.VISIBLE : View.GONE);

        int color;
        if (running) {
            statusText.setText(R.string.state_running);
            color = R.color.ok;
        } else if (access && game) {
            statusText.setText(R.string.state_ready);
            color = R.color.accent;
        } else {
            statusText.setText(R.string.state_setup);
            color = R.color.warn;
        }
        statusDot.setBackgroundTintList(ColorStateList.valueOf(ContextCompat.getColor(this, color)));

        String detail = Bridge.status;
        if (detail == null || detail.isEmpty()) {
            detail = getString(running ? R.string.detail_running : R.string.detail_idle);
        }
        statusDetail.setText(detail);
        levelCount.setText(String.valueOf(Bridge.levels));

        startButton.setText(running ? R.string.stop : R.string.start);
        startButton.setIconResource(running ? R.drawable.ic_stop : R.drawable.ic_play);
        startButton.setBackgroundTintList(ColorStateList.valueOf(
                ContextCompat.getColor(this, running ? R.color.danger : R.color.accent)));

        swBoosters.setEnabled(!running);
        swCoins.setEnabled(!running);

        String log = Bridge.logText();
        logView.setText(log.isEmpty() ? getString(R.string.log_empty) : log);
    }

    private String gameName() {
        return getString("car".equals(game) ? R.string.game_car : R.string.game_yarn);
    }

    private void setCheck(ImageView icon, MaterialButton button, boolean ok) {
        icon.setImageResource(ok ? R.drawable.ic_check : R.drawable.ic_todo);
        button.setVisibility(ok ? View.GONE : View.VISIBLE);
    }

    // ------------------------------------------------------------ Pruefungen

    static boolean isAccessibilityOn(Context c) {
        if (TapService.instance != null) {
            return true;
        }
        String enabled = Settings.Secure.getString(c.getContentResolver(),
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES);
        return enabled != null && enabled.toLowerCase().contains(c.getPackageName().toLowerCase() + "/");
    }

    private boolean isGameInstalled() {
        return getPackageManager().getLaunchIntentForPackage(Bridge.packageFor(game)) != null;
    }

    private boolean isBatteryFree() {
        PowerManager pm = getSystemService(PowerManager.class);
        return pm != null && pm.isIgnoringBatteryOptimizations(getPackageName());
    }

    // ------------------------------------------------------------ Einstellungen oeffnen

    private void openAccessibility() {
        try {
            startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
            toast(R.string.access_hint_toast);
        } catch (ActivityNotFoundException e) {
            openAppInfo();
        }
    }

    private void openAppInfo() {
        Intent i = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.parse("package:" + getPackageName()));
        try {
            startActivity(i);
        } catch (ActivityNotFoundException ignored) {
        }
    }

    private void openPlayStore() {
        try {
            startActivity(new Intent(Intent.ACTION_VIEW,
                    Uri.parse("market://details?id=" + Bridge.packageFor(game))));
        } catch (ActivityNotFoundException e) {
            startActivity(new Intent(Intent.ACTION_VIEW,
                    Uri.parse("https://play.google.com/store/apps/details?id=" + Bridge.packageFor(game))));
        }
    }

    @SuppressWarnings("BatteryLife")
    private void askBattery() {
        try {
            startActivity(new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                    Uri.parse("package:" + getPackageName())));
        } catch (ActivityNotFoundException e) {
            openAppInfo();
        }
    }

    private void toast(int res) {
        Toast.makeText(this, res, Toast.LENGTH_LONG).show();
    }
}
