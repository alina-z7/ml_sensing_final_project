"""
Breathing Pattern Classification — Demo App
Run with: python app.py
Then open the local URL printed in terminal.

Install deps:
    pip install gradio matplotlib numpy scipy requests scikit-learn librosa
"""

import gradio as gr
import numpy as np
import pandas as pd
import pickle
import librosa
import threading
import time
import requests
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from scipy.fft import fft, fftfreq

# ── Constants ────────────────────────────────────────────────────────────────
AUDIO_SR     = 22050
N_MFCC       = 13
ACCEL_SR     = 100
BREATH_LOW   = 0.1
BREATH_HIGH  = 0.5
WINDOW_SEC   = 10
CLASS_NAMES  = ['Passive (Resting)', 'Active (Post-Exercise)']
COLORS       = ['#4ecdc4', '#ff6b6b']

AUDIO_FEATURE_NAMES = (
    [f'mfcc_{i}_mean' for i in range(N_MFCC)] +
    ['centroid_mean', 'zcr_mean']
)
ACCEL_FEATURE_NAMES = [
    'mag_mean', 'mag_std', 'mag_max', 'mag_min', 'mag_range', 'mag_var',
    'x_mean', 'x_std', 'y_mean', 'y_std', 'z_mean', 'z_std',
    'dominant_freq_hz', 'breath_band_power', 'breath_power_ratio',
]
ALL_FEATURE_NAMES = AUDIO_FEATURE_NAMES + ACCEL_FEATURE_NAMES

# ── Load model + scaler ───────────────────────────────────────────────────────
def load_artifacts():
    model, scaler = None, None
    for p in ['./content/best_model.pkl', './best_model.pkl']:
        if os.path.exists(p):
            with open(p, 'rb') as f: model = pickle.load(f)
            break
    for p in ['./content/scaler.pkl', './scaler.pkl']:
        if os.path.exists(p):
            with open(p, 'rb') as f: scaler = pickle.load(f)
            break
    return model, scaler

model, scaler = load_artifacts()

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_audio_features(file_path):
    audio, sr = librosa.load(file_path, sr=AUDIO_SR)
    mfccs    = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(audio)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(centroid), np.mean(zcr)])

def extract_accel_features_arr(x_arr, y_arr, z_arr):
    x, y, z   = np.array(x_arr), np.array(y_arr), np.array(z_arr)
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    time_feat = np.array([
        np.mean(magnitude), np.std(magnitude),
        np.max(magnitude),  np.min(magnitude),
        np.max(magnitude) - np.min(magnitude), np.var(magnitude),
        np.mean(x), np.std(x), np.mean(y), np.std(y), np.mean(z), np.std(z),
    ])
    n = len(magnitude)
    freqs    = fftfreq(n, d=1.0 / ACCEL_SR)
    fft_mag  = np.abs(fft(magnitude))
    pos      = freqs > 0
    fp, fm   = freqs[pos], fft_mag[pos]
    bm       = (fp >= BREATH_LOW) & (fp <= BREATH_HIGH)
    tp       = np.sum(fm) + 1e-8
    bp       = np.sum(fm[bm]) if bm.sum() > 0 else 0.0
    df       = fp[bm][np.argmax(fm[bm])] if bm.sum() > 0 else 0.0
    return np.array([*time_feat, df, bp, bp / tp])

def extract_accel_features_df(df):
    return extract_accel_features_arr(df['x'].values, df['y'].values, df['z'].values)

def run_model(audio_feat, accel_feat):
    if model is None:
        return None, None
    af = audio_feat if audio_feat is not None else np.zeros(len(AUDIO_FEATURE_NAMES))
    ef = accel_feat if accel_feat is not None else np.zeros(len(ACCEL_FEATURE_NAMES))
    fused = np.concatenate([af, ef]).reshape(1, -1)
    if scaler is not None:
        fused = scaler.transform(fused)
    pred   = model.predict(fused)[0]
    proba  = model.predict_proba(fused)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
    return int(pred), float(proba[pred])

# ── Plotting helpers ──────────────────────────────────────────────────────────
PLOT_STYLE = dict(facecolor='#0f1117')
AX_STYLE   = '#1a1d27'
SPINE_COL  = '#333344'

def _style_ax(ax):
    ax.set_facecolor(AX_STYLE)
    ax.tick_params(colors='#777777', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(SPINE_COL)

def make_signal_fig(t, x, y, z, title="Accelerometer Signal"):
    mag = np.sqrt(np.array(x)**2 + np.array(y)**2 + np.array(z)**2)
    t   = np.array(t)
    t_rel = t - t[0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), **PLOT_STYLE)
    fig.suptitle(title, color='white', fontsize=13, fontweight='bold', y=0.99)

    ax = axes[0]; _style_ax(ax)
    for arr, col, lbl in zip([x, y, z], ['#ff6b6b','#4ecdc4','#ffe66d'], ['X','Y','Z']):
        ax.plot(t_rel, arr, color=col, lw=1.1, alpha=0.9, label=lbl)
    ax.set_ylabel('m/s²', color='#aaa', fontsize=8)
    ax.set_title('Raw X / Y / Z', color='white', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.2)

    ax = axes[1]; _style_ax(ax)
    ax.plot(t_rel, mag, color='#a78bfa', lw=1.3)
    ax.fill_between(t_rel, mag, alpha=0.12, color='#a78bfa')
    ax.set_ylabel('m/s²', color='#aaa', fontsize=8)
    ax.set_title('Magnitude', color='white', fontsize=9)

    ax = axes[2]; _style_ax(ax)
    n      = len(mag)
    freqs  = fftfreq(n, d=1.0 / ACCEL_SR)
    fm     = np.abs(fft(mag))
    pos    = freqs > 0
    ax.plot(freqs[pos], fm[pos], color='#34d399', lw=1.1)
    ax.axvspan(BREATH_LOW, BREATH_HIGH, alpha=0.2, color='#34d399',
               label=f'{BREATH_LOW}–{BREATH_HIGH} Hz breathing band')
    ax.set_xlim(0, 2)
    ax.set_xlabel('Frequency (Hz)', color='#aaa', fontsize=8)
    ax.set_title('FFT — Breathing Band Highlighted', color='white', fontsize=9)
    ax.legend(fontsize=7, framealpha=0.2, labelcolor='white')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def make_feature_fig(audio_feat, accel_feat, pred_idx, confidence):
    features = np.concatenate([
        audio_feat if audio_feat is not None else np.zeros(len(AUDIO_FEATURE_NAMES)),
        accel_feat if accel_feat is not None else np.zeros(len(ACCEL_FEATURE_NAMES)),
    ])
    color = COLORS[pred_idx] if pred_idx is not None else '#888'
    fig, ax = plt.subplots(figsize=(10, 5), **PLOT_STYLE)
    _style_ax(ax)
    ax.barh(ALL_FEATURE_NAMES, np.abs(features), color=color, alpha=0.75, height=0.6)
    ax.set_xlabel('|Feature Value|', color='#aaa', fontsize=9)
    title = (f'Features  |  {CLASS_NAMES[pred_idx]}  ({confidence:.1%} confidence)'
             if pred_idx is not None else 'Feature Values')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LIVE STREAM STATE
# ─────────────────────────────────────────────────────────────────────────────
class LiveState:
    def __init__(self):
        self.reset()

    def reset(self):
        buf = WINDOW_SEC * ACCEL_SR * 3
        self.t_buf       = deque(maxlen=buf)
        self.x_buf       = deque(maxlen=buf)
        self.y_buf       = deque(maxlen=buf)
        self.z_buf       = deque(maxlen=buf)
        self.running     = False
        self.last_time   = 0.0
        self.pred_idx    = None
        self.confidence  = 0.0
        self.status      = "Idle"
        self.thread      = None

live = LiveState()

def _to_float_array(val):
    """
    Convert PhyPhox buffer data into a numpy float array.

    Handles:
      [1,2,3]
      {"buffer":[1,2,3]}
      [{"buffer":1}, {"buffer":2}]
      0 / None
    """
    if val is None or val == 0:
        return None

    # {"buffer":[...]}
    if isinstance(val, dict):
        if "buffer" in val:
            return np.array(val["buffer"], dtype=float)
        return None

    # [1,2,3]
    if isinstance(val, list):
        if len(val) == 0:
            return None

        # [{"buffer":1}, ...]
        if isinstance(val[0], dict):
            return np.array(
                [v.get("buffer", 0.0) for v in val],
                dtype=float
            )

        return np.array(val, dtype=float)

    return None

def _poll_loop(ip: str):
    base = f"http://{ip}"

    while live.running:

        try:

            r = requests.get(
                f"{base}/get?accX=full&accY=full&accZ=full&acc_time=full",
                timeout=2
            )

            if r.status_code != 200:
                live.status = f"PhyPhox returned {r.status_code}"
                time.sleep(0.25)
                continue

            data = r.json()

            buf = data.get("buffer", {})

            raw_t = buf.get("acc_time")
            raw_x = buf.get("accX")
            raw_y = buf.get("accY")
            raw_z = buf.get("accZ")

            t = _to_float_array(raw_t)
            x = _to_float_array(raw_x)
            y = _to_float_array(raw_y)
            z = _to_float_array(raw_z)

            if any(v is None for v in [t, x, y, z]):
                live.status = (
                    "Waiting for data. "
                    "Make sure PhyPhox is running and the experiment is recording."
                )
                time.sleep(0.25)
                continue

            min_len = min(len(t), len(x), len(y), len(z))

            if min_len == 0:
                live.status = "No accelerometer samples yet."
                time.sleep(0.25)
                continue

            t = t[-min_len:]
            x = x[-min_len:]
            y = y[-min_len:]
            z = z[-min_len:]

            if live.last_time == 0:
                new_mask = np.ones(len(t), dtype=bool)
            else:
                new_mask = t > live.last_time

            if np.any(new_mask):

                live.t_buf.extend(t[new_mask].tolist())
                live.x_buf.extend(x[new_mask].tolist())
                live.y_buf.extend(y[new_mask].tolist())
                live.z_buf.extend(z[new_mask].tolist())

                live.last_time = float(t[new_mask][-1])

                live.status = (
                    f"Streaming — "
                    f"{len(live.t_buf)} samples collected"
                )

            else:
                live.status = "Connected — waiting for new samples"

        except Exception as e:
            live.status = f"Connection error: {str(e)}"

        time.sleep(0.1)

def start_stream(ip):
    if not ip or not ip.strip():
        return "⚠️ Enter a PhyPhox IP address first."
    ip = ip.strip()
    if live.running:
        return "Already streaming. Stop first."
    try:
        r = requests.get(f"http://{ip}/config", timeout=3)
        if r.status_code != 200:
            return f"❌ Could not connect to http://{ip} — check the IP and that Remote Access is on."
    except Exception as e:
        return f"❌ Could not reach http://{ip}: {e}"

    live.reset()
    live.running = True
    live.thread  = threading.Thread(target=_poll_loop, args=(ip,), daemon=True)
    live.thread.start()
    return f"✅ Connected to {ip} — press Play in PhyPhox. The graph will update automatically."

def stop_stream():
    live.running = False
    live.status  = "Stopped"
    return "Stream stopped."

def refresh_live():

    if len(live.t_buf) < 20:
        return None, f"No usable data yet — {live.status}"
    
    display_sec = 5
    win = display_sec * ACCEL_SR
    t = list(live.t_buf)[-win:]
    x = list(live.x_buf)[-win:]
    y = list(live.y_buf)[-win:]
    z = list(live.z_buf)[-win:]

    fig = make_signal_fig(t, x, y, z, title="Live Accelerometer Signal (PhyPhox)")

    if len(t) >= ACCEL_SR * 2:
        accel_feat      = extract_accel_features_arr(x, y, z)
        pred, conf      = run_model(None, accel_feat)
        live.pred_idx   = pred
        live.confidence = conf

    if live.pred_idx is not None:
        emoji  = "🧘" if live.pred_idx == 0 else "🏃"
        label  = CLASS_NAMES[live.pred_idx]
        result = f"{emoji} **{label}**\nConfidence: {live.confidence:.1%}\n\n_{live.status}_"
    else:
        result = f"_Collecting data... ({live.status})_"

    return fig, result

# ── Tab 1: Classify uploaded recording ───────────────────────────────────────
def predict_uploaded(audio_file, accel_file):
    if model is None:
        return "⚠️ No model found — run the training notebook first.", None, None, None

    errors, audio_feat, accel_feat, accel_plot, feat_table = [], None, None, None, None

    if audio_file:
        try:
            audio_feat = extract_audio_features(audio_file)
        except Exception as e:
            errors.append(f"Audio error: {e}")

    if accel_file:
        try:
            df = pd.read_csv(accel_file)
            missing = [c for c in ['x','y','z','seconds_elapsed'] if c not in df.columns]
            if missing:
                errors.append(f"CSV missing columns: {missing}")
            else:
                accel_feat = extract_accel_features_df(df)
                accel_plot = make_signal_fig(
                    df['seconds_elapsed'].values,
                    df['x'].values, df['y'].values, df['z'].values
                )
                feat_table = pd.DataFrame({
                    'Feature': ACCEL_FEATURE_NAMES,
                    'Value':   [f'{v:.4f}' for v in accel_feat]
                })
        except Exception as e:
            errors.append(f"Accel error: {e}")

    if audio_feat is None and accel_feat is None:
        return "Upload at least one file.", None, None, None

    pred, conf = run_model(audio_feat, accel_feat)
    if pred is None:
        return "⚠️ Model error.", accel_plot, None, feat_table

    emoji = "🧘" if pred == 0 else "🏃"
    text  = f"{emoji} **{CLASS_NAMES[pred]}**\nConfidence: {conf:.1%}"
    if errors:
        text += "\n\n" + "\n".join(errors)

    feat_fig = make_feature_fig(audio_feat, accel_feat, pred, conf)
    return text, accel_plot, feat_fig, feat_table

# ── UI ────────────────────────────────────────────────────────────────────────
css = """
body, .gradio-container { background: #0f1117 !important; }
.tab-nav button { background: #1a1d27 !important; color: #aaa !important; }
.tab-nav button.selected { color: white !important; border-bottom: 2px solid #a78bfa !important; }
.result-box { background: #1a1d27; border: 1px solid #333344; border-radius: 8px; padding: 16px; color: white; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Base(), css=css, title="Breathing Pattern Classifier") as demo:

    gr.Markdown("""
# 🫁 Breathing Pattern Classifier
**ML & Sensing Final Project** — Maanvi Sarwadi, Katie Jiang, Aanand Patel, Alina Zacaria, Hayah Ubaid
""")

    with gr.Tabs():

        with gr.TabItem("🔬 Classify Recording"):
            gr.Markdown("Upload an `Accelerometer.csv` from PhyPhox and optionally a `.wav` audio file.")
            with gr.Row():
                with gr.Column(scale=1):
                    accel_input   = gr.File(label="Accelerometer CSV (PhyPhox)", file_types=[".csv"])
                    audio_input   = gr.File(label="Audio File (optional)", file_types=[".wav",".m4a",".mp3"])
                    predict_btn   = gr.Button("Classify", variant="primary")
                with gr.Column(scale=1):
                    result_out    = gr.Markdown(elem_classes=["result-box"])
            accel_plot_out    = gr.Plot(label="Accelerometer Signal")
            feature_plot_out  = gr.Plot(label="Feature Importance")
            feature_table_out = gr.Dataframe(label="Accel Features", headers=["Feature","Value"])
            predict_btn.click(predict_uploaded,
                              [audio_input, accel_input],
                              [result_out, accel_plot_out, feature_plot_out, feature_table_out])

        with gr.TabItem("📡 Live Stream (PhyPhox)"):
            gr.Markdown("""
**How to connect:**
1. Open PhyPhox → *Acceleration (without g)* → tap **⋮ → Remote Access**
2. Note the IP shown (e.g. `192.168.1.5`) — phone and laptop must be on the **same WiFi**
3. Enter the IP below, click **Start**, then press **Play** in PhyPhox
4. Click **Refresh** to pull the latest data and get a new prediction
""")
            with gr.Row():
                ip_input    = gr.Textbox(label="PhyPhox IP Address", placeholder="e.g. 192.168.1.5", scale=3)
                start_btn   = gr.Button("▶ Start", variant="primary", scale=1)
                stop_btn    = gr.Button("⏹ Stop", variant="secondary", scale=1)
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)

            stream_status = gr.Markdown("_Stream idle_")
            live_plot     = gr.Plot(label="Live Signal")
            live_result   = gr.Markdown(elem_classes=["result-box"])

            timer = gr.Timer(value=0.25)
            timer.tick(
                refresh_live,
                inputs=[],
                outputs=[live_plot, live_result]
            )

            start_btn.click(start_stream, [ip_input], [stream_status])
            stop_btn.click(stop_stream, [], [stream_status])
            refresh_btn.click(refresh_live, [], [live_plot, live_result])

        with gr.TabItem("📖 How to Use"):
            gr.Markdown("""
## Recording with PhyPhox
1. Open PhyPhox → **Acceleration (without g)** experiment
2. Place phone **flat on your diaphragm** (chest/stomach)
3. Press **Record** for 10–15 seconds
4. Export → **CSV** → share the `Accelerometer.csv` to your laptop
5. Upload it in the **Classify Recording** tab

## CSV Format (PhyPhox output)
```
time, seconds_elapsed, x, y, z
```
- `x`, `y`, `z` — acceleration in m/s²
- Sampling rate: ~100 Hz

## Live Stream
- Uses PhyPhox Remote Access over WiFi — no USB needed
- Click **Refresh** to pull the latest data and get a new prediction
- The model classifies the most recent 10-second window

## Model
- **Features**: 15 audio + 15 accel = 30 total
- **Classes**: Passive (resting) vs Active (post-exercise)
- Model files expected at `./best_model.pkl` and `./scaler.pkl`
""")

if __name__ == "__main__":
    print("\n🫁 Breathing Pattern Classifier")
    print("=" * 40)
    if model:
        print("✅ Model loaded")
    else:
        print("⚠️  No model found — run training notebook first")
        print("   Expected: ./best_model.pkl and ./scaler.pkl")
    print("=" * 40)
    demo.launch(share=True)