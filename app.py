"""
Breathing Pattern Classification — Demo App
Run in Colab with:
    !pip install gradio -q
    !python gradio_app.py
Or inline:
    import gradio as gr
    # (paste this file's contents into a cell)
"""

import gradio as gr
import numpy as np
import pandas as pd
import pickle
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# ── Constants (must match training config) ────────────────────────────────────
AUDIO_SR    = 22050
N_MFCC      = 13
ACCEL_SR    = 100
BREATH_LOW  = 0.1
BREATH_HIGH = 0.5
CLASS_NAMES = ['Passive (Resting)', 'Active (Post-Exercise)']

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
    for model_path in ['/content/best_model.pkl', './best_model.pkl']:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            break
    for scaler_path in ['/content/scaler.pkl', './scaler.pkl']:
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            break
    return model, scaler

model, scaler = load_artifacts()

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_audio_features(file_path):
    audio, sr = librosa.load(file_path, sr=AUDIO_SR)
    mfccs     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(audio)
    return np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(centroid),
        np.mean(zcr),
    ])

def extract_accel_features(df):
    magnitude = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2).values
    time_features = np.array([
        np.mean(magnitude), np.std(magnitude),
        np.max(magnitude),  np.min(magnitude),
        np.max(magnitude) - np.min(magnitude),
        np.var(magnitude),
        np.mean(df['x']), np.std(df['x']),
        np.mean(df['y']), np.std(df['y']),
        np.mean(df['z']), np.std(df['z']),
    ])
    n         = len(magnitude)
    freqs     = fftfreq(n, d=1.0 / ACCEL_SR)
    fft_mag   = np.abs(fft(magnitude))
    pos_mask  = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_pos   = fft_mag[pos_mask]
    breath_mask  = (freqs_pos >= BREATH_LOW) & (freqs_pos <= BREATH_HIGH)
    total_power  = np.sum(fft_pos) + 1e-8
    if breath_mask.sum() > 0:
        breath_power  = np.sum(fft_pos[breath_mask])
        dominant_freq = freqs_pos[breath_mask][np.argmax(fft_pos[breath_mask])]
    else:
        breath_power, dominant_freq = 0.0, 0.0
    return np.array([*time_features, dominant_freq, breath_power, breath_power / total_power])

# ── Signal visualization ──────────────────────────────────────────────────────
def plot_accel_signal(df):
    t         = df['seconds_elapsed'].values
    magnitude = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2).values

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor='#0f1117')
    fig.suptitle('Accelerometer Signal Analysis', color='white', fontsize=14, fontweight='bold', y=0.98)

    # Raw axes
    ax = axes[0]
    ax.set_facecolor('#1a1d27')
    for col, color in zip(['x', 'y', 'z'], ['#ff6b6b', '#4ecdc4', '#ffe66d']):
        ax.plot(t, df[col], label=col.upper(), color=color, linewidth=1.2, alpha=0.9)
    ax.set_ylabel('Accel (m/s²)', color='#aaaaaa', fontsize=9)
    ax.set_title('Raw X / Y / Z', color='white', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.3)
    ax.tick_params(colors='#777777')
    for spine in ax.spines.values():
        spine.set_color('#333344')

    # Magnitude
    ax = axes[1]
    ax.set_facecolor('#1a1d27')
    ax.plot(t, magnitude, color='#a78bfa', linewidth=1.4)
    ax.fill_between(t, magnitude, alpha=0.15, color='#a78bfa')
    ax.set_ylabel('Magnitude (m/s²)', color='#aaaaaa', fontsize=9)
    ax.set_title('Signal Magnitude', color='white', fontsize=10)
    ax.tick_params(colors='#777777')
    for spine in ax.spines.values():
        spine.set_color('#333344')

    # FFT
    ax = axes[2]
    ax.set_facecolor('#1a1d27')
    n         = len(magnitude)
    freqs     = fftfreq(n, d=1.0 / ACCEL_SR)
    fft_mag   = np.abs(fft(magnitude))
    pos_mask  = freqs > 0
    ax.plot(freqs[pos_mask], fft_mag[pos_mask], color='#34d399', linewidth=1.2)
    ax.axvspan(BREATH_LOW, BREATH_HIGH, alpha=0.2, color='#34d399',
               label=f'Breathing band ({BREATH_LOW}–{BREATH_HIGH} Hz)')
    ax.set_xlim(0, 2)
    ax.set_xlabel('Frequency (Hz)', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Amplitude', color='#aaaaaa', fontsize=9)
    ax.set_title('FFT — Breathing Band Highlighted', color='white', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor='white')
    ax.tick_params(colors='#777777')
    for spine in ax.spines.values():
        spine.set_color('#333344')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_feature_bar(features, feature_names, prediction, confidence):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')

    color = '#4ecdc4' if prediction == 0 else '#ff6b6b'
    bars  = ax.barh(feature_names, np.abs(features), color=color, alpha=0.75, height=0.6)

    ax.set_xlabel('|Feature Value|', color='#aaaaaa', fontsize=9)
    ax.set_title(
        f'Feature Values  |  Prediction: {CLASS_NAMES[prediction]}  ({confidence:.1%} confidence)',
        color='white', fontsize=11, fontweight='bold'
    )
    ax.tick_params(colors='#777777', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333344')
    plt.tight_layout()
    return fig

# ── Main prediction function ──────────────────────────────────────────────────
def predict(audio_file, accel_file):
    if model is None:
        return (
            "⚠️ No model found. Run the training notebook first to generate best_model.pkl",
            None, None, None
        )

    errors = []

    # Audio features
    audio_feat = None
    if audio_file is not None:
        try:
            audio_feat = extract_audio_features(audio_file)
        except Exception as e:
            errors.append(f"Audio error: {e}")

    # Accel features + plot
    accel_feat = None
    accel_plot = None
    feature_table = None

    if accel_file is not None:
        try:
            df = pd.read_csv(accel_file)
            # Validate columns
            missing = [c for c in ['x', 'y', 'z', 'seconds_elapsed'] if c not in df.columns]
            if missing:
                errors.append(f"CSV missing columns: {missing}. Expected: time, seconds_elapsed, x, y, z")
            else:
                accel_feat = extract_accel_features(df)
                accel_plot = plot_accel_signal(df)

                # Feature table
                feature_table = pd.DataFrame({
                    'Feature': ACCEL_FEATURE_NAMES,
                    'Value':   [f'{v:.4f}' for v in accel_feat]
                })
        except Exception as e:
            errors.append(f"Accel error: {e}")

    # Build fused feature vector
    if audio_feat is None and accel_feat is None:
        return "Upload at least one file to get a prediction.", None, None, None

    if audio_feat is None:
        audio_feat = np.zeros(len(AUDIO_FEATURE_NAMES))
        errors.append("⚠️ No audio file — audio features set to zero")
    if accel_feat is None:
        accel_feat = np.zeros(len(ACCEL_FEATURE_NAMES))
        errors.append("⚠️ No accel file — accel features set to zero")

    fused = np.concatenate([audio_feat, accel_feat]).reshape(1, -1)

    if scaler is not None:
        fused = scaler.transform(fused)

    pred       = model.predict(fused)[0]
    proba      = model.predict_proba(fused)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
    confidence = proba[pred]

    label = CLASS_NAMES[pred]
    emoji = "🧘" if pred == 0 else "🏃"

    result_text = f"{emoji}  **{label}**\nConfidence: {confidence:.1%}"
    if errors:
        result_text += "\n\n" + "\n".join(errors)

    feat_plot = plot_feature_bar(
        np.concatenate([audio_feat.flatten(), accel_feat.flatten()]),
        ALL_FEATURE_NAMES, pred, confidence
    )

    return result_text, accel_plot, feat_plot, feature_table

# ── Accel-only live preview (CSV upload → signal plot, no model needed) ───────
def preview_signal(accel_file):
    if accel_file is None:
        return None, "Upload a CSV file to preview the signal."
    try:
        df = pd.read_csv(accel_file)
        missing = [c for c in ['x', 'y', 'z', 'seconds_elapsed'] if c not in df.columns]
        if missing:
            return None, f"Missing columns: {missing}"
        fig = plot_accel_signal(df)
        feats = extract_accel_features(df)
        summary = "\n".join([f"{n}: {v:.4f}" for n, v in zip(ACCEL_FEATURE_NAMES, feats)])
        return fig, f"**Extracted Features:**\n```\n{summary}\n```"
    except Exception as e:
        return None, f"Error: {e}"

# ── UI ────────────────────────────────────────────────────────────────────────
css = """
body { background: #0f1117; }
.gradio-container { font-family: 'JetBrains Mono', monospace; background: #0f1117; }
.tab-nav { background: #1a1d27; border-bottom: 1px solid #333; }
.result-box { background: #1a1d27; border: 1px solid #333344; border-radius: 8px; padding: 16px; }
h1 { color: white; letter-spacing: -0.5px; }
"""

with gr.Blocks(theme=gr.themes.Base(), css=css, title="Breathing Pattern Classifier") as demo:
    gr.Markdown("""
    # 🫁 Breathing Pattern Classifier
    **ML & Sensing Final Project** — Maanvi Sarwadi, Katie Jiang, Aanand Patel, Alina Zacaria, Hayah Ubaid
    
    Upload an **Accelerometer.csv** from PhyPhox (and optionally a **.wav** audio file) to classify as **Resting** or **Active** breathing.
    """)

    with gr.Tabs():

        # ── Tab 1: Full prediction ────────────────────────────────────────────
        with gr.TabItem("🔬 Classify Recording"):
            with gr.Row():
                with gr.Column(scale=1):
                    accel_input = gr.File(label="Accelerometer CSV (PhyPhox)", file_types=[".csv"])
                    audio_input = gr.File(label="Audio File (optional)", file_types=[".wav", ".m4a", ".mp3"])
                    predict_btn = gr.Button("Classify", variant="primary")

                with gr.Column(scale=1):
                    result_out = gr.Markdown(label="Prediction", elem_classes=["result-box"])

            accel_plot_out  = gr.Plot(label="Accelerometer Signal")
            feature_plot_out = gr.Plot(label="Feature Values")
            feature_table_out = gr.Dataframe(label="Accel Feature Values", headers=["Feature", "Value"])

            predict_btn.click(
                fn=predict,
                inputs=[audio_input, accel_input],
                outputs=[result_out, accel_plot_out, feature_plot_out, feature_table_out]
            )

        # ── Tab 2: Signal preview only ────────────────────────────────────────
        with gr.TabItem("📈 Preview Signal"):
            gr.Markdown("""
            Upload any **Accelerometer.csv** to visualize the raw signal and extracted features — 
            no model needed. Useful for checking your PhyPhox recordings before classifying.
            """)
            preview_input = gr.File(label="Accelerometer CSV", file_types=[".csv"])
            preview_btn   = gr.Button("Preview", variant="secondary")
            preview_plot  = gr.Plot(label="Signal Visualization")
            preview_text  = gr.Markdown()

            preview_btn.click(
                fn=preview_signal,
                inputs=[preview_input],
                outputs=[preview_plot, preview_text]
            )

        # ── Tab 3: How to use ─────────────────────────────────────────────────
        with gr.TabItem("📖 How to Use"):
            gr.Markdown("""
            ## Recording with PhyPhox
            1. Open PhyPhox → **Acceleration (without g)** experiment
            2. Place phone **flat on your diaphragm** (chest/stomach)
            3. Press **Record** and breathe normally for 10–15 seconds (resting) or record right after exercise (active)
            4. Export → **CSV** → share to your computer
            5. Upload the `Accelerometer.csv` in the **Classify** tab

            ## CSV Format Expected
            PhyPhox exports a CSV with these columns:
            ```
            time, seconds_elapsed, x, y, z
            ```
            - `x`, `y`, `z` — acceleration in m/s²
            - `seconds_elapsed` — time axis for plots
            - Sampling rate: ~100 Hz

            ## Model Info
            - **Features**: 15 audio (MFCCs, spectral centroid, ZCR) + 15 accel (time-domain + FFT breathing band)
            - **Models trained**: Random Forest, Logistic Regression
            - **Classes**: Passive (resting) vs Active (post-exercise)
            
            ## Notes on 100% Accuracy
            The model currently achieves perfect scores because resting vs. post-exercise states 
            are very physically distinct. Future work: classify subtle breathing irregularities 
            within the same activity state for a harder and more clinically meaningful task.
            """)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)