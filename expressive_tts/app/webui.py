"""
Expressive TTS WebUI

A Gradio-based web interface for the Expressive TTS pipeline.
Features:
- Upload reference voice or use default
- Text input for synthesis
- Automatic emotion detection from text (BERT)
- Manual emotion override with intensity slider
- Prosody visualization
- GPU-accelerated inference with FP16
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPRESSIVE_TTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(EXPRESSIVE_TTS_DIR)
INDEX_TTS_DIR = os.path.join(PROJECT_DIR, "index-tts")

sys.path.insert(0, EXPRESSIVE_TTS_DIR)
sys.path.insert(0, INDEX_TTS_DIR)
sys.path.insert(0, os.path.join(INDEX_TTS_DIR, "indextts"))

import gradio as gr
import torch

from emotion_model.emotion_classifier import EmotionDetector
from prosody.prosody_controller import ProsodyController
from indextts.infer_v2 import IndexTTS2

# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────
DEFAULT_VOICE = os.path.join(INDEX_TTS_DIR, "examples", "reference_voice.wav")
CHECKPOINTS_DIR = os.path.join(INDEX_TTS_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(EXPRESSIVE_TTS_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTIONS = ["auto-detect", "happy", "sad", "anger", "surprise", "fear", "disgust", "neutral"]

EMOTION_TO_VECTOR = {
    "happy":    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "anger":    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad":      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fear":     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "disgust":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "surprise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "neutral":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
}

EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "anger": "😠", "surprise": "😲",
    "fear": "😨", "disgust": "🤢", "neutral": "😐",
}

# ────────────────────────────────────────────────────────────
# Load Models (once at startup)
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("  Loading Expressive TTS Models...")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
if device == "cuda":
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")

# Load emotion detector
print("  Loading Emotion Detector (BERT)...")
emotion_detector = EmotionDetector(device=device)

# Load prosody controller
prosody_controller = ProsodyController()

# Load Index-TTS2
print("  Loading Index-TTS2 engine (FP16)...")
tts_model = IndexTTS2(
    cfg_path=os.path.join(CHECKPOINTS_DIR, "config.yaml"),
    model_dir=CHECKPOINTS_DIR,
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False,
)

print("=" * 60)
print("  All models loaded successfully!")
print("=" * 60)


# ────────────────────────────────────────────────────────────
# Core Pipeline Function
# ────────────────────────────────────────────────────────────
def generate_speech(
    reference_audio,
    text,
    emotion_mode,
    intensity,
    speed,
    progress=gr.Progress(),
):
    """Main synthesis pipeline."""
    if not text or text.strip() == "":
        raise gr.Error("Please enter some text to synthesize.")

    # Use default voice if none uploaded
    if reference_audio is None:
        if os.path.exists(DEFAULT_VOICE):
            reference_audio = DEFAULT_VOICE
        else:
            raise gr.Error("No reference voice found. Please upload a voice audio file.")

    output_path = os.path.join(OUTPUT_DIR, f"gen_{int(time.time())}.wav")
    log_lines = []

    # ── Step 1: Emotion ──────────────────────────────────────
    progress(0.1, desc="Analyzing emotion...")

    # We track what to pass to IndexTTS2
    use_emo_text = False
    emo_vector = None

    if emotion_mode == "auto-detect":
        # Use IndexTTS2's own Qwen-based emotion detection (same as original WebUI)
        # This produces the BEST emotion results because it was trained specifically
        # for IndexTTS2's emotion matrix system.
        use_emo_text = True
        emoji = "🎭"

        # Also run our BERT detector just for the UI display
        emotion_info = emotion_detector.detect_emotion(text)
        emotion_label = emotion_info["emotion"]
        detected_confidence = emotion_info["intensity"]
        emoji = EMOTION_EMOJI.get(emotion_label, "🎭")

        log_lines.append(f"**Emotion Detection (BERT):** {emoji} **{emotion_label.upper()}** (confidence: {detected_confidence:.0%})")
        log_lines.append(f"**Method:** Using IndexTTS2's native Qwen emotion engine ✨")
        log_lines.append(f"**Emotion Intensity (alpha):** {intensity:.0%}")
    else:
        # Manual mode — use emotion vectors directly
        emotion_label = emotion_mode
        emoji = EMOTION_EMOJI.get(emotion_label, "🎭")
        base_vector = EMOTION_TO_VECTOR.get(emotion_label, EMOTION_TO_VECTOR["neutral"])
        # Vector-only guidance.
        # Keep `emo_vector` as a normalized direction vector; use `emo_alpha` for strength.
        emo_vector = tts_model.normalize_emo_vec(base_vector, apply_bias=True)
        log_lines.append(f"**Manual Emotion:** {emoji} **{emotion_label.upper()}**")

        log_lines.append(f"**Intensity:** {intensity:.0%}")
        log_lines.append(f"**Emotion Vector:** `{[round(v, 3) for v in emo_vector]}`")

    # ── Step 2: Prosody ──────────────────────────────────────
    progress(0.2, desc="Planning prosody...")
    prosody_label = emotion_label if emotion_mode != "auto-detect" else "neutral"
    prosody_intensity = intensity
    prosody = prosody_controller.generate_prosody_scaling(prosody_label, prosody_intensity)
    log_lines.append("")
    log_lines.append("**Prosody Scales:**")
    log_lines.append(f"  - Pitch: `{prosody['pitch_scale']:.3f}`")
    log_lines.append(f"  - Energy: `{prosody['energy_scale']:.3f}`")
    log_lines.append(f"  - Duration: `{prosody['duration_scale']:.3f}`")

    # ── Step 3: Synthesize ───────────────────────────────────
    diffusion_steps = int(speed)
    progress(0.4, desc=f"Synthesizing speech ({diffusion_steps} diffusion steps)...")
    start_time = time.time()

    tts_model.gr_progress = progress
    result = tts_model.infer(
        spk_audio_prompt=reference_audio,
        text=text,
        output_path=output_path,
        emo_vector=emo_vector,
        emo_alpha=intensity,          # slider controls emotion strength
        use_emo_text=use_emo_text,     # True for auto-detect, False for manual
        diffusion_steps=diffusion_steps,
        verbose=False,
    )
    log_lines.append(f"**Diffusion Steps:** {diffusion_steps}")
    print(f">> Using {diffusion_steps} diffusion steps, use_emo_text={use_emo_text}, emo_vector={emo_vector}")
    tts_model.gr_progress = None

    elapsed = time.time() - start_time
    log_lines.append("")
    log_lines.append(f"**Inference Time:** {elapsed:.1f}s")

    progress(1.0, desc="Done!")

    pipeline_log = "\n".join(log_lines)
    return output_path, pipeline_log


def detect_emotion_preview(text):
    """Quick emotion preview without full synthesis."""
    if not text or text.strip() == "":
        return "Enter text to see emotion analysis"

    info = emotion_detector.detect_emotion(text)
    label = info["emotion"]
    intensity = info["intensity"]
    emoji = EMOTION_EMOJI.get(label, "🎭")
    prosody = prosody_controller.generate_prosody_scaling(label, intensity)

    result = f"""### {emoji} Detected: **{label.upper()}**
**Confidence:** {intensity:.0%}

| Prosody | Scale |
|---|---|
| 🎵 Pitch | {prosody['pitch_scale']:.3f} |
| 🔊 Energy | {prosody['energy_scale']:.3f} |
| ⏱️ Duration | {prosody['duration_scale']:.3f} |
"""
    return result


# ────────────────────────────────────────────────────────────
# Gradio Interface
# ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: 800;
    margin-bottom: 0;
    padding-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-top: 0;
}
.pipeline-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px;
    background: linear-gradient(145deg, #f8f9ff 0%, #f0f2ff 100%);
}
.emotion-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="Expressive TTS — Emotion-Aware Speech Synthesis",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
) as demo:

    # ── Header ───────────────────────────────────────────────
    gr.HTML("""
        <h1 class="main-title">🎭 Expressive TTS</h1>
        <p class="subtitle">Emotion-Aware Speech Synthesis powered by IndexTTS2 + BERT Emotion Detection</p>
        <hr style="border: 1px solid #e8e8f0; margin: 8px 0 16px 0;">
    """)

    with gr.Row():
        # ── Left Column: Inputs ──────────────────────────────
        with gr.Column(scale=5):
            gr.Markdown("### 🎤 Voice & Text")

            reference_audio = gr.Audio(
                label="Reference Voice (upload or record — leave empty for default)",
                sources=["upload", "microphone"],
                type="filepath",
            )

            text_input = gr.TextArea(
                label="Text to Synthesize",
                placeholder="Type what you want the voice to say...\n\nExample: I can't believe how beautiful this sunset is!",
                lines=4,
            )

            gr.Markdown("### 🎭 Emotion Control")

            with gr.Row():
                emotion_mode = gr.Dropdown(
                    label="Emotion Mode",
                    choices=EMOTIONS,
                    value="auto-detect",
                    info="'auto-detect' uses BERT to detect emotion from your text",
                )
                intensity_slider = gr.Slider(
                    label="Emotion Intensity",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                    info="Controls how strongly the emotion affects the voice",
                )

            with gr.Row():
                speed_slider = gr.Slider(
                    label="⚡ Speed (Diffusion Steps)",
                    minimum=5,
                    maximum=25,
                    value=10,
                    step=1,
                    info="Lower = faster but slightly less quality. 10 is recommended for RTX 3050.",
                )

            generate_btn = gr.Button(
                "🔊 Generate Speech",
                variant="primary",
                size="lg",
            )

        # ── Right Column: Outputs ────────────────────────────
        with gr.Column(scale=4):
            gr.Markdown("### 🔊 Output")

            output_audio = gr.Audio(
                label="Generated Speech",
                type="filepath",
                interactive=False,
            )

            gr.Markdown("### 📊 Pipeline Analysis")

            pipeline_log = gr.Markdown(
                value="*Generate speech to see the analysis...*",
                label="Pipeline Log",
            )

    # ── Emotion Preview Section ──────────────────────────────
    with gr.Accordion("🔍 Live Emotion Preview (type text above to see)", open=False):
        emotion_preview = gr.Markdown(
            value="Enter text above to see real-time emotion analysis",
        )

    # ── Example Texts ────────────────────────────────────────
    with gr.Accordion("💡 Example Texts", open=False):
        gr.Examples(
            examples=[
                ["I am so incredibly happy! This is the best day ever!", "auto-detect", 0.85],
                ["This is terrible news. I can't believe this happened.", "auto-detect", 0.85],
                ["What?! How did you do that? That's absolutely amazing!", "auto-detect", 0.85],
                ["Please help me, I'm really scared right now.", "auto-detect", 0.85],
                ["The weather today is quite normal, nothing special.", "auto-detect", 0.85],
                ["I am going to present my final year project.", "neutral", 0.50],
                ["Stop doing that! I told you a hundred times!", "anger", 0.90],
            ],
            inputs=[text_input, emotion_mode, intensity_slider],
            label="Click an example to fill the text and settings",
        )

    # ── System Info ──────────────────────────────────────────
    with gr.Accordion("⚙️ System Info", open=False):
        gpu_info = "No GPU" if not torch.cuda.is_available() else f"{torch.cuda.get_device_name(0)}"
        gr.Markdown(f"""
| Component | Status |
|---|---|
| **Device** | `{device.upper()}` |
| **GPU** | `{gpu_info}` |
| **Precision** | `FP16` |
| **Emotion Model** | `BERT (bert-base-uncased)` |
| **TTS Engine** | `IndexTTS2` |
| **Vocoder** | `BigVGAN` |
| **Default Voice** | `{'✅ Found' if os.path.exists(DEFAULT_VOICE) else '❌ Not found'}` |
        """)

    # ── Event Handlers ───────────────────────────────────────
    generate_btn.click(
        fn=generate_speech,
        inputs=[reference_audio, text_input, emotion_mode, intensity_slider, speed_slider],
        outputs=[output_audio, pipeline_log],
    )

    text_input.change(
        fn=detect_emotion_preview,
        inputs=[text_input],
        outputs=[emotion_preview],
    )


# ────────────────────────────────────────────────────────────
# Launch
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861, help="Port number")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--share", action="store_true", default=False, help="Create public link")
    args = parser.parse_args()

    demo.queue(20)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[INDEX_TTS_DIR, EXPRESSIVE_TTS_DIR],
    )
