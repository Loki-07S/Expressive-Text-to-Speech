"""
Main Inference Script for Expressive-TTS

Complete pipeline:
1. Detect emotion & intensity from text (BERT)
2. Generate prosody scaling via Prosody Planner
3. Map detected emotion to IndexTTS2's 8-dimensional emotion vector
4. Pass emotion vector + text + reference audio to real Index-TTS2 engine
5. Index-TTS2 generates mel spectrogram internally
6. BigVGAN vocoder inside Index-TTS2 generates the final waveform
7. Save .wav output

Supports:
- Automatic emotion detection from text
- Manual emotion override
- GPU acceleration (CUDA) with FP16 for speed
- Fully offline operation
"""

import os
import sys
import argparse
import time

# Add parent directories to path so we can import our modules AND Index-TTS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPRESSIVE_TTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(EXPRESSIVE_TTS_DIR)
INDEX_TTS_DIR = os.path.join(PROJECT_DIR, "index-tts")

sys.path.insert(0, EXPRESSIVE_TTS_DIR)
sys.path.insert(0, INDEX_TTS_DIR)
sys.path.insert(0, os.path.join(INDEX_TTS_DIR, "indextts"))

# Our custom modules
from emotion_model.emotion_classifier import EmotionDetector
from prosody.prosody_controller import ProsodyController

# The real Index-TTS2 engine
from indextts.infer_v2 import IndexTTS2

# Mapping from our 7-class emotion labels to IndexTTS2's 8-dimensional emotion vector
# IndexTTS2 vector order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMOTION_TO_VECTOR = {
    "happy":    [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "anger":    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad":      [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fear":     [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
    "disgust":  [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
    "surprise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
    "neutral":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
}

# Global model cache — avoids reloading the heavy model on every call
_tts_model = None


def scale_emotion_vector(base_vector, intensity):
    """Scale emotion vector by detected intensity (0-1)."""
    return [v * intensity for v in base_vector]


def get_tts_model(model_dir, use_fp16=True):
    """
    Load Index-TTS2 model once and cache it globally.
    Subsequent calls return the cached model instantly.
    """
    global _tts_model
    if _tts_model is None:
        print("       Loading model (first time — this takes ~30s, subsequent calls are instant)...")
        _tts_model = IndexTTS2(
            cfg_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            use_fp16=use_fp16,        # FP16 = ~2x faster, ~50% less VRAM
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
    else:
        print("       Model already loaded (cached) ✓")
    return _tts_model


def generate_expressive_speech(
    text,
    reference_audio,
    output_path="output.wav",
    emotion_override=None,
    use_fp16=True,
    model_dir=None,
    verbose=False,
):
    """
    Generates an expressive waveform by passing text through the full pipeline.
    Uses the REAL Index-TTS2 model for actual speech synthesis.

    Args:
        text (str): The text to synthesize.
        reference_audio (str): Path to reference voice audio (.wav) for voice cloning.
        output_path (str): Where to save the generated .wav file.
        emotion_override (str, optional): Force a specific emotion (e.g. "happy", "sad").
        use_fp16 (bool): Use half-precision for faster inference & lower VRAM.
        model_dir (str): Path to the Index-TTS2 checkpoints directory.
        verbose (bool): Print detailed debug info.
    """
    if model_dir is None:
        model_dir = os.path.join(INDEX_TTS_DIR, "checkpoints")

    print("=" * 60)
    print("  EXPRESSIVE TTS PIPELINE")
    print("=" * 60)
    pipeline_start = time.time()

    # ─── Step 1: Emotion Detection ───────────────────────────
    if emotion_override:
        emotion_label = emotion_override.lower()
        intensity = 1.0
        print(f"[1/6] Emotion OVERRIDDEN → {emotion_label.upper()} (intensity: {intensity:.2f})")
    else:
        print("[1/6] Detecting emotion from text...")
        detector = EmotionDetector()
        emotion_info = detector.detect_emotion(text)
        emotion_label = emotion_info["emotion"]
        intensity = emotion_info["intensity"]
        print(f"       Detected: {emotion_label.upper()} (intensity: {intensity:.2f})")

    # ─── Step 2: Prosody Planning ────────────────────────────
    print("[2/6] Generating prosody scales...")
    planner = ProsodyController()
    prosody = planner.generate_prosody_scaling(emotion_label, intensity)
    print(f"       Pitch:    {prosody['pitch_scale']:.3f}")
    print(f"       Energy:   {prosody['energy_scale']:.3f}")
    print(f"       Duration: {prosody['duration_scale']:.3f}")

    # ─── Step 3: Map Emotion to IndexTTS2 Vector ─────────────
    print("[3/6] Mapping emotion to IndexTTS2 emotion vector...")
    base_vector = EMOTION_TO_VECTOR.get(emotion_label, EMOTION_TO_VECTOR["neutral"])
    # Keep vector as *direction*; use `emo_alpha` as strength (IndexTTS2 caps it to 0..1).
    # If we scale by intensity before normalizing, intensity can get washed out.
    emo_vector = base_vector
    print(f"       Vector (base): {[round(v, 4) for v in emo_vector]}")

    # ─── Step 4: Load Index-TTS2 Engine (FP16 for speed) ─────
    print(f"[4/6] Preparing Index-TTS2 model (FP16={use_fp16})...")
    start_load = time.time()
    tts = get_tts_model(model_dir, use_fp16=use_fp16)
    print(f"       Model ready in {time.time() - start_load:.1f}s")

    # Normalize the emotion vector using IndexTTS2's built-in method
    emo_vector = tts.normalize_emo_vec(emo_vector, apply_bias=True)
    print(f"       Normalized vector: {[round(v, 4) for v in emo_vector]}")

    # ─── Step 5: Run Inference ───────────────────────────────
    print(f"[5/6] Synthesizing speech...")
    print(f"       Text: \"{text}\"")
    print(f"       Reference audio: {reference_audio}")
    start_infer = time.time()

    result = tts.infer(
        spk_audio_prompt=reference_audio,
        text=text,
        output_path=output_path,
        emo_vector=emo_vector,
        emo_alpha=float(intensity),
        verbose=verbose,
    )

    infer_time = time.time() - start_infer
    total_time = time.time() - pipeline_start

    # ─── Step 6: Done ────────────────────────────────────────
    print(f"[6/6] Saved output to: {output_path}")
    print(f"       Inference time:  {infer_time:.1f}s")
    print(f"       Total pipeline:  {total_time:.1f}s")
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "output_path": output_path,
        "emotion": emotion_label,
        "intensity": intensity,
        "prosody": prosody,
        "emo_vector": emo_vector,
        "inference_time": infer_time,
        "total_time": total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expressive TTS - Generate emotional speech using Index-TTS2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text", type=str, required=True, help="Input text to synthesize")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference voice audio (.wav)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output .wav path")
    parser.add_argument("--override", type=str, default=None,
                        help="Force a specific emotion (happy, sad, anger, fear, disgust, surprise, neutral)")
    parser.add_argument("--no-fp16", action="store_true", default=False, help="Disable FP16 (use full FP32)")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to Index-TTS2 checkpoints")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose output")
    args = parser.parse_args()

    generate_expressive_speech(
        text=args.text,
        reference_audio=args.ref,
        output_path=args.output,
        emotion_override=args.override,
        use_fp16=not args.no_fp16,  # FP16 ON by default
        model_dir=args.model_dir,
        verbose=args.verbose,
    )
