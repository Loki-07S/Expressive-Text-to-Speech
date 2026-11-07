# Expressive TTS (Final Year Project)

Emotion-aware text-to-speech: the system detects emotion from text, maps it to an IndexTTS2 emotion vector, and synthesizes speech with a reference voice.

Built on [IndexTTS2](https://github.com/index-tts/index-tts). Your project code is in `expressive_tts/`. IndexTTS checkpoints are **not** in this repo (they are too large for GitHub). Download them locally after cloning.

## Sample outputs

Play these in the [`samples/`](samples/) folder (GitHub shows a player on each `.wav` file):

| File | Notes |
|------|--------|
| [sample_01.wav](samples/sample_01.wav) | Generated speech sample |
| [sample_02.wav](samples/sample_02.wav) | Generated speech sample |
| [sample_03.wav](samples/sample_03.wav) | Generated speech sample |
| [sample_04.wav](samples/sample_04.wav) | Generated speech sample |
| [sample_05.wav](samples/sample_05.wav) | Generated speech sample |
| [sample_06.wav](samples/sample_06.wav) | Generated speech sample |

## Pipeline

1. **Emotion detection** — BERT classifier over the input text (`happy`, `sad`, `anger`, `surprise`, `fear`, `disgust`, `neutral`) plus an intensity score.
2. **Prosody control** — scales pitch / energy / duration from the detected emotion.
3. **Synthesis** — IndexTTS2 generates the waveform from text, a reference voice, and the 8-D emotion vector.
4. **Web UI** — Gradio app for text, optional emotion override, and playback.

## Repository layout

```
expressive_tts/          # this project's code
  app/webui.py           # Gradio UI
  inference/infer.py     # CLI / inference pipeline
  emotion_model/         # BERT emotion detector
  prosody/               # prosody controller
  index_tts_core/        # acoustic-model wrapper
  evaluation.py          # MCD / pitch RMSE helpers
samples/                 # curated demo voices for GitHub
index-tts/               # IndexTTS2 engine (clone locally; weights not committed)
```

## Setup

1. Clone this repository.
2. Clone IndexTTS2 next to `expressive_tts` (folder name must be `index-tts`):

   ```bash
   git clone https://github.com/index-tts/index-tts.git
   ```

   If `index-tts` is already on your machine, keep it in this project folder.

3. Install IndexTTS2 and download checkpoints as described in their [README](https://github.com/index-tts/index-tts) (Hugging Face / ModelScope). Place weights in `index-tts/checkpoints/`.

4. Run the UI:

   ```bash
   python expressive_tts/app/webui.py
   ```

   Or run inference:

   ```bash
   python expressive_tts/inference/infer.py
   ```

## What is not uploaded

- `index-tts/.venv/` and other virtualenvs
- `index-tts/checkpoints/*.pth` (multi-GB model files)
- Hugging Face caches
- The full `expressive_tts/outputs/` dump (only selected clips are in `samples/`)
