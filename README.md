# tongflow-router-cometapi

Official [TongFlow](https://github.com/tong-io/tongflow) plugin for [CometAPI](https://www.cometapi.com) — one API key in front of 500+ models (OpenAI, Anthropic, Google, xAI, DeepSeek, Qwen, ByteDance, Kling, Sora, Veo, Wan, MiniMax …) behind OpenAI-compatible routes.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU). Every slot has a per-node **model picker**:

| Area | Slots | Default model | Route |
| --- | --- | --- | --- |
| Text | `gen-text`, `split-text`, `combine-text` | `gemini-3.5-flash` | `POST /v1/chat/completions` |
| Vision | `image-describe`, `image-gen-text` | `gemini-3.5-flash` | chat, `image_url` data URI |
| Video / audio understanding | `video-describe`, `video-gen-text`, `audio-describe` | `gemini-3.5-flash` | Gemini models: native `POST /v1beta/models/{model}:generateContent` (inline media); others: chat `video_url` / `input_audio` parts |
| Image | `image-gen` | `gpt-image-2` | `POST /v1/images/generations` |
| Image | `image-edit`, `image-fusion` | `gpt-image-2` | `POST /v1/images/edits` (multipart) |
| Video | `text-gen-video`, `image-gen-video`, `images-gen-video` | `sora-2` / `happyhorse-1.1` | `POST /v1/videos` → poll `GET /v1/videos/{id}` |
| Video | `video-edit` | `omni-fast-v2v` (beta) | `POST /v1/videos` (JSON, inline MP4) |
| Speech | `text-gen-speech-preset`, `text-gen-speech-instruct` | `gpt-4o-mini-tts` | `POST /v1/audio/speech` |
| Transcription | `transcribe`, `transcribe-timestamp` | `whisper-1` | `POST /v1/audio/transcriptions` |

### Live model list

The dropdown shows a curated shortlist (`TONGFLOW_SLOT_MODELS`, first entry = default) **plus** whatever CometAPI's public catalog (`GET https://api.cometapi.com/api/models`, no auth) currently exposes for that slot. The canvas fetches the catalog in the browser and filters it per slot with the rules in `TONGFLOW_MODEL_CATALOG` (e.g. `gen-text` = `features` contains `text-to-text` **and** `endpoints` contains `/v1/chat/completions`; upcoming models are hidden). The plugin applies the same rules at run time, so any id you pick from the dropdown is accepted. TTS / ASR models are shortlist-only — the catalog does not tag those endpoints.

Video sizes and durations are passed through as `size` (`WxH`; default `1280x720`, or `720x1280` when the first reference image is portrait) and `seconds` (rounded, default `4`); each model accepts a different set (Sora: 4/8/12 s, Veo 3.1: 4/6/8 s, Wan 2.7: 2–15 s …) and CometAPI returns a clear 4xx for unsupported values. Reference images are scale-and-center-cropped to the requested size (Pillow) because Sora rejects mismatched frames. Image generation sends `size` only when the node sets width/height — minimum sizes differ per model (Seedream 5 needs ≥ 1920×1920). Gemini through the OpenAI-compatible chat route drops `video_url` parts on CometAPI, so video/audio understanding goes through the native Gemini route for `gemini-*` models.

All 19 slots were exercised against the live gateway (2026-08-19): text, vision, video/audio understanding, GPT Image 2 + Seedream generation, edit + fusion, Veo 3.1 / Sora 2 / HappyHorse / Omni video, TTS, Whisper.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `COMETAPI_KEY` | ✅ | Create one in the [CometAPI console](https://api.cometapi.com/console/token). |
| `COMETAPI_BASE_URL` | optional | Override the default `https://api.cometapi.com/v1`. |
| `COMETAPI_POLL_TIMEOUT_S` | optional | Max seconds to wait for an async video task (default `900`). |
| `COMETAPI_TTS_VOICE` | optional | Default TTS voice when the node sets no speaker (default `alloy`). |

Values are stored locally and take effect without a restart. `requirements.txt` pulls in Pillow (reference-image fitting); TongFlow installs it into the plugin venv automatically.

## Smoke test

```bash
cd plugins/tongflow-router-cometapi
echo '{"nodeSlot":"gen-text","model":"gemini-3.5-flash","prompt":{"text":"Say hi in five words."}}' \
  | COMETAPI_KEY=sk-... PYTHONPATH=../../sdk python entry.py
```

Plugin logs go to stderr; stdout is the single ABI JSON response.

## Getting help with CometAPI

Questions about CometAPI itself — pricing, model coverage, quota — go to CometAPI, not this repo: [emery@askcometapi.com](mailto:emery@askcometapi.com). Mention you came from TongFlow and they can set you up with starter credits.

Bugs in this plugin (a slot failing, a wrong request shape) belong in [issues](https://github.com/tong-io/tongflow-router-cometapi/issues).
