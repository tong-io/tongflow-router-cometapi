# tongflow-router-cometapi

Official [TongFlow](https://github.com/tong-io/tongflow) plugin for [CometAPI](https://www.cometapi.com) — one API key in front of 500+ models (OpenAI, Anthropic, Google, xAI, DeepSeek, Qwen, ByteDance, Kling, Sora, Veo, Wan, MiniMax …) behind OpenAI-compatible routes.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU). Every slot has a per-node **model picker**:

| Area | Slots | Default model | Route |
| --- | --- | --- | --- |
| Text | `gen-text`, `split-text`, `combine-text` | `gemini-3.5-flash` | `POST /v1/chat/completions` |
| Vision | `image-describe`, `image-gen-text` | `gemini-3.5-flash` | chat, `image_url` data URI |
| Video / audio understanding | `video-describe`, `video-gen-text`, `audio-describe` | `gemini-3.5-flash` | chat, `video_url` / `input_audio` parts |
| Image | `image-gen` | `gpt-image-2` | `POST /v1/images/generations` |
| Image | `image-edit`, `image-fusion` | `gpt-image-2` | `POST /v1/images/edits` (multipart) |
| Video | `text-gen-video`, `image-gen-video`, `images-gen-video` | `sora-2` / `happyhorse-1.1` | `POST /v1/videos` → poll `GET /v1/videos/{id}` |
| Video | `video-edit` | `omni-fast-v2v` (beta) | `POST /v1/videos` (JSON, inline MP4) |
| Speech | `text-gen-speech-preset`, `text-gen-speech-instruct` | `gpt-4o-mini-tts` | `POST /v1/audio/speech` |
| Transcription | `transcribe`, `transcribe-timestamp` | `whisper-1` | `POST /v1/audio/transcriptions` |

### Live model list

The dropdown shows a curated shortlist (`TONGFLOW_SLOT_MODELS`, first entry = default) **plus** whatever CometAPI's public catalog (`GET https://api.cometapi.com/api/models`, no auth) currently exposes for that slot. The canvas fetches the catalog in the browser and filters it per slot with the rules in `TONGFLOW_MODEL_CATALOG` (e.g. `gen-text` = `features` contains `text-to-text` **and** `endpoints` contains `/v1/chat/completions`; upcoming models are hidden). The plugin applies the same rules at run time, so any id you pick from the dropdown is accepted. TTS / ASR models are shortlist-only — the catalog does not tag those endpoints.

Video sizes and durations are passed through as `size` (`WxH`, default `1280x720`) and `seconds` (rounded, default `4`); each model accepts a different set (Sora: 4/8/12 s, Veo 3.1: 4/6/8 s, Wan 2.7: 2–15 s …) and CometAPI returns a clear 4xx for unsupported values. The `video_url` / `input_audio` chat parts follow the usual OpenAI-compatible convention but are not in CometAPI's chat reference — verify the video / audio understanding slots with a live key.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `COMETAPI_KEY` | ✅ | Create one in the [CometAPI console](https://api.cometapi.com/console/token). |
| `COMETAPI_BASE_URL` | optional | Override the default `https://api.cometapi.com/v1`. |
| `COMETAPI_POLL_TIMEOUT_S` | optional | Max seconds to wait for an async video task (default `900`). |
| `COMETAPI_TTS_VOICE` | optional | Default TTS voice when the node sets no speaker (default `alloy`). |

Values are stored locally and take effect without a restart.

## Smoke test

```bash
cd plugins/tongflow-router-cometapi
echo '{"nodeSlot":"gen-text","model":"gemini-3.5-flash","prompt":{"text":"Say hi in five words."}}' \
  | COMETAPI_KEY=sk-... PYTHONPATH=../../sdk python entry.py
```

Plugin logs go to stderr; stdout is the single ABI JSON response.
