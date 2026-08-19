"""TongFlow plugin for CometAPI (https://www.cometapi.com) — one key in front
of 500+ models across labs. Everything the plugin touches is OpenAI-shaped on
CometAPI's side:

- ``POST /v1/chat/completions``      text, vision, video and audio understanding
- ``POST /v1/images/generations``    text → image (sync; b64 or URL result)
- ``POST /v1/images/edits``          image edit / multi-image fusion (multipart)
- ``POST /v1/videos``                text / image(s) → video, submit + poll
- ``POST /v1/audio/speech``          TTS (binary response)
- ``POST /v1/audio/transcriptions``  Whisper-style ASR (+ verbose_json segments)

Model choice is per node: ``TONGFLOW_SLOT_MODELS`` is the curated shortlist
(first entry = default) and ``TONGFLOW_MODEL_CATALOG`` lets the canvas extend
each dropdown live from CometAPI's public catalog (``GET /api/models``).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tongflow.node_slots import NodeSlots
from tongflow.progress import progress
from tongflow.protocol import Asset, asset
from tongflow.slots import node_slot
from tongflow.models.audio_describe import AudioDescribeInput, AudioDescribeOutput
from tongflow.models.combine_text import CombineTextInput, CombineTextOutput
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.image_describe import ImageDescribeInput, ImageDescribeOutput
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_fusion import ImageFusionInput, ImageFusionOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_gen_text import ImageGenTextInput, ImageGenTextOutput
from tongflow.models.image_gen_video import ImageGenVideoInput, ImageGenVideoOutput
from tongflow.models.images_gen_video import ImagesGenVideoInput, ImagesGenVideoOutput
from tongflow.models.split_text import SplitTextInput, SplitTextOutput
from tongflow.models.text_gen_speech_instruct import (
    TextGenSpeechInstructInput,
    TextGenSpeechInstructOutput,
)
from tongflow.models.text_gen_speech_preset import (
    TextGenSpeechPresetInput,
    TextGenSpeechPresetOutput,
)
from tongflow.models.text_gen_video import TextGenVideoInput, TextGenVideoOutput
from tongflow.models.transcribe import TranscribeInput, TranscribeOutput
from tongflow.models.transcribe_timestamp import (
    TranscribeTimestampInput,
    TranscribeTimestampOutput,
    TranscribeTimestampOutputRootTimeStampsItem,
)
from tongflow.models.video_describe import VideoDescribeInput, VideoDescribeOutput
from tongflow.models.video_edit import VideoEditInput, VideoEditOutput
from tongflow.models.video_gen_text import VideoGenTextInput, VideoGenTextOutput

# ── Per-node model picker ───────────────────────────────────────────────────
# Pure dict literal read by the platform scanner via AST (no variable
# references — the text list is repeated verbatim on purpose). First entry per
# slot = default. This is only the curated shortlist; TONGFLOW_MODEL_CATALOG
# below extends each dropdown from the live catalog, and _active_model()
# accepts any id the live catalog knows for that slot.
TONGFLOW_SLOT_MODELS = {
    "gen-text": ["gemini-3.5-flash", "gpt-5.5", "claude-sonnet-5", "deepseek-v4-flash", "grok-4.6", "qwen3.7-plus", "kimi-k3", "glm-5.3"],
    "split-text": ["gemini-3.5-flash", "gpt-5.5", "claude-sonnet-5", "deepseek-v4-flash", "grok-4.6", "qwen3.7-plus", "kimi-k3", "glm-5.3"],
    "combine-text": ["gemini-3.5-flash", "gpt-5.5", "claude-sonnet-5", "deepseek-v4-flash", "grok-4.6", "qwen3.7-plus", "kimi-k3", "glm-5.3"],
    "image-describe": ["gemini-3.5-flash", "gpt-5.5", "claude-sonnet-5", "grok-4.6", "qwen3.6-plus"],
    "image-gen-text": ["gemini-3.5-flash", "gpt-5.5", "claude-sonnet-5", "grok-4.6", "qwen3.6-plus"],
    "video-describe": ["gemini-3.5-flash", "gemini-3.7-flash", "qwen3.8-max"],
    "video-gen-text": ["gemini-3.5-flash", "gemini-3.7-flash", "qwen3.8-max"],
    "audio-describe": ["gemini-3.5-flash", "gemini-3.7-flash", "gpt-audio-1.5"],
    "image-gen": ["gpt-image-2", "doubao-seedream-5-0-260128", "doubao-seedream-4-5-251128", "gpt-image-1.5", "grok-imagine-image-quality"],
    "image-edit": ["gpt-image-2", "grok-imagine-image-quality"],
    "image-fusion": ["gpt-image-2"],
    "text-gen-video": ["sora-2", "veo3.1-fast", "veo3.1", "seedance-2-5", "wan2.7", "minimax-h3", "happyhorse-1.1", "viduq3", "sora-2-pro"],
    "image-gen-video": ["sora-2", "veo3.1-fast", "veo3.1", "seedance-2-5", "wan2.7", "minimax-h3", "happyhorse-1.1", "viduq3", "sora-2-pro"],
    "images-gen-video": ["happyhorse-1.1", "happyhorse-1.0"],
    "video-edit": ["omni-fast-v2v"],
    "transcribe": ["whisper-1", "gpt-4o-transcribe"],
    "transcribe-timestamp": ["whisper-1"],
    "text-gen-speech-preset": ["gpt-4o-mini-tts", "tts-1-hd", "tts-1"],
    "text-gen-speech-instruct": ["gpt-4o-mini-tts"],
}

# Live catalog for the canvas dropdowns (public endpoint, CORS-enabled, no
# auth). A record matches a slot when every token is a substring of the named
# field (arrays/objects are JSON-serialized first). Audio slots are absent —
# CometAPI's catalog does not tag TTS/ASR endpoints, so they stay shortlist-only.
TONGFLOW_MODEL_CATALOG = {
    "url": "https://api.cometapi.com/api/models",
    "items": "data",
    "id": "id",
    "exclude": {"upcoming": True},
    "slots": {
        "gen-text": {"features": "text-to-text", "endpoints": "/v1/chat/completions"},
        "split-text": {"features": "text-to-text", "endpoints": "/v1/chat/completions"},
        "combine-text": {"features": "text-to-text", "endpoints": "/v1/chat/completions"},
        "image-describe": {"features": "image-to-text", "endpoints": "/v1/chat/completions"},
        "image-gen-text": {"features": "image-to-text", "endpoints": "/v1/chat/completions"},
        "video-describe": {"features": "video-to-text", "endpoints": "/v1/chat/completions"},
        "video-gen-text": {"features": "video-to-text", "endpoints": "/v1/chat/completions"},
        "audio-describe": {"features": "speech-to-text", "endpoints": "/v1/chat/completions"},
        "image-gen": {"features": "text-to-image", "endpoints": "/v1/images/generations"},
        "image-edit": {"endpoints": "/v1/images/edits"},
        "image-fusion": {"endpoints": "/v1/images/edits"},
        "text-gen-video": {"features": "text-to-video", "endpoints": "/v1/videos"},
        "image-gen-video": {"features": "image-to-video", "endpoints": "/v1/videos"},
        "video-edit": {"features": "video-editing", "endpoints": "/v1/videos"},
    },
}

# Plugin logs go to stderr — stdout is reserved for the ABI JSON response.
logging.basicConfig(
    level=os.environ.get("TONGFLOW_PLUGIN_LOG_LEVEL", "INFO").upper(),
    stream=sys.stderr,
    format="[cometapi] %(levelname)s %(message)s",
)
log = logging.getLogger("tongflow.plugins.cometapi")

DEFAULT_BASE_URL = "https://api.cometapi.com/v1"
DEFAULT_POLL_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 10.0
DEFAULT_VIDEO_SIZE = "1280x720"
DEFAULT_VIDEO_SECONDS = 4
DEFAULT_TTS_VOICE = "alloy"
DEFAULT_TTS_FORMAT = "mp3"

# Model chosen on the node; set by main() from the request envelope. Empty →
# each slot's default (first entry in TONGFLOW_SLOT_MODELS).
_REQUEST_MODEL: str = ""


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _require_api_key() -> str:
    api_key = _env("COMETAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "COMETAPI_KEY is not set. Create one at https://api.cometapi.com/console/token "
            "and add it in TongFlow Settings."
        )
    return api_key


def _base_url() -> str:
    return (_env("COMETAPI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _poll_timeout() -> float:
    raw = _env("COMETAPI_POLL_TIMEOUT_S")
    try:
        return float(raw) if raw else DEFAULT_POLL_TIMEOUT_S
    except ValueError:
        return DEFAULT_POLL_TIMEOUT_S


# ── Live catalog (mirrors the canvas filter, so a dropdown pick is accepted) ──

_CATALOG: List[Dict[str, Any]] | None = None


def _catalog_records() -> List[Dict[str, Any]]:
    global _CATALOG
    if _CATALOG is None:
        records: List[Dict[str, Any]] = []
        try:
            url = str(TONGFLOW_MODEL_CATALOG["url"])
            obj = json.loads(urlopen(url, timeout=15).read().decode("utf-8", errors="replace"))  # noqa: S310
            data = obj.get("data") if isinstance(obj, dict) else None
            records = [m for m in (data or []) if isinstance(m, dict)]
        except (HTTPError, URLError, ValueError, TimeoutError) as e:
            log.warning("could not fetch CometAPI catalog: %s", e)
        _CATALOG = records
    return _CATALOG


def _catalog_ids_for(slot: str) -> set[str]:
    """Ids the live catalog exposes for a slot, using the same rules the canvas applies."""
    slots = TONGFLOW_MODEL_CATALOG["slots"]
    rules = slots.get(slot) if isinstance(slots, dict) else None
    if not isinstance(rules, dict):
        return set()
    exclude = TONGFLOW_MODEL_CATALOG.get("exclude")
    exclude_items = list(exclude.items()) if isinstance(exclude, dict) else []
    ids: set[str] = set()
    for rec in _catalog_records():
        mid = rec.get("id")
        if not isinstance(mid, str) or not mid.strip():
            continue
        if any(rec.get(field) == literal for field, literal in exclude_items):
            continue
        ok = True
        for field, token in rules.items():
            val = rec.get(field)
            text = val if isinstance(val, str) else json.dumps(val) if val is not None else ""
            if str(token) not in text:
                ok = False
                break
        if ok:
            ids.add(mid)
    return ids


def _active_model(slot: str) -> str:
    models = TONGFLOW_SLOT_MODELS[slot]
    if not _REQUEST_MODEL:
        return models[0]
    if _REQUEST_MODEL in models or _REQUEST_MODEL in _catalog_ids_for(slot):
        return _REQUEST_MODEL
    raise RuntimeError(
        f"unknown model {_REQUEST_MODEL!r} for {slot} (not in the shortlist or the "
        f"live CometAPI catalog for this slot)"
    )


# ── HTTP helpers ───────────────────────────────────────────────────────────


def _headers(content_type: Optional[str] = None) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {_require_api_key()}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _http(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    content_type: Optional[str] = None,
    timeout: float = 300,
) -> Tuple[bytes, str]:
    """Raw request; returns (body, content-type). HTTP errors surface the body."""
    log.info("%s %s", method, url)
    req = Request(url, data=body, headers=_headers(content_type), method=method)
    try:
        resp = urlopen(req, timeout=timeout)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        log.error("HTTP %s on %s\nresponse body: %s", e.code, url, err_body[:2000])
        raise RuntimeError(f"HTTP {e.code} from CometAPI: {err_body[:500] or e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error contacting CometAPI: {e.reason}") from e
    return resp.read(), resp.headers.get_content_type() or ""


def _json_request(
    method: str, url: str, body: Dict[str, Any] | None = None, timeout: float = 300
) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    raw, _ = _http(method, url, body=data, content_type="application/json" if data else None, timeout=timeout)
    text = raw.decode("utf-8", errors="replace")
    obj = json.loads(text) if text.strip() else {}
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected non-object response: {text[:200]}")
    return obj


def _multipart(
    fields: Dict[str, str], files: List[Tuple[str, str, str, bytes]]
) -> Tuple[bytes, str]:
    boundary = "----tongflow" + os.urandom(16).hex()
    line = boundary.encode()
    parts: List[bytes] = []
    for name, value in fields.items():
        parts.append(b"--" + line)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode("utf-8"))
    for field, filename, mime, content in files:
        parts.append(b"--" + line)
        parts.append(
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode()
        )
        parts.append(f"Content-Type: {mime}".encode())
        parts.append(b"")
        parts.append(content)
    parts.append(b"--" + line + b"--")
    parts.append(b"")
    return b"\r\n".join(parts), f"multipart/form-data; boundary={boundary}"


def _multipart_json(url: str, fields: Dict[str, str], files: List[Tuple[str, str, str, bytes]], timeout: float = 600) -> Dict[str, Any]:
    body, ctype = _multipart(fields, files)
    raw, _ = _http("POST", url, body=body, content_type=ctype, timeout=timeout)
    text = raw.decode("utf-8", errors="replace")
    obj = json.loads(text) if text.strip() else {}
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected non-object response: {text[:200]}")
    return obj


def _raise_if_error(obj: Dict[str, Any]) -> None:
    """Some upstreams answer 200 with an `error` object; surface its message."""
    err = obj.get("error")
    if not err:
        return
    msg = err.get("message") if isinstance(err, dict) else err
    raise RuntimeError(f"CometAPI error: {str(msg)[:500]}")


def _download(url: str) -> Tuple[bytes, str]:
    try:
        resp = urlopen(url, timeout=600)  # noqa: S310
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading result: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error downloading result: {e.reason}") from e
    return resp.read(), resp.headers.get_content_type() or ""


def _asset_bytes(a: Asset) -> bytes:
    return base64.b64decode(a.bytesBase64)


def _asset_file(field: str, a: Asset, *, default_mime: str, default_ext: str) -> Tuple[str, str, str, bytes]:
    mime = (a.mime or default_mime).strip() or default_mime
    name = (a.filename or "").strip() or f"{field}.{default_ext}"
    return field, name, mime, _asset_bytes(a)


def _data_url(a: Asset, *, default_mime: str) -> str:
    mime = (a.mime or default_mime).strip() or default_mime
    return f"data:{mime};base64,{a.bytesBase64}"


# ── Chat completions (text, vision, video, audio understanding) ───────────


def _chat(slot: str, messages: List[Dict[str, Any]], **params: Any) -> str:
    body: Dict[str, Any] = {"model": _active_model(slot), "messages": messages, "stream": False}
    for key, val in params.items():
        if val is not None:
            body[key] = val
    obj = _json_request("POST", _base_url() + "/chat/completions", body)
    choices = obj.get("choices") or []
    content: Any = ""
    if choices and isinstance(choices[0], dict):
        content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        # Some gateways return content parts; keep the text ones.
        content = "".join(
            str(p.get("text") or "") for p in content if isinstance(p, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Empty completion from CometAPI: {str(obj)[:300]}")
    return content


def _text_and_image_parts(text: str, images: List[Asset]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        parts.append({"type": "image_url", "image_url": {"url": _data_url(img, default_mime="image/png")}})
    return parts


def _text_and_video_parts(text: str, video: Asset) -> List[Dict[str, Any]]:
    # OpenAI-compatible `video_url` part (the shape Gemini/Qwen accept through
    # OpenRouter-style gateways). Not in CometAPI's chat reference — verify with
    # a live key; a 4xx here means the gateway wants a different part type.
    return [
        {"type": "text", "text": text},
        {"type": "video_url", "video_url": {"url": _data_url(video, default_mime="video/mp4")}},
    ]


_AUDIO_FORMATS = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/x-wav": "wav", "audio/wave": "wav"}


def _text_and_audio_parts(text: str, audio: Asset) -> List[Dict[str, Any]]:
    mime = (audio.mime or "audio/wav").strip().lower()
    fmt = _AUDIO_FORMATS.get(mime) or mime.removeprefix("audio/") or "wav"
    return [
        {"type": "text", "text": text},
        {"type": "input_audio", "input_audio": {"data": audio.bytesBase64, "format": fmt}},
    ]


def _gemini_root() -> str:
    """Gateway root (no /v1) — the native Gemini route lives at /v1beta."""
    base = _base_url()
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _gemini_generate(model: str, system: Optional[str], text: str, media: Asset, *, default_mime: str, **params: Any) -> str:
    """Native `generateContent` with inline media. Gemini through the
    OpenAI-compatible chat route silently drops `video_url` parts on CometAPI,
    so video/audio understanding goes native whenever the model is a Gemini."""
    mime = (media.mime or default_mime).strip() or default_mime
    body: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": text},
                    {"inline_data": {"mime_type": mime, "data": media.bytesBase64}},
                ],
            }
        ]
    }
    if system and system.strip():
        body["systemInstruction"] = {"parts": [{"text": system.strip()}]}
    gen: Dict[str, Any] = {}
    if params.get("temperature") is not None:
        gen["temperature"] = params["temperature"]
    if params.get("top_p") is not None:
        gen["topP"] = params["top_p"]
    if params.get("max_tokens") is not None:
        gen["maxOutputTokens"] = params["max_tokens"]
    if gen:
        body["generationConfig"] = gen
    url = f"{_gemini_root()}/v1beta/models/{model}:generateContent"
    log.info("POST %s", url)
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"x-goog-api-key": _require_api_key(), "Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=600)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"HTTP {e.code} from CometAPI (gemini): {err_body[:500] or e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error contacting CometAPI: {e.reason}") from e
    obj = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    _raise_if_error(obj)
    cands = obj.get("candidates") or []
    parts = ((cands[0].get("content") or {}).get("parts") or []) if cands and isinstance(cands[0], dict) else []
    answer = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    if not answer.strip():
        raise RuntimeError(f"Empty Gemini answer from CometAPI: {str(obj)[:300]}")
    return answer


def _is_gemini(model: str) -> bool:
    return model.lower().startswith("gemini")


def _understand_media(slot: str, system: Optional[str], text: str, media: Asset, *, kind: str, **params: Any) -> str:
    """Video/audio → text. Gemini models go native; others get OpenAI-style
    `video_url` / `input_audio` chat parts."""
    model = _active_model(slot)
    if _is_gemini(model):
        default_mime = "video/mp4" if kind == "video" else "audio/wav"
        return _gemini_generate(model, system, text, media, default_mime=default_mime, **params)
    content = _text_and_video_parts(text, media) if kind == "video" else _text_and_audio_parts(text, media)
    return _chat(slot, _messages(system, content), **params)


def _messages(system: Optional[str], user_content: Any) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user_content})
    return messages


# ── Text slots ─────────────────────────────────────────────────────────────


@node_slot(NodeSlots.GEN_TEXT)
def gen_text(input: GenTextInput) -> GenTextOutput:
    text = (input.text or "").strip()
    if not text:
        return GenTextOutput(success=False, error="Missing input text")
    return GenTextOutput(success=True, text=_chat("gen-text", _messages(input.userPrompt, text)))


def _parse_split_texts(raw: str) -> List[str]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            _, _, s = s.partition("\n")
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    obj = json.loads(s)
    items = obj if isinstance(obj, list) else (obj.get("texts") if isinstance(obj, dict) else None)
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("LLM did not return a JSON array of strings")
    cleaned = [x.strip() for x in items if x.strip()]
    if not cleaned:
        raise ValueError("LLM returned an empty split")
    return cleaned


@node_slot(NodeSlots.SPLIT_TEXT)
def split_text(input: SplitTextInput) -> SplitTextOutput:
    instruction = (input.userPrompt or "").strip() or "Split into natural, coherent segments."
    user_message = (
        "Split the following text into multiple segments according to this instruction:\n"
        f"{instruction}\n\n"
        "Return ONLY a JSON array of strings — no prose, no markdown, no keys, no code fences. "
        "Each array element is one segment. Preserve the original wording; do not summarize.\n\n"
        f"TEXT:\n{input.text}"
    )
    raw = _chat("split-text", [{"role": "user", "content": user_message}])
    try:
        texts = _parse_split_texts(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return SplitTextOutput(success=False, error=str(e))
    return SplitTextOutput(success=True, texts=texts)


@node_slot(NodeSlots.COMBINE_TEXT)
def combine_text(input: CombineTextInput) -> CombineTextOutput:
    segments = [s for s in (input.texts or []) if isinstance(s, str) and s.strip()]
    if not segments:
        return CombineTextOutput(success=False, error="combine-text requires input texts")
    instruction = (input.userPrompt or "").strip() or "Merge the following segments into one coherent text."
    joined = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(segments))
    user_message = (
        f"{instruction}\n\n"
        "Return ONLY the merged text — no prose, no numbering, no markdown.\n\n"
        f"SEGMENTS:\n{joined}"
    )
    return CombineTextOutput(success=True, text=_chat("combine-text", [{"role": "user", "content": user_message}]))


# ── Understanding slots ────────────────────────────────────────────────────


@node_slot(NodeSlots.IMAGE_DESCRIBE)
def image_describe(input: ImageDescribeInput) -> ImageDescribeOutput:
    instruction = (
        (input.userPrompt or "").strip()
        or (input.text or "").strip()
        or "Describe this image in detail."
    )
    content = _text_and_image_parts(instruction, [input.image])
    return ImageDescribeOutput(success=True, text=_chat("image-describe", _messages(None, content)))


@node_slot(NodeSlots.IMAGE_GEN_TEXT)
def image_gen_text(input: ImageGenTextInput) -> ImageGenTextOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageGenTextOutput(success=False, error="Missing input text")
    images = [input.image] if input.image is not None else []
    answer = _chat(
        "image-gen-text",
        _messages(input.system, _text_and_image_parts(text, images)),
        temperature=input.temperature,
        top_p=input.top_p,
        max_tokens=input.max_new_tokens,
    )
    return ImageGenTextOutput(success=True, text=answer)


@node_slot(NodeSlots.VIDEO_DESCRIBE)
def video_describe(input: VideoDescribeInput) -> VideoDescribeOutput:
    instruction = (
        (input.userPrompt or "").strip()
        or (input.text or "").strip()
        or "Describe this video in detail: scenes, subjects, actions, camera, mood."
    )
    answer = _understand_media("video-describe", None, instruction, input.video, kind="video")
    return VideoDescribeOutput(success=True, text=answer)


@node_slot(NodeSlots.VIDEO_GEN_TEXT)
def video_gen_text(input: VideoGenTextInput) -> VideoGenTextOutput:
    text = (input.text or "").strip()
    if not text:
        return VideoGenTextOutput(success=False, error="Missing input text")
    answer = _understand_media(
        "video-gen-text",
        input.system,
        text,
        input.video,
        kind="video",
        temperature=input.temperature,
        top_p=input.top_p,
        max_tokens=input.max_new_tokens,
    )
    return VideoGenTextOutput(success=True, text=answer)


@node_slot(NodeSlots.AUDIO_DESCRIBE)
def audio_describe(input: AudioDescribeInput) -> AudioDescribeOutput:
    instruction = (
        (input.userPrompt or "").strip()
        or (input.text or "").strip()
        or "Describe this audio in detail (speech content, speakers, music, mood, notable events)."
    )
    answer = _understand_media("audio-describe", None, instruction, input.audio, kind="audio")
    return AudioDescribeOutput(success=True, text=answer)


# ── Images ─────────────────────────────────────────────────────────────────


def _extract_image(obj: Dict[str, Any]) -> Asset:
    _raise_if_error(obj)
    # Sync responses put the list at `data`; async-task lookups nest it one
    # level deeper (`data.data`) — accept both.
    data = obj.get("data")
    if isinstance(data, dict):
        data = data.get("data")
    entry = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    b64 = entry.get("b64_json")
    if isinstance(b64, str) and b64:
        return asset(base64.b64decode(b64), mime="image/png")
    url = entry.get("url")
    if isinstance(url, str) and url:
        content, mime = _download(url)
        return asset(content, mime=mime or "image/png")
    raise RuntimeError(f"CometAPI returned neither b64_json nor url: {str(obj)[:300]}")


@node_slot(NodeSlots.IMAGE_GEN)
def image_gen(input: ImageGenInput) -> ImageGenOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageGenOutput(success=False, error="Missing text prompt")
    body: Dict[str, Any] = {"model": _active_model("image-gen"), "prompt": text, "n": 1}
    # Only pin `size` when the node asks for one — minimum/allowed sizes differ
    # per model (Seedream 5 wants >= 1920x1920, GPT Image 1024-ish), so the
    # model's own default is the safe choice otherwise.
    if input.width and input.height:
        body["size"] = f"{input.width}x{input.height}"
    obj = _json_request("POST", _base_url() + "/images/generations", body, timeout=600)
    return ImageGenOutput(success=True, image=_extract_image(obj))


def _image_edit(slot: str, prompt: str, images: List[Asset], width: Optional[int], height: Optional[int]) -> Asset:
    fields: Dict[str, str] = {"model": _active_model(slot), "prompt": prompt, "n": "1"}
    if width and height:
        fields["size"] = f"{width}x{height}"
    # OpenAI convention: a single file goes in `image`, several in `image[]`.
    field = "image" if len(images) == 1 else "image[]"
    files = [_asset_file(field, img, default_mime="image/png", default_ext="png") for img in images]
    obj = _multipart_json(_base_url() + "/images/edits", fields, files)
    return _extract_image(obj)


@node_slot(NodeSlots.IMAGE_EDIT)
def image_edit(input: ImageEditInput) -> ImageEditOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageEditOutput(success=False, error="Missing text prompt")
    image = _image_edit("image-edit", text, [input.image], input.width, input.height)
    return ImageEditOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_FUSION)
def image_fusion(input: ImageFusionInput) -> ImageFusionOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageFusionOutput(success=False, error="Missing text prompt")
    images = list(input.images or [])
    if not images:
        return ImageFusionOutput(success=False, error="image-fusion requires at least one input image")
    image = _image_edit("image-fusion", text, images, input.width, input.height)
    return ImageFusionOutput(success=True, image=image)


# ── Videos (OpenAI Videos-API shape: multipart submit, poll, download) ─────


def _seconds(duration: Optional[float]) -> str:
    if not duration or duration <= 0:
        return str(DEFAULT_VIDEO_SECONDS)
    return str(max(1, int(round(duration))))


def _video_submit_multipart(fields: Dict[str, str], files: List[Tuple[str, str, str, bytes]]) -> str:
    obj = _multipart_json(_base_url() + "/videos", fields, files)
    return _task_id(obj)


def _task_id(obj: Dict[str, Any]) -> str:
    task_id = obj.get("id") or obj.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"CometAPI video submit returned no id: {str(obj)[:300]}")
    log.info("submitted video task %s", task_id)
    return task_id


def _video_poll(task_id: str) -> Asset:
    deadline = time.monotonic() + _poll_timeout()
    while True:
        obj = _json_request("GET", f"{_base_url()}/videos/{task_id}", timeout=60)
        _raise_if_error(obj)
        status = str(obj.get("status") or "").lower()
        if status == "completed":
            url = obj.get("video_url") or obj.get("url")
            if isinstance(url, str) and url.startswith("http"):
                content, mime = _download(url)
            else:
                content, mime = _http("GET", f"{_base_url()}/videos/{task_id}/content", timeout=600)
            return asset(content, mime=mime if mime.startswith("video/") else "video/mp4")
        if status in ("failed", "error"):
            err = obj.get("error")
            msg = err.get("message") if isinstance(err, dict) else (err or obj.get("fail_reason"))
            raise RuntimeError(f"CometAPI video {task_id} {status}: {msg or 'unknown error'}")
        pct = obj.get("progress")
        progress(
            f"CometAPI video {status or 'queued'}",
            percent=float(pct) if isinstance(pct, (int, float)) else None,
        )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"CometAPI video {task_id} did not finish within {int(_poll_timeout())}s "
                f"(last status: {status or 'unknown'})"
            )
        time.sleep(POLL_INTERVAL_S)


def _image_dims(data: bytes) -> Optional[Tuple[int, int]]:
    try:
        from io import BytesIO

        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(BytesIO(data)) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001 — Pillow missing or undecodable input
        return None


def _fit_reference(data: bytes, size: str) -> Tuple[bytes, str]:
    """Scale-and-center-crop a reference image to exactly `size` (WxH) as PNG.
    Sora rejects references whose pixels differ from the requested size; other
    routes tolerate it, and a frame that already matches is passed through."""
    try:
        from io import BytesIO

        from PIL import Image, ImageOps  # type: ignore[import-not-found]

        w, h = (int(x) for x in size.lower().split("x"))
        with Image.open(BytesIO(data)) as im:
            if (im.width, im.height) == (w, h):
                return data, "image/png" if im.format == "PNG" else f"image/{(im.format or 'png').lower()}"
            fitted = ImageOps.fit(im.convert("RGB"), (w, h), method=Image.Resampling.LANCZOS)
            buf = BytesIO()
            fitted.save(buf, format="PNG")
            return buf.getvalue(), "image/png"
    except Exception as e:  # noqa: BLE001 — fall back to the original bytes
        log.warning("could not fit reference image to %s: %s", size, e)
        return data, "image/png"


def _generate_video(
    slot: str,
    prompt: str,
    duration: Optional[float],
    width: Optional[int],
    height: Optional[int],
    images: List[Asset],
) -> Asset:
    raw_images = [_asset_bytes(img) for img in images]
    if width and height:
        size = f"{width}x{height}"
    else:
        # No explicit size: follow the first reference's orientation.
        dims = _image_dims(raw_images[0]) if raw_images else None
        size = "720x1280" if dims and dims[1] > dims[0] else DEFAULT_VIDEO_SIZE
    fields = {
        "model": _active_model(slot),
        "prompt": prompt,
        "seconds": _seconds(duration),
        "size": size,
    }
    files: List[Tuple[str, str, str, bytes]] = []
    for i, data in enumerate(raw_images):
        fitted, mime = _fit_reference(data, size)
        files.append(("input_reference", f"reference_{i}.png", mime, fitted))
    return _video_poll(_video_submit_multipart(fields, files))


@node_slot(NodeSlots.TEXT_GEN_VIDEO)
def text_gen_video(input: TextGenVideoInput) -> TextGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenVideoOutput(success=False, error="Missing text prompt")
    video = _generate_video("text-gen-video", text, input.duration, input.width, input.height, [])
    return TextGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGE_GEN_VIDEO)
def image_gen_video(input: ImageGenVideoInput) -> ImageGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageGenVideoOutput(success=False, error="Missing text prompt")
    video = _generate_video(
        "image-gen-video", text, input.duration, input.width, input.height, [input.image]
    )
    return ImageGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGES_GEN_VIDEO)
def images_gen_video(input: ImagesGenVideoInput) -> ImagesGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return ImagesGenVideoOutput(success=False, error="Missing text prompt")
    images = list(input.images or [])
    if not images:
        return ImagesGenVideoOutput(success=False, error="images-gen-video requires at least one input image")
    video = _generate_video(
        "images-gen-video", text, input.duration, input.width, input.height, images
    )
    return ImagesGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.VIDEO_EDIT)
def video_edit(input: VideoEditInput) -> VideoEditOutput:
    text = (input.text or "").strip()
    if not text:
        return VideoEditOutput(success=False, error="Missing text prompt")
    # Reference-video edits are a JSON body with an inline data URL (Omni beta).
    body: Dict[str, Any] = {
        "model": _active_model("video-edit"),
        "prompt": text,
        "video": _data_url(input.video, default_mime="video/mp4"),
        "seconds": str(DEFAULT_VIDEO_SECONDS),
    }
    obj = _json_request("POST", _base_url() + "/videos", body, timeout=600)
    return VideoEditOutput(success=True, video=_video_poll(_task_id(obj)))


# ── Audio ──────────────────────────────────────────────────────────────────


def _speech(slot: str, text: str, voice: str, instructions: Optional[str]) -> Asset:
    body: Dict[str, Any] = {
        "model": _active_model(slot),
        "input": text,
        "voice": voice,
        "response_format": DEFAULT_TTS_FORMAT,
    }
    if instructions:
        body["instructions"] = instructions
    raw, mime = _http(
        "POST",
        _base_url() + "/audio/speech",
        body=json.dumps(body).encode("utf-8"),
        content_type="application/json",
    )
    if not raw:
        raise RuntimeError("CometAPI returned empty audio")
    return asset(raw, mime=mime if mime.startswith("audio/") else "audio/mpeg")


@node_slot(NodeSlots.TEXT_GEN_SPEECH_PRESET)
def text_gen_speech_preset(input: TextGenSpeechPresetInput) -> TextGenSpeechPresetOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenSpeechPresetOutput(success=False, error="Missing input text")
    voice = (input.speaker or "").strip() or _env("COMETAPI_TTS_VOICE") or DEFAULT_TTS_VOICE
    audio = _speech("text-gen-speech-preset", text, voice, (input.instruct or "").strip() or None)
    return TextGenSpeechPresetOutput(success=True, audio=audio)


@node_slot(NodeSlots.TEXT_GEN_SPEECH_INSTRUCT)
def text_gen_speech_instruct(input: TextGenSpeechInstructInput) -> TextGenSpeechInstructOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenSpeechInstructOutput(success=False, error="Missing input text")
    voice = _env("COMETAPI_TTS_VOICE") or DEFAULT_TTS_VOICE
    audio = _speech("text-gen-speech-instruct", text, voice, (input.instruct or "").strip() or None)
    return TextGenSpeechInstructOutput(success=True, audio=audio)


def _transcribe(slot: str, audio: Asset, language: Optional[str], prompt: Optional[str], *, verbose: bool) -> Dict[str, Any]:
    fields: Dict[str, str] = {"model": _active_model(slot)}
    if language and language.strip():
        fields["language"] = language.strip()
    if prompt and prompt.strip():
        fields["prompt"] = prompt.strip()
    if verbose:
        fields["response_format"] = "verbose_json"
        fields["timestamp_granularities[]"] = "segment"
    else:
        fields["response_format"] = "json"
    files = [_asset_file("file", audio, default_mime="audio/wav", default_ext="wav")]
    return _multipart_json(_base_url() + "/audio/transcriptions", fields, files)


@node_slot(NodeSlots.TRANSCRIBE)
def transcribe(input: TranscribeInput) -> TranscribeOutput:
    obj = _transcribe("transcribe", input.audio, input.language, input.context, verbose=False)
    text = obj.get("text")
    if not isinstance(text, str):
        return TranscribeOutput(success=False, error=f"No transcript in response: {str(obj)[:200]}")
    return TranscribeOutput(success=True, text=text.strip())


@node_slot(NodeSlots.TRANSCRIBE_TIMESTAMP)
def transcribe_timestamp(input: TranscribeTimestampInput) -> TranscribeTimestampOutput:
    obj = _transcribe("transcribe-timestamp", input.audio, input.language, input.context, verbose=True)
    text = obj.get("text")
    if not isinstance(text, str):
        return TranscribeTimestampOutput(success=False, error=f"No transcript in response: {str(obj)[:200]}")
    stamps: List[TranscribeTimestampOutputRootTimeStampsItem] = []
    for seg in obj.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        seg_text = seg.get("text")
        start, end = seg.get("start"), seg.get("end")
        if not isinstance(seg_text, str) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        stamps.append(
            TranscribeTimestampOutputRootTimeStampsItem(start=float(start), end=float(end), text=seg_text.strip())
        )
    return TranscribeTimestampOutput(success=True, text=text.strip(), time_stamps=stamps)


# Runtime dispatcher. The @node_slot wrapper accepts a raw dict here (it
# deep-constructs the typed BaseModel internally) and dumps the BaseModel
# return to a dict. `Any` reflects the I/O boundary, not the plugin contract.
_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.GEN_TEXT: gen_text,
    NodeSlots.SPLIT_TEXT: split_text,
    NodeSlots.COMBINE_TEXT: combine_text,
    NodeSlots.IMAGE_DESCRIBE: image_describe,
    NodeSlots.IMAGE_GEN_TEXT: image_gen_text,
    NodeSlots.VIDEO_DESCRIBE: video_describe,
    NodeSlots.VIDEO_GEN_TEXT: video_gen_text,
    NodeSlots.AUDIO_DESCRIBE: audio_describe,
    NodeSlots.IMAGE_GEN: image_gen,
    NodeSlots.IMAGE_EDIT: image_edit,
    NodeSlots.IMAGE_FUSION: image_fusion,
    NodeSlots.TEXT_GEN_VIDEO: text_gen_video,
    NodeSlots.IMAGE_GEN_VIDEO: image_gen_video,
    NodeSlots.IMAGES_GEN_VIDEO: images_gen_video,
    NodeSlots.VIDEO_EDIT: video_edit,
    NodeSlots.TRANSCRIBE: transcribe,
    NodeSlots.TRANSCRIBE_TIMESTAMP: transcribe_timestamp,
    NodeSlots.TEXT_GEN_SPEECH_PRESET: text_gen_speech_preset,
    NodeSlots.TEXT_GEN_SPEECH_INSTRUCT: text_gen_speech_instruct,
}


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    global _REQUEST_MODEL
    try:
        if set(TONGFLOW_SLOT_MODELS) != {str(s) for s in _SLOT_HANDLERS}:
            raise RuntimeError("TONGFLOW_SLOT_MODELS drifted from _SLOT_HANDLERS — keep them in sync")
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        if not isinstance(req, dict):
            req = {}
        prompt = req.get("prompt")
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "")
        _REQUEST_MODEL = str(req.get("model") or "").strip()

        handler = _SLOT_HANDLERS.get(slot)
        if handler is None:
            raise RuntimeError(f"unsupported nodeSlot: {slot!r}")
        out = handler(prompt)
    except Exception as e:  # noqa: BLE001 — surfaced as ABI failure
        _write({"success": False, "error": str(e)})
        return 1

    _write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
