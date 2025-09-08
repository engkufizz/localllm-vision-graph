import os
import uuid
from typing import Any, Dict, List, Union

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

LMSTUDIO_API_BASE = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "").strip()
TIMEOUT = int(os.getenv("PROXY_TIMEOUT", "300"))
MODEL_NAME = os.getenv("LMSTUDIO_PROXY_MODEL_NAME", "lmstudio-proxy")

app = FastAPI(title="LM Studio Vision Proxy (OpenAI-compatible)")

def _session():
    s = requests.Session()
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "LMStudio-Vision-Proxy/1.0"}
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"
    s.headers.update(headers)
    return s

def looks_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:") and ";base64," in s

def is_http_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def to_data_url_from_base64(raw_b64: str, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{raw_b64}"

def prepare_image_urls(images_in: List[Union[str, Dict[str, str]]]) -> List[str]:
    """
    Normalise a mixed list of image descriptors into URL strings (data URLs or http URLs).
    Supports dicts like {"id": "...", "data": "<data-url or raw b64>"} and plain strings.
    """
    urls: List[str] = []
    for x in images_in or []:
        if isinstance(x, dict):
            data = x.get("data", "")
            if isinstance(data, str) and data:
                url = data if looks_data_url(data) else to_data_url_from_base64(data)
                urls.append(url)
        elif isinstance(x, str):
            if looks_data_url(x) or is_http_url(x):
                urls.append(x)
            else:
                urls.append(to_data_url_from_base64(x))
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

def ensure_multimodal_message(messages: List[Dict[str, Any]], image_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Insert image_url items into the last user message. If no user message exists, create one.
    Converts string content to list-of-blocks when needed.
    """
    if not image_urls:
        return messages

    # Find last user message
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_idx = i
            break

    img_blocks = [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]

    if user_idx is None:
        # No user message; create one with only images
        messages = list(messages) + [{"role": "user", "content": img_blocks}]
        return messages

    # Ensure content is a list of blocks
    user_msg = dict(messages[user_idx])
    content = user_msg.get("content", "")
    if isinstance(content, str):
        content_blocks = []
        if content.strip():
            content_blocks.append({"type": "text", "text": content})
        content_blocks.extend(img_blocks)
        user_msg["content"] = content_blocks
    elif isinstance(content, list):
        # Append image blocks
        user_msg["content"] = content + img_blocks
    else:
        # Unknown type; replace with just images
        user_msg["content"] = img_blocks

    messages = list(messages)
    messages[user_idx] = user_msg
    return messages

@app.get("/health")
def health():
    return {"ok": True, "upstream": LMSTUDIO_API_BASE, "model": MODEL_NAME}

@app.get("/v1/models")
def list_models():
    s = _session()
    try:
        r = s.get(f"{LMSTUDIO_API_BASE}/models", timeout=30)
        r.raise_for_status()
        return JSONResponse(content=r.json())
    except Exception as e:
        return JSONResponse(
            content={"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}], "note": f"fallback: {e}"},
            status_code=200,
        )

@app.post("/v1/chat/completions")
def chat_completions(body: Dict[str, Any]):
    """
    Accepts OpenAI chat.completions with optional top-level:
      - images: [{"id": "...", "data": "<data-url or raw b64>"} | "<data-url or http url>" ]
      - allImages: same shape
    Converts to OpenAI multimodal message content for LM Studio vision models.
    """
    want_stream = bool(body.get("stream", False))
    messages = body.get("messages") or []
    images_in = body.get("images") or body.get("allImages") or []

    image_urls = prepare_image_urls(images_in)
    messages_mm = ensure_multimodal_message(messages, image_urls)

    upstream_payload = dict(body)
    upstream_payload["messages"] = messages_mm
    # Leave other fields (model, temperature, etc.) intact

    s = _session()
    url = f"{LMSTUDIO_API_BASE}/chat/completions"
    try:
        if want_stream:
            r = s.post(url, json=upstream_payload, stream=True, timeout=TIMEOUT)
            r.raise_for_status()

            def gen():
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
            media_type = r.headers.get("Content-Type", "text/event-stream")
            return StreamingResponse(gen(), media_type=media_type or "text/event-stream")
        else:
            r = s.post(url, json=upstream_payload, timeout=TIMEOUT)
            r.raise_for_status()
            j = r.json()
            # Ensure id/model fields exist
            if isinstance(j, dict):
                j.setdefault("id", f"chatcmpl-{uuid.uuid4().hex[:24]}")
                j.setdefault("model", body.get("model") or "local")
            return JSONResponse(content=j)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        try:
            err = e.response.json()
        except Exception:
            err = {"error": {"message": str(e)}}
        return JSONResponse(status_code=status, content=err)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})
