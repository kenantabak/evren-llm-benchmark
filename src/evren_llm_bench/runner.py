"""Environment-variable-driven client for EVREN LLM API (OpenAI-compatible + OCR).

Configuration via environment variables (never hardcode secrets):
  - ``EVREN_LLM_BASE_URL`` (default: ``https://evren-llmapi.ssyz.org.tr/v1``)
  - ``EVREN_LLM_API_KEY`` (single key fallback)
  - ``EVREN_LLM_API_KEY_REASONING_OFF`` (optional dedicated key for thinking=off)
  - ``EVREN_LLM_API_KEY_REASONING_ON`` (optional dedicated key for thinking=on)
"""

from __future__ import annotations

import http.client
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


DEFAULT_BASE_URL = "https://evren-llmapi.ssyz.org.tr/v1"


def build_ocr_multipart(
    file_path: str,
    model: str,
    boundary: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Build a ``multipart/form-data`` request body for ``POST /v1/ocr``."""
    b = boundary or f"evrenocr{uuid.uuid4().hex}"
    filename = os.path.basename(file_path)
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        raw = f.read()

    parts = [
        f"--{b}\r\n".encode("utf-8"),
        b'Content-Disposition: form-data; name="model"\r\n\r\n',
        f"{model}\r\n".encode("utf-8"),
        f"--{b}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(
            "utf-8"
        ),
        f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
        raw,
        b"\r\n",
        f"--{b}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={b}"


class EvrenLLMRunner:
    """Sequential single-stream runner supporting SSE chat completions and OCR."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_reasoning_off: Optional[str] = None,
        api_key_reasoning_on: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("EVREN_LLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        fallback = api_key or os.environ.get("EVREN_LLM_API_KEY", "")
        self.key_off = (
            api_key_reasoning_off
            or os.environ.get("EVREN_LLM_API_KEY_REASONING_OFF")
            or fallback
        )
        self.key_on = (
            api_key_reasoning_on
            or os.environ.get("EVREN_LLM_API_KEY_REASONING_ON")
            or fallback
        )

    def resolve_key(self, mode: str = "reasoning_off") -> str:
        key = self.key_on if mode == "reasoning_on" else self.key_off
        if not key:
            raise ValueError(
                "Missing EVREN API key. Set EVREN_LLM_API_KEY "
                "(or EVREN_LLM_API_KEY_REASONING_OFF / EVREN_LLM_API_KEY_REASONING_ON)."
            )
        return key

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        mode: str = "reasoning_off",
        max_tokens: int = 16384,
        wall_cap_s: int = 420,
        temperature: float = 0.0,
        extra_body: Optional[Dict[str, Any]] = None,
        max_attempts: int = 4,
    ) -> Dict[str, Any]:
        """Call ``POST /v1/chat/completions`` with ``stream=True`` and wall-cap protection."""
        api_key = self.resolve_key(mode)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            payload.update(extra_body)

        parsed = urlparse(f"{self.base_url}/chat/completions")
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        retries = 0
        last_status = 0
        last_err: Optional[str] = None
        for attempt in range(max_attempts):
            if attempt > 0:
                retries += 1
            t0 = time.perf_counter()
            content_parts: List[str] = []
            reasoning_parts: List[str] = []
            finish_reason: Optional[str] = None
            routed_model: Optional[str] = None
            usage: Dict[str, Any] = {}
            err: Optional[str] = None
            conn = None
            try:
                conn_cls = (
                    http.client.HTTPSConnection
                    if parsed.scheme == "https"
                    else http.client.HTTPConnection
                )
                conn = conn_cls(
                    parsed.hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    timeout=120,
                )
                conn.request("POST", parsed.path, body=body, headers=headers)
                resp = conn.getresponse()
                last_status = resp.status
                if resp.status in (429, 502, 503, 504):
                    err_txt = resp.read().decode("utf-8", "ignore")[:240]
                    conn.close()
                    last_err = f"HTTP {resp.status}: {err_txt}"
                    time.sleep(8 * (attempt + 1))
                    continue
                if resp.status != 200:
                    err_txt = resp.read().decode("utf-8", "ignore")[:300]
                    conn.close()
                    return {
                        "ok_http": False,
                        "status_code": resp.status,
                        "retries": retries,
                        "content": "",
                        "reasoning_content": "",
                        "finish_reason": f"http_{resp.status}",
                        "routed_model": None,
                        "latency_s": round(time.perf_counter() - t0, 2),
                        "usage": {},
                        "error": err_txt,
                    }

                while True:
                    if time.perf_counter() - t0 > wall_cap_s:
                        finish_reason = "wall_cap"
                        err = f"wall_cap_{wall_cap_s}s"
                        break
                    chunk = resp.readline()
                    if not chunk:
                        break
                    line = chunk.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if event.get("model") and not routed_model:
                        routed_model = event["model"]
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning_parts.append(delta["reasoning_content"])
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                conn.close()
                return {
                    "ok_http": True,
                    "status_code": 200,
                    "retries": retries,
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                    "finish_reason": finish_reason or "stop",
                    "routed_model": routed_model,
                    "latency_s": round(time.perf_counter() - t0, 2),
                    "usage": usage,
                    "error": err,
                }
            except Exception as exc:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                last_err = f"{type(exc).__name__}: {str(exc)[:180]}"
                if time.perf_counter() - t0 > wall_cap_s:
                    return {
                        "ok_http": False,
                        "status_code": last_status,
                        "retries": retries,
                        "content": "".join(content_parts),
                        "reasoning_content": "".join(reasoning_parts),
                        "finish_reason": "wall_cap",
                        "routed_model": routed_model,
                        "latency_s": round(time.perf_counter() - t0, 2),
                        "usage": usage,
                        "error": last_err,
                    }
                time.sleep(6 * (attempt + 1))

        return {
            "ok_http": False,
            "status_code": last_status,
            "retries": retries,
            "content": "",
            "reasoning_content": "",
            "finish_reason": "error",
            "routed_model": None,
            "latency_s": float(wall_cap_s),
            "usage": {},
            "error": last_err or "max_retries_exceeded",
        }

    def ocr(
        self,
        file_path: str,
        model: str = "deepseek-ocr-2",
        timeout_s: int = 120,
    ) -> Dict[str, Any]:
        """Call ``POST /v1/ocr`` using ``multipart/form-data``."""
        api_key = self.resolve_key("reasoning_off")
        body, content_type = build_ocr_multipart(file_path, model=model)
        req = urllib.request.Request(
            f"{self.base_url}/ocr",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": content_type,
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", "ignore")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"text": raw}
                return {
                    "ok_http": True,
                    "status_code": resp.status,
                    "latency_s": round(time.perf_counter() - t0, 2),
                    "model": parsed.get("model", model),
                    "text": parsed.get("text", raw),
                }
        except urllib.error.HTTPError as exc:
            return {
                "ok_http": False,
                "status_code": exc.code,
                "latency_s": round(time.perf_counter() - t0, 2),
                "model": model,
                "text": "",
                "error": exc.read().decode("utf-8", "ignore")[:300],
            }
