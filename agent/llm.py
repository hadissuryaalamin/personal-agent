"""Otak agent. Dua backend: Claude API (default) atau Ollama lokal.

Keduanya punya antarmuka sama: `.chat(text) -> str` dan `.reset()`, jadi bisa
ditukar lewat LLM_BACKEND di .env tanpa nyentuh kode lain.
"""

from __future__ import annotations

import logging
import re

from . import config

log = logging.getLogger(__name__)

# Model yang nolak parameter `effort` (400 kalau dikirim). Haiku & Sonnet 4.5
# generasi lama nggak punya kontrol effort sama sekali.
_TANPA_EFFORT = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-haiku-3")

# Buang sisa markup yang kelewat: tag XML, bintang markdown, backtick.
# Penting karena teksnya dibacakan speaker — "asterisk asterisk" itu ganggu.
_TAG = re.compile(r"<[^>]{1,40}>")
_MARKUP = re.compile(r"[*_`#]+")


def _clean_for_speech(text: str) -> str:
    text = _TAG.sub(" ", text)
    text = _MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class _BaseConversation:
    """Riwayat chat ephemeral. Hilang pas proses mati.

    TODO(tahap 2): persist ke disk biar nyambung antar sesi.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.messages: list[dict[str, str]] = []

    def reset(self) -> None:
        self.messages = []

    def _trim(self) -> None:
        limit = config.MAX_HISTORY_MESSAGES
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def chat(self, text: str) -> str:
        raise NotImplementedError


class ClaudeConversation(_BaseConversation):
    """Claude API. Butuh ANTHROPIC_API_KEY di environment atau .env."""

    name = "claude"

    def __init__(self, system_prompt: str | None = None) -> None:
        super().__init__(system_prompt)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        import os

        import anthropic  # import lokal: cuma kepake kalau backend claude

        # SDK-nya baru ngeluh pas request, jadi dicek duluan biar pesannya jelas
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                "ANTHROPIC_API_KEY belum diisi. Tambahin barisnya di file .env "
                "(lihat .env.example), atau pakai otak lokal: LLM_BACKEND=ollama"
            )

        # Kunci diambil otomatis dari ANTHROPIC_API_KEY (load_dotenv di config
        # udah masukin isi .env ke os.environ).
        self._client = anthropic.Anthropic(timeout=config.CLAUDE_TIMEOUT)
        log.info("Client Claude siap (model=%s)", config.CLAUDE_MODEL)
        return self._client

    def chat(self, text: str) -> str:
        import anthropic

        client = self._get_client()
        self.messages.append({"role": "user", "content": text})

        kwargs = {}
        # Effort rendah = balasan cepet. Buat obrolan 1-2 kalimat, mikir dalam
        # nggak nambah kualitas tapi nambah jeda. Model lama nolak parameternya.
        if config.CLAUDE_EFFORT and not config.CLAUDE_MODEL.startswith(_TANPA_EFFORT):
            kwargs["output_config"] = {"effort": config.CLAUDE_EFFORT}

        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                system=self.system_prompt,
                messages=self.messages,
                **kwargs,
            )
        except anthropic.AuthenticationError:
            self.messages.pop()
            raise RuntimeError(
                "ANTHROPIC_API_KEY salah atau belum diisi. Taruh di file .env."
            ) from None
        except Exception:
            self.messages.pop()
            raise

        if response.stop_reason == "refusal":
            self.messages.pop()
            raise RuntimeError("Claude nolak jawab permintaan ini.")

        reply = " ".join(
            b.text for b in response.content if b.type == "text" and b.text
        ).strip()
        if not reply:
            self.messages.pop()
            raise RuntimeError(f"Claude balikin respons kosong ({response.stop_reason})")

        self.messages.append({"role": "assistant", "content": reply})
        self._trim()
        log.info(
            "LLM claude (%d in / %d out): %s",
            response.usage.input_tokens,
            response.usage.output_tokens,
            reply,
        )
        return _clean_for_speech(reply)


class OllamaConversation(_BaseConversation):
    """Ollama lokal. Gratis, jalan offline, tapi lebih lemot & kurang pintar."""

    name = "ollama"

    def chat(self, text: str) -> str:
        import requests

        self.messages.append({"role": "user", "content": text})
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "system", "content": self.system_prompt}]
            + self.messages,
            "stream": False,
            "options": {"temperature": 0.7},
        }

        try:
            resp = requests.post(
                config.OLLAMA_CHAT_URL, json=payload, timeout=config.OLLAMA_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            self.messages.pop()
            raise

        reply = (data.get("message") or {}).get("content", "").strip()
        if not reply:
            self.messages.pop()
            raise RuntimeError(f"Ollama balikin respons kosong: {data!r}")

        self.messages.append({"role": "assistant", "content": reply})
        self._trim()
        log.info("LLM ollama (%d chars): %s", len(reply), reply)
        return _clean_for_speech(reply)


_conversation: _BaseConversation | None = None


def get_conversation() -> _BaseConversation:
    global _conversation
    if _conversation is None:
        if config.LLM_BACKEND == "ollama":
            _conversation = OllamaConversation()
        else:
            if config.LLM_BACKEND != "claude":
                log.warning(
                    "LLM_BACKEND '%s' nggak dikenal, balik ke 'claude'",
                    config.LLM_BACKEND,
                )
            _conversation = ClaudeConversation()
        log.info("Backend LLM: %s", _conversation.name)
    return _conversation


def chat(text: str) -> str:
    return get_conversation().chat(text)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    prompt = " ".join(sys.argv[1:]) or "Halo, kenalin diri kamu dong."
    print(f"> {prompt}")
    print(chat(prompt))
