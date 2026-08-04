"""Otak agent. Dua backend: Ollama lokal (default) atau Claude API.

Keduanya punya antarmuka sama — `chat()` dan `chat_stream()` — jadi ganti
backend nggak nyentuh kode pemanggilnya.

Riwayat cuma hidup selama proses jalan. Nggak disimpen ke disk, dan itu
disengaja: riwayat yang bertahan antar-sesi berkali-kali bikin model ngarang
soal data yang udah berubah, sampai ngaku udah ngerjain sesuatu yang nggak
pernah dikerjain. Nyambungin obrolan itu berguna, tapi bukan sumber fakta.
"""

from __future__ import annotations

import logging
import re

from . import config

log = logging.getLogger(__name__)

# Tag <think> dari model reasoning, dan markup yang nggak enak dibacakan.
_TAG = re.compile(r"<think>.*?</think>", re.S | re.I)
_MARKUP = re.compile(r"[*_`#]+")


def _clean_for_speech(text: str) -> str:
    text = _TAG.sub(" ", text)
    text = _MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# Titik yang BUKAN akhir kalimat. Tanpa ini, "3 p.m." — bentuk yang justru
# dikeluarkan Parakeet — kepotong jadi dua, dan TTS ngomong "three pee." terus
# berhenti sejenak sebelum "em."
_BUKAN_AKHIR = re.compile(
    r"(?:\b(?:[a-z]|mr|mrs|ms|dr|prof|st|vs|etc|e\.g|i\.e|a\.m|p\.m|no)\.)\s*$",
    re.I,
)
_AKHIR_KALIMAT = re.compile(r"[.!?]+[\"')\]]*\s")


def _potong_kalimat(buf: str) -> tuple[str | None, str]:
    """Ambil satu kalimat utuh dari depan `buf`.

    Balikin `(kalimat, sisa)`, atau `(None, buf)` kalau belum ada kalimat utuh.
    """
    for m in _AKHIR_KALIMAT.finditer(buf):
        potong = buf[: m.end()]
        if _BUKAN_AKHIR.search(potong):
            continue  # singkatan, bukan akhir kalimat
        return potong.strip(), buf[m.end() :]
    return None, buf


class _BaseConversation:
    name = "?"

    def __init__(self, system_prompt: str | None = None) -> None:
        self.base_prompt = system_prompt or config.SYSTEM_PROMPT
        self.messages: list[dict] = []

    @property
    def system_prompt(self) -> str:
        bagian = [self.base_prompt]
        if config.REPLY_MAX_WORDS > 0:
            bagian.append(
                f"HARD LIMIT: answer in {config.REPLY_MAX_WORDS} words or fewer. "
                "Never exceed it. Give the single most useful fact; if they want "
                "more, they will ask.\n"
                # Panjang kalimat diatur terpisah dari panjang jawaban: satu
                # kalimat panjang nunda bunyi pertama, karena TTS baru mulai
                # setelah kalimatnya utuh.
                "Use SHORT sentences, at most 12 words each. Two short sentences "
                "beat one long one."
            )
        return "\n\n".join(bagian)

    def reset(self) -> None:
        self.messages = []

    def _trim(self) -> None:
        limit = config.MAX_HISTORY_MESSAGES
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def chat(self, text: str) -> str:
        raise NotImplementedError

    def chat_stream(self, text: str):
        """Sama kayak chat(), tapi ngeluarin KALIMAT satu per satu pas jadi.

        Default-nya ngeluarin satu potong utuh, jadi backend yang nggak dukung
        streaming tetep jalan tanpa pemanggilnya perlu tau bedanya.
        """
        yield self.chat(text)


class OllamaConversation(_BaseConversation):
    """Ollama lokal. Gratis, jalan offline."""

    name = "ollama"

    def _payload(self, stream: bool) -> dict:
        return {
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "system", "content": self.system_prompt}]
            + self.messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "num_predict": config.OLLAMA_NUM_PREDICT,
                "num_ctx": config.OLLAMA_NUM_CTX,
            },
        }

    def chat(self, text: str) -> str:
        import requests

        self.messages.append({"role": "user", "content": text})
        try:
            resp = requests.post(
                config.OLLAMA_CHAT_URL,
                json=self._payload(stream=False),
                timeout=config.OLLAMA_TIMEOUT,
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

    def chat_stream(self, text: str):
        """Keluarin kalimat begitu jadi, jangan nunggu paragrafnya selesai.

        Terukur: waktu sampai bunyi pertama turun 53-71%. Yang dipangkas bukan
        waktu totalnya, tapi diamnya — dan itu yang bikin percakapan kerasa
        hidup atau kerasa nge-lag.
        """
        import json as _json

        import requests

        self.messages.append({"role": "user", "content": text})
        penuh: list[str] = []
        sisa = ""
        try:
            with requests.post(
                config.OLLAMA_CHAT_URL,
                json=self._payload(stream=True),
                timeout=config.OLLAMA_TIMEOUT,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for baris in resp.iter_lines():
                    if not baris:
                        continue
                    potong = (_json.loads(baris).get("message") or {}).get("content", "")
                    if not potong:
                        continue
                    penuh.append(potong)
                    sisa += potong
                    # Nunggu kalimat utuh itu sengaja: ngirim potongan setengah
                    # kalimat ke TTS bikin intonasinya ngawur dan jedanya
                    # kedengeran di tempat yang salah.
                    while True:
                        kal, sisa_baru = _potong_kalimat(sisa)
                        if kal is None:
                            break
                        sisa = sisa_baru
                        bersih = _clean_for_speech(kal)
                        if bersih:
                            yield bersih
        except Exception:
            self.messages.pop()
            raise

        ekor = _clean_for_speech(sisa)
        if ekor:
            yield ekor

        reply = "".join(penuh).strip()
        if not reply:
            self.messages.pop()
            raise RuntimeError("Ollama balikin respons kosong")

        self.messages.append({"role": "assistant", "content": reply})
        self._trim()
        log.info("LLM ollama stream (%d chars): %s", len(reply), reply)


class ClaudeConversation(_BaseConversation):
    """Claude API. Lebih pintar, tapi butuh jaringan & kunci berbayar."""

    name = "claude"

    def __init__(self, system_prompt: str | None = None) -> None:
        super().__init__(system_prompt)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            if not (config.ANTHROPIC_API_KEY or config.ANTHROPIC_AUTH_TOKEN):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY belum diisi. Tambahin barisnya di file .env "
                    "(lihat .env.example), atau pakai LLM_BACKEND=ollama."
                )
            self._client = anthropic.Anthropic(
                api_key=config.ANTHROPIC_API_KEY or None,
                auth_token=config.ANTHROPIC_AUTH_TOKEN or None,
                timeout=config.CLAUDE_TIMEOUT,
            )
        return self._client

    def _kwargs(self) -> dict:
        kwargs: dict = {}
        if config.CLAUDE_EFFORT:
            kwargs["effort"] = config.CLAUDE_EFFORT
        if config.CLAUDE_THINKING in ("disabled", "adaptive"):
            kwargs["thinking"] = {"type": config.CLAUDE_THINKING}
        return kwargs

    def chat(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        try:
            response = self._get_client().messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                system=self.system_prompt,
                messages=self.messages,
                **self._kwargs(),
            )
        except Exception:
            self.messages.pop()
            raise

        reply = " ".join(b.text for b in response.content if b.type == "text").strip()
        if not reply:
            self.messages.pop()
            raise RuntimeError("Claude balikin respons kosong")

        self.messages.append({"role": "assistant", "content": reply})
        self._trim()
        log.info("LLM claude (%d chars): %s", len(reply), reply)
        return _clean_for_speech(reply)


_conversation: _BaseConversation | None = None


def get_conversation() -> _BaseConversation:
    global _conversation
    if _conversation is None:
        if config.LLM_BACKEND == "ollama":
            _conversation = OllamaConversation()
        elif config.LLM_BACKEND == "claude":
            _conversation = ClaudeConversation()
        else:
            log.warning(
                "LLM_BACKEND '%s' nggak dikenal, balik ke 'ollama'", config.LLM_BACKEND
            )
            _conversation = OllamaConversation()
        log.info("Backend LLM: %s", _conversation.name)
    return _conversation


def chat(text: str) -> str:
    return get_conversation().chat(text)


if __name__ == "__main__":
    #     .venv-agent\Scripts\python.exe -m agent.llm "halo"
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pesan = " ".join(sys.argv[1:]) or "Hello, can you hear me?"
    print(chat(pesan))
