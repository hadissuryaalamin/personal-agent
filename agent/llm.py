"""The agent's brain. Two backends: local Ollama (default) or the Claude API.

Both share one interface — `chat()` and `chat_stream()` — so swapping backends
never touches the calling code.

History lives only as long as the process. It is deliberately not written to
disk: history that survived between sessions repeatedly led the model to invent
things about data that had since changed, up to claiming it had done work it
never did. Continuing a conversation is useful; treating it as a source of fact
is not.
"""

from __future__ import annotations

import logging
import re

from . import config

log = logging.getLogger(__name__)

# <think> tags from reasoning models, and markup that reads badly aloud.
_TAG = re.compile(r"<think>.*?</think>", re.S | re.I)
_MARKUP = re.compile(r"[*_`#]+")


def _clean_for_speech(text: str) -> str:
    text = _TAG.sub(" ", text)
    text = _MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# Full stops that do NOT end a sentence. Without this, "3 p.m." — the exact
# form Parakeet produces — splits in two, and the TTS says "three pee." then
# pauses before "em."
_NOT_SENTENCE_END = re.compile(
    r"(?:\b(?:[a-z]|mr|mrs|ms|dr|prof|st|vs|etc|e\.g|i\.e|a\.m|p\.m|no)\.)\s*$",
    re.I,
)
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s")


def _split_sentence(buf: str) -> tuple[str | None, str]:
    """Take one complete sentence from the front of `buf`.

    Returns `(sentence, rest)`, or `(None, buf)` when no full sentence is ready.
    """
    for m in _SENTENCE_END.finditer(buf):
        chunk = buf[: m.end()]
        if _NOT_SENTENCE_END.search(chunk):
            continue  # an abbreviation, not a sentence end
        return chunk.strip(), buf[m.end() :]
    return None, buf


class _BaseConversation:
    name = "?"

    def __init__(self, system_prompt: str | None = None) -> None:
        self.base_prompt = system_prompt or config.SYSTEM_PROMPT
        self.messages: list[dict] = []

    @property
    def system_prompt(self) -> str:
        parts = [self.base_prompt]
        if config.REPLY_MAX_WORDS > 0:
            parts.append(
                f"HARD LIMIT: answer in {config.REPLY_MAX_WORDS} words or fewer. "
                "Never exceed it. Give the single most useful fact; if they want "
                "more, they will ask.\n"
                # Sentence length is controlled separately from reply length:
                # one long sentence delays the first sound, because the TTS
                # cannot start until a sentence is complete.
                "Use SHORT sentences, at most 12 words each. Two short sentences "
                "beat one long one."
            )
        return "\n\n".join(parts)

    def reset(self) -> None:
        self.messages = []

    def _trim(self) -> None:
        limit = config.MAX_HISTORY_MESSAGES
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def chat(self, text: str) -> str:
        raise NotImplementedError

    def chat_stream(self, text: str):
        """Like chat(), but yields SENTENCES as they become available.

        The default yields one whole chunk, so a backend without streaming
        support still works and the caller never has to know the difference.
        """
        yield self.chat(text)


class OllamaConversation(_BaseConversation):
    """Local Ollama. Free, runs offline."""

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
            raise RuntimeError(f"Ollama returned an empty response: {data!r}")

        self.messages.append({"role": "assistant", "content": reply})
        self._trim()
        log.info("LLM ollama (%d chars): %s", len(reply), reply)
        return _clean_for_speech(reply)

    def chat_stream(self, text: str):
        """Yield each sentence as soon as it lands, without waiting for the
        whole paragraph.

        Measured: time to first sound drops 53-71%. What gets cut is not the
        total time but the silence — and that is what makes a conversation feel
        alive rather than laggy.
        """
        import json as _json

        import requests

        self.messages.append({"role": "user", "content": text})
        full: list[str] = []
        rest = ""
        try:
            with requests.post(
                config.OLLAMA_CHAT_URL,
                json=self._payload(stream=True),
                timeout=config.OLLAMA_TIMEOUT,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = (_json.loads(line).get("message") or {}).get("content", "")
                    if not chunk:
                        continue
                    full.append(chunk)
                    rest += chunk
                    # Waiting for a whole sentence is deliberate: feeding half
                    # a sentence to the TTS wrecks the intonation and puts the
                    # pauses in the wrong places.
                    while True:
                        sentence, new_rest = _split_sentence(rest)
                        if sentence is None:
                            break
                        rest = new_rest
                        clean = _clean_for_speech(sentence)
                        if clean:
                            yield clean
        except Exception:
            self.messages.pop()
            raise

        tail = _clean_for_speech(rest)
        if tail:
            yield tail

        reply = "".join(full).strip()
        if not reply:
            self.messages.pop()
            raise RuntimeError("Ollama returned an empty response")

        self.messages.append({"role": "assistant", "content": reply})
        self._trim()
        log.info("LLM ollama stream (%d chars): %s", len(reply), reply)


class ClaudeConversation(_BaseConversation):
    """Claude API. Smarter, but needs the network and a paid key."""

    name = "claude"

    def __init__(self, system_prompt: str | None = None) -> None:
        super().__init__(system_prompt)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            if not (config.ANTHROPIC_API_KEY or config.ANTHROPIC_AUTH_TOKEN):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file "
                    "(see .env.example), or use LLM_BACKEND=ollama."
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
            raise RuntimeError("Claude returned an empty response")

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
                "LLM_BACKEND '%s' not recognised, falling back to 'ollama'",
                config.LLM_BACKEND,
            )
            _conversation = OllamaConversation()
        log.info("LLM backend: %s", _conversation.name)
    return _conversation


def chat(text: str) -> str:
    return get_conversation().chat(text)


if __name__ == "__main__":
    #     .venv-agent\Scripts\python.exe -m agent.llm "hello"
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    message = " ".join(sys.argv[1:]) or "Hello, can you hear me?"
    print(chat(message))
