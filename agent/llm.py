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

from . import config, tools

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


# The model announcing it is about to look something up, without actually
# emitting a tool call. Two shapes, both measured on qwen2.5:7b at 3 in 8:
#
#   "CallCheck for your next class time and location."   <- mangled tool call
#   "Let me check your schedule."                        <- empty promise
#
# Either one spoken aloud is worse than a wrong answer: it sounds like the
# agent is working when nothing is happening, and no answer ever arrives.
_STALL = re.compile(
    r"^\s*(?:"
    r"call\w*\s*(?:check|get|read)"          # CallCheck, Callget_schedule
    r"|let(?:'?s| me)\s+(?:check|look|see|find)"
    r"|i(?:'?ll| will)\s+(?:check|look|see|find)"
    r"|checking\b|looking\b|one moment while"
    r")",
    re.I,
)


# The other shape a failed tool call takes: it comes out looking like code
# rather than prose. Measured examples, all spoken aloud as gibberish:
#
#   .GetOrdinal("nextlecture")
#   Callget_schedule(kind="class")
#
# A real spoken answer never contains a bare function call, a snake_case
# identifier, or a leading dot — so shape alone is enough to spot it, without
# having to guess at the next wording the model will invent.
_CODEY = re.compile(
    r"^\s*[.\w]*\w+\s*\(|"     # foo(  /  .GetOrdinal(
    r"^\s*\.\w|"               # leading dot
    r"\b\w+_\w+\s*\(",         # snake_case(
)


def _looks_like_stall(text: str) -> bool:
    """A promise to look something up, or a mangled tool call — not an answer.

    Only meaningful when no tool call came with it. The same sentence next to a
    real tool call is harmless narration.
    """
    text = text or ""
    return bool(_STALL.match(text) or _CODEY.search(text))


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

    def _payload(self, stream: bool, with_tools: bool = True) -> dict:
        body = {
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
        if with_tools and tools.enabled():
            body["tools"] = tools.SCHEMA
        return body

    def _run_tools(self, calls: list[dict]) -> None:
        """Execute what the model asked for and append the results.

        The assistant's request is appended too, not just the result. Ollama
        needs the pair to make sense of the exchange — a `tool` message with no
        preceding `tool_calls` reads as a reply to nothing.
        """
        self.messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {})
            result = tools.run(fn.get("name", ""), fn.get("arguments"))
            self.messages.append({"role": "tool", "content": result})

    def _post(self, stream: bool = False) -> dict:
        import requests

        resp = requests.post(
            config.OLLAMA_CHAT_URL,
            json=self._payload(stream=stream),
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("message") or {}

    def chat(self, text: str) -> str:
        n_before = len(self.messages)
        self.messages.append({"role": "user", "content": text})
        try:
            msg = self._post()
            # A model may need several rounds: read the schedule, then save
            # something based on it. Bounded, because a model that keeps asking
            # for the same tool would otherwise loop until the timeout.
            for _ in range(config.TOOL_MAX_ROUNDS):
                calls = msg.get("tool_calls") or []
                if not calls:
                    break
                self._run_tools(calls)
                msg = self._post()
        except Exception:
            del self.messages[n_before:]
            raise

        reply = (msg.get("content") or "").strip()
        if not reply:
            del self.messages[n_before:]
            raise RuntimeError("Ollama returned an empty response")

        self.messages.append({"role": "assistant", "content": reply})
        self._trim()
        log.info("LLM ollama (%d chars): %s", len(reply), reply)
        return _clean_for_speech(reply)

    def _stream_once(self):
        """One streamed request. Yields text as it arrives; returns any tool
        calls the model made instead of answering.

        Streaming is kept even when tools are in play. The alternative — a
        non-streamed round to check for tool calls first — would cost the
        53-71% silence reduction on every ordinary question, to serve the ones
        that need a tool. Ollama streams `tool_calls` in their own chunks, so
        both can be read off the same response.
        """
        import json as _json

        import requests

        calls: list[dict] = []
        rest = ""
        full: list[str] = []

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
                msg = _json.loads(line).get("message") or {}
                if msg.get("tool_calls"):
                    calls.extend(msg["tool_calls"])
                chunk = msg.get("content", "")
                if not chunk:
                    continue
                full.append(chunk)
                rest += chunk
                # Waiting for a whole sentence is deliberate: feeding half a
                # sentence to the TTS wrecks the intonation and puts the pauses
                # in the wrong places.
                while True:
                    sentence, new_rest = _split_sentence(rest)
                    if sentence is None:
                        break
                    rest = new_rest
                    clean = _clean_for_speech(sentence)
                    if clean:
                        yield clean

        tail = _clean_for_speech(rest)
        if tail:
            yield tail

        # Handed back through the generator rather than returned, so the caller
        # can act on them after consuming the text.
        self._last_calls = calls
        self._last_text = "".join(full).strip()

    def chat_stream(self, text: str):
        """Yield each sentence as soon as it lands, without waiting for the
        whole paragraph.

        Measured: time to first sound drops 53-71%. What gets cut is not the
        total time but the silence — and that is what makes a conversation feel
        alive rather than laggy.
        """
        n_before = len(self.messages)
        self.messages.append({"role": "user", "content": text})
        spoken: list[str] = []

        try:
            for attempt in range(config.TOOL_MAX_ROUNDS + 1):
                held: list[str] = []
                suspect = False

                for sentence in self._stream_once():
                    # The FIRST sentence decides how the rest is treated. It
                    # costs nothing to wait for: a sentence is only yielded
                    # once complete anyway.
                    if not held and not spoken:
                        held.append(sentence)
                        suspect = _looks_like_stall(sentence)
                        if not suspect:
                            spoken.append(sentence)
                            yield sentence
                            held.clear()
                        continue

                    if suspect:
                        # Once the opening reads as a stall, the whole reply is
                        # withheld — not just that sentence. Measured failure:
                        # "Checking your schedule... You have 5 assignments due
                        # this month." The store held one. Releasing the first
                        # sentence the moment a second arrived let the invented
                        # number straight through.
                        held.append(sentence)
                        continue

                    spoken.append(sentence)
                    yield sentence

                calls = getattr(self, "_last_calls", None)

                if suspect and not calls:
                    if attempt < config.TOOL_MAX_ROUNDS:
                        log.info("stall with no tool call, retrying: %r", held[0])
                        continue
                    # Out of retries. Saying the stall would promise a lookup
                    # that is never coming; saying nothing sounds like a crash.
                    log.warning("gave up after %d stalls: %r",
                                attempt + 1, " ".join(held))
                    fallback = "Sorry, I couldn't read your schedule just then."
                    spoken.append(fallback)
                    yield fallback
                    self._last_text = fallback
                    break

                # Not a stall after all — release what was held.
                for sentence in held:
                    spoken.append(sentence)
                    yield sentence

                if not calls:
                    break
                # Anything said before a tool call is narration, not an answer.
                # It has been spoken already, but must not enter the history as
                # the assistant's reply.
                self._run_tools(calls)
                self._last_calls = None
        except Exception:
            del self.messages[n_before:]
            raise

        reply = getattr(self, "_last_text", "") or " ".join(spoken)
        if not reply.strip():
            del self.messages[n_before:]
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
