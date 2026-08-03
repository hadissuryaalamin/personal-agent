"""Otak agent. Dua backend: Claude API (default) atau Ollama lokal.

Keduanya punya antarmuka sama: `.chat(text) -> str` dan `.reset()`, jadi bisa
ditukar lewat LLM_BACKEND di .env tanpa nyentuh kode lain.
"""

from __future__ import annotations

import logging
import re
import threading

from . import calendar, config, memory, tugas

log = logging.getLogger(__name__)

# Model yang nolak parameter `effort` (400 kalau dikirim). Haiku & Sonnet 4.5
# generasi lama nggak punya kontrol effort sama sekali.
_TANPA_EFFORT = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-haiku-3")
# Model generasi lama yang kontrol thinking-nya beda bentuk; parameternya
# nggak dikirim biar nggak ditolak.
_TANPA_THINKING = ("claude-haiku-3",)


def _opsi_model() -> dict:
    """Parameter yang berbeda-beda per model. Dipisah biar dipakai sama semua
    pemanggilan (obrolan, penyaring fakta, pengurai acara)."""
    opsi = {}
    if config.CLAUDE_EFFORT and not config.CLAUDE_MODEL.startswith(_TANPA_EFFORT):
        opsi["output_config"] = {"effort": config.CLAUDE_EFFORT}
    if config.CLAUDE_THINKING and not config.CLAUDE_MODEL.startswith(_TANPA_THINKING):
        opsi["thinking"] = {"type": config.CLAUDE_THINKING}
    return opsi

# Buang sisa markup yang kelewat: tag XML, bintang markdown, backtick.
# Penting karena teksnya dibacakan speaker — "asterisk asterisk" itu ganggu.
_TAG = re.compile(r"<[^>]{1,40}>")
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


FACTS_PROMPT = """You filter long-term memory for a personal assistant. Decide \
what is worth remembering about the user from one exchange.

WORTH remembering: name, what they like to be called, where they study or work, \
important people in their life, preferences that will come up again (how they \
want to be addressed, what answer style they like), projects they are working \
on, conditions worth being aware of.

NOT worth it: one-off chatter, general knowledge questions, things true only \
today, and anything already on the list.

NEVER store class schedules, lecture times, room locations, event dates, or \
tasks. All of those have their own always-current source, and a copy here goes \
stale and starts contradicting it.

Reply with the COMPLETE updated list of facts, one per line, each starting with \
"- ". Merge or update existing entries rather than duplicating. Drop anything no \
longer true. If nothing needs to change, reply exactly: NO CHANGE

Do not write any explanation outside the list."""


class _BaseConversation:
    """Riwayat chat, dimuat dari disk pas mulai dan disimpan tiap giliran."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.base_prompt = system_prompt or config.SYSTEM_PROMPT
        self.messages: list[dict[str, str]] = memory.load_history()

    @property
    def system_prompt(self) -> str:
        """Seluruh system prompt sebagai satu teks (dipakai Ollama & _oneshot)."""
        stabil, berubah = self._bagian_prompt()
        return stabil + "\n\n" + berubah

    def _bagian_prompt(self) -> tuple[str, str]:
        """Pisah yang jarang berubah dari yang berubah tiap menit.

        Prompt caching itu cocok-awalan: sekali ada byte yang beda, semua yang
        di belakangnya ikut batal. Jam sekarang berubah tiap menit, jadi kalau
        ditaruh di depan, fakta + jadwal + tugas ikut kebuang dari cache terus.
        Makanya jam ditaruh paling belakang.
        """
        bagian = [self.base_prompt]

        fakta = memory.read_facts()
        if fakta:
            bagian.append(
                f"What you remember about the user from earlier conversations:\n{fakta}\n"
                "Use it when relevant, but don't bring it up unless asked."
            )

        try:
            jadwal = calendar.agenda()
        except Exception:
            log.warning("gagal nyusun agenda kalender", exc_info=True)
            jadwal = ""
        if jadwal:
            bagian.append(
                f"{jadwal}\n"
                "For schedule questions, answer ONLY from the list above. Read each "
                "line whole, left to right — never mix a course name from one "
                "line with a location from another. For 'what's next', use the "
                "NEXT UP line; for today or tomorrow, use the TODAY and "
                "TOMORROW lines, which already include the count.\n"
                "Say times naturally: 9 becomes 'nine', 15 30 becomes 'half "
                "past three'. Add am or pm so it is never ambiguous.\n"
                "Don't read the whole list unless the user asks for all of it."
            )

        try:
            task_list = tugas.summary()
        except Exception:
            log.warning("gagal nyusun daftar tugas", exc_info=True)
            task_list = ""
        if task_list:
            bagian.append(
                f"{task_list}\n"
                "If the user asks what to work on, weigh the nearest deadline, the "
                "estimated hours, and the schedule above (hours spent in class "
                "aren't available for coursework). Name one or two of the most "
                "pressing, not the whole list."
            )

        # Ditaruh paling belakang bagian stabil, persis sebelum jam. Instruksi
        # panjang-jawaban gampang tenggelam kalau ketimbun jadwal & tugas.
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

        return "\n\n".join(bagian), calendar.konteks_waktu()

    def chat_stream(self, text: str):
        """Sama kayak chat(), tapi ngeluarin KALIMAT satu per satu pas jadi.

        Default-nya: nggak ada backend yang wajib streaming, jadi yang nggak
        dukung cukup ngeluarin satu potong utuh. Pemanggilnya nggak perlu tau
        bedanya.
        """
        yield self.chat(text)

    def reset(self) -> None:
        self.messages = []
        memory.save_history([])

    def forget(self) -> None:
        """Hapus semua memori, di RAM maupun di disk."""
        self.messages = []
        memory.forget_all()

    def _trim(self) -> None:
        limit = config.MAX_HISTORY_MESSAGES
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def _selesai_giliran(self, user_text: str, reply: str) -> None:
        """Simpan riwayat, lalu saring fakta di latar belakang.

        Penyaringan fakta itu panggilan API tambahan. Dijalanin di thread lain
        supaya user nggak nunggu — dia udah dapet jawabannya duluan.
        """
        memory.save_history(self.messages)
        if not (config.MEMORY_ENABLED and config.MEMORY_AUTO_FACTS):
            return
        threading.Thread(
            target=self._saring_fakta,
            args=(user_text, reply),
            name="saring-fakta",
            daemon=True,
        ).start()

    def _saring_fakta(self, user_text: str, reply: str) -> None:
        try:
            lama = memory.read_facts() or "(empty so far)"
            hasil = self._oneshot(
                FACTS_PROMPT,
                f"Current fact list:\n{lama}\n\n"
                f"Latest exchange:\nUser: {user_text}\nAssistant: {reply}",
            ).strip()

            if not hasil or hasil.upper().startswith("NO CHANGE"):
                return
            # Baris harus punya isi setelah tanda hubungnya. Tanpa ini, model
            # kecil yang balas '-' doang bakal ngehapus seluruh fakta.
            baris = [
                b for b in hasil.splitlines()
                if b.strip().startswith("-") and len(b.strip().lstrip("- ")) >= 3
            ]
            if not baris:
                log.debug("penyaring fakta balikin format aneh, diabaikan: %r", hasil)
                return
            memory.write_facts("\n".join(baris))
            log.info("fakta diperbarui (%d baris)", len(baris))
        except Exception:
            # Memori itu bonus; jangan sampai bikin obrolan gagal
            log.warning("gagal nyaring fakta", exc_info=True)

    def _oneshot(self, system: str, user: str) -> str:
        """Sekali tanya-jawab tanpa nyentuh riwayat obrolan."""
        raise NotImplementedError

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

        stabil, berubah = self._bagian_prompt()
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                # Bagian stabil ditandai buat di-cache; jam sekarang nyusul di
                # blok terpisah supaya pergantian menit nggak ngebatalin cache.
                system=[
                    {
                        "type": "text",
                        "text": stabil,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": berubah},
                ],
                messages=self.messages,
                **_opsi_model(),
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
        u = response.usage
        log.info(
            "LLM claude (%d in / %d out | cache: %d baca, %d tulis): %s",
            u.input_tokens,
            u.output_tokens,
            u.cache_read_input_tokens or 0,
            u.cache_creation_input_tokens or 0,
            reply,
        )
        self._selesai_giliran(text, reply)
        return _clean_for_speech(reply)

    def _oneshot(self, system: str, user: str, skema: dict | None = None) -> str:
        kwargs = _opsi_model()
        if skema is not None:
            # Structured outputs: jawabannya dijamin JSON yang cocok skema, jadi
            # nggak perlu bersihin teks pembuka atau pagar markdown.
            # Digabung, bukan ditimpa — effort ada di objek yang sama.
            kwargs.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": skema,
            }
        response = self._get_client().messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.CLAUDE_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        return " ".join(b.text for b in response.content if b.type == "text")


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
            "options": {
                "temperature": 0.7,
                "num_predict": config.OLLAMA_NUM_PREDICT,
                "num_ctx": config.OLLAMA_NUM_CTX,
            },
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
        self._selesai_giliran(text, reply)
        return _clean_for_speech(reply)

    def chat_stream(self, text: str):
        """Keluarin kalimat begitu jadi, jangan nunggu paragrafnya selesai.

        Ini yang bikin mode sesi kerasa hidup: TTS bisa mulai ngomong di
        kalimat pertama sementara model masih ngarang kalimat kedua. Yang
        kepotong bukan waktu totalnya, tapi **waktu sampai bunyi pertama** —
        dan itu yang dirasain user.
        """
        import json as _json

        import requests

        self.messages.append({"role": "user", "content": text})
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "system", "content": self.system_prompt}]
            + self.messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": config.OLLAMA_NUM_PREDICT,
                "num_ctx": config.OLLAMA_NUM_CTX,
            },
        }

        penuh: list[str] = []
        sisa = ""
        try:
            with requests.post(
                config.OLLAMA_CHAT_URL,
                json=payload,
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
                    # Pecah di batas kalimat. Nunggu kalimat utuh itu sengaja:
                    # ngirim potongan setengah kalimat ke TTS bikin intonasinya
                    # ngawur dan jedanya kedengeran di tempat yang salah.
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
        self._selesai_giliran(text, reply)

    def _oneshot(self, system: str, user: str, skema: dict | None = None) -> str:
        import requests

        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if skema is not None:
            payload["format"] = skema  # Ollama juga dukung JSON schema
        resp = requests.post(
            config.OLLAMA_CHAT_URL, json=payload, timeout=config.OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content", "")


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
