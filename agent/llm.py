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


PROMPT_FAKTA = """Kamu penyaring memori buat asisten pribadi. Tugasmu mutusin apa \
yang layak diingat jangka panjang tentang user dari satu tukar obrolan.

Yang LAYAK diingat: nama, panggilan, kuliah/kerja di mana, orang penting di \
hidupnya, jadwal rutin, preferensi yang bakal kepakai lagi (cara dipanggil, gaya \
jawaban yang disuka), proyek yang lagi dikerjain, kondisi yang perlu diperhatiin.

Yang TIDAK layak: obrolan sekali lewat, pertanyaan pengetahuan umum, hal yang \
cuma berlaku hari itu, dan apa pun yang udah ada di daftar.

Balas dengan daftar fakta LENGKAP yang terbaru, satu per baris, diawali "- ". \
Gabung atau perbarui yang sudah ada daripada bikin duplikat. Buang yang ternyata \
sudah nggak berlaku. Kalau nggak ada yang perlu diubah, balas persis: TIDAK ADA

Jangan nulis penjelasan apa pun di luar daftar."""


class _BaseConversation:
    """Riwayat chat, dimuat dari disk pas mulai dan disimpan tiap giliran."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.base_prompt = system_prompt or config.SYSTEM_PROMPT
        self.messages: list[dict[str, str]] = memory.load_history()

    @property
    def system_prompt(self) -> str:
        """System prompt + waktu sekarang + fakta + jadwal, disusun tiap giliran.

        Disusun ulang (bukan di-cache) supaya jam terus akurat dan suntingan
        manual di facts.md langsung kepakai tanpa restart.
        """
        bagian = [self.base_prompt, calendar.konteks_waktu()]

        fakta = memory.read_facts()
        if fakta:
            bagian.append(
                f"Yang kamu inget tentang user dari obrolan sebelumnya:\n{fakta}\n"
                "Pakai ini kalau relevan, tapi jangan disebut-sebut kecuali ditanya."
            )

        try:
            jadwal = calendar.agenda()
        except Exception:
            log.warning("gagal nyusun agenda kalender", exc_info=True)
            jadwal = ""
        if jadwal:
            bagian.append(
                f"{jadwal}\n"
                "Kalau ditanya soal jadwal, jawab HANYA dari daftar di atas. "
                "Baca satu baris utuh dari kiri ke kanan — jangan campur nama "
                "matkul dari satu baris dengan lokasi dari baris lain. Kalau "
                "ditanya kelas berikutnya, pakai baris KELAS BERIKUTNYA.\n"
                "Sebut jam persis: 09:00 = 'jam sembilan pagi', 15:30 = 'jam "
                "tiga lewat tiga puluh sore'. JANGAN pakai bentuk 'setengah "
                "sembilan' — itu gampang meleset setengah jam.\n"
                "Jangan bacakan semuanya kecuali user emang minta semua."
            )

        try:
            daftar_tugas = tugas.ringkasan()
        except Exception:
            log.warning("gagal nyusun daftar tugas", exc_info=True)
            daftar_tugas = ""
        if daftar_tugas:
            bagian.append(
                f"{daftar_tugas}\n"
                "Kalau user nanya harus ngerjain apa, pertimbangkan tenggat "
                "terdekat, perkiraan lamanya, dan jadwal kuliah di atas (waktu "
                "yang kepakai kelas nggak bisa dipakai ngerjain tugas). Kasih "
                "satu atau dua yang paling mendesak, jangan bacakan semuanya."
            )

        return "\n\n".join(bagian)

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
        if not config.MEMORY_ENABLED:
            return
        threading.Thread(
            target=self._saring_fakta,
            args=(user_text, reply),
            name="saring-fakta",
            daemon=True,
        ).start()

    def _saring_fakta(self, user_text: str, reply: str) -> None:
        try:
            lama = memory.read_facts() or "(masih kosong)"
            hasil = self._oneshot(
                PROMPT_FAKTA,
                f"Daftar fakta sekarang:\n{lama}\n\n"
                f"Obrolan terbaru:\nUser: {user_text}\nAsisten: {reply}",
            ).strip()

            if not hasil or hasil.upper().startswith("TIDAK ADA"):
                return
            # Jaga-jaga kalau modelnya ngoceh di luar format daftar
            baris = [b for b in hasil.splitlines() if b.strip().startswith("-")]
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

        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                system=self.system_prompt,
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
        log.info(
            "LLM claude (%d in / %d out): %s",
            response.usage.input_tokens,
            response.usage.output_tokens,
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
        self._selesai_giliran(text, reply)
        return _clean_for_speech(reply)

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
