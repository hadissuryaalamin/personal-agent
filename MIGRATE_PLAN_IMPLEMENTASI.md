# Rencana Implementasi Migrasi

Jawaban atas [MIGRATE_PLAN.md](MIGRATE_PLAN.md) §23 — Milestone 1 (Architecture
Audit). **Belum ada kode yang diubah.**

Semua klaim di dokumen ini diverifikasi lewat pengujian nyata di mesin ini
(`pip --dry-run`, halaman resmi model), bukan dari ingatan.

---

## 0. Ringkasan untuk pengambilan keputusan

Tiga temuan mengubah bentuk rencananya secara mendasar.

### 0.1 Parakeet menghapus dukungan Bahasa Indonesia — permanen

| Model | Bahasa |
|---|---|
| `parakeet-tdt-0.6b-v2` | **English saja** |
| `parakeet-tdt-0.6b-v3` | 25 bahasa Eropa — **Indonesia tidak termasuk** |
| Whisper (sekarang) | 99 bahasa, termasuk Indonesia |

Ini bukan soal setelan. Setelah migrasi, **perintah suara Bahasa Indonesia
berhenti bekerja sepenuhnya.** Seluruh interaksi harus dalam Bahasa Inggris.

Data yang ada saat ini semuanya Bahasa Indonesia: `facts.md`, `tugas.json`,
riwayat obrolan, dan 90 acara kalender berjudul campuran.

**Ini keputusan produk, bukan keputusan teknis** — dan satu-satunya bagian dari
rencana ini yang tidak bisa dibatalkan tanpa mengganti model lagi. Lihat §8
untuk pilihannya.

### 0.2 NeMo bisa dihindari sepenuhnya

MIGRATE_PLAN §3 mengasumsikan NVIDIA NeMo. Diuji di mesin ini:

| Jalur | Paket baru | Ketergantungan berat |
|---|---:|---|
| `nemo_toolkit[asr]` | **~150** | torch 2.13, lightning, wandb, tensorboard, optuna, datasets, transformers |
| **`onnx-asr`** | **0** | tidak ada — numpy + onnxruntime **sudah terpasang** |

`nemo_toolkit[asr]` memang *resolve* di Windows (di luar dugaanku), tapi yang
ditariknya adalah tumpukan **pelatihan**, bukan inferensi. Untuk asisten latar
belakang yang idle-nya 28 MB RAM, itu tidak proporsional.

`onnx-asr` menjalankan Parakeet dengan numpy + onnxruntime saja, mendukung CUDA,
dan **tidak menambah satu paket pun** karena semuanya sudah ada untuk Piper dan
faster-whisper.

> **Rekomendasi: pakai `onnx-asr`, bukan NeMo.** Ini menyimpang dari
> MIGRATE_PLAN §3, dan menurutku menyimpangnya benar.

### 0.3 Milestone 3 sudah selesai

MIGRATE_PLAN §4 meminta Qwen2.5 7B lokal lewat Ollama. Itu **sudah berjalan
sekarang**: `LLM_BACKEND=ollama`, `qwen2.5:7b` (Q4_K_M, 100% GPU), dan seluruh
fitur — kalender, memori, tugas, konfirmasi — sudah diuji di atasnya.

Milestone 3 tinggal validasi offline.

---

## 1. Audit keadaan sekarang

### 1.1 Antarmuka yang jadi batas abstraksi

```python
# stt.py
get_model()            # muat (lazy)
is_loaded() -> bool
unload_model() -> bool
warmup()
transcribe(audio: np.ndarray) -> str

# tts.py
get_voice()
speak(text: str) -> bytes        # WAV

# llm.py
get_conversation() -> _BaseConversation
chat(text: str) -> str
class _BaseConversation:  system_prompt, reset(), forget(), chat()
class ClaudeConversation / OllamaConversation
```

Ketiganya **sudah memenuhi syarat** MIGRATE_PLAN §3/§5: `main.py` tidak
mengandung kode spesifik Whisper maupun Piper. Migrasi bisa dilakukan
seluruhnya di balik antarmuka ini.

`stt.transcribe()` menerima `np.ndarray` float32 mono 16 kHz — **sama persis**
dengan yang dibutuhkan Parakeet. Tidak perlu perubahan di `audio.py`.

### 1.2 Ketergantungan Bahasa Indonesia

| Berkas | Yang perlu diterjemahkan | Perkiraan |
|---|---|---|
| `config.py` | `SYSTEM_PROMPT`, `WHISPER_PROMPT`, `WHISPER_LANG` | ~20 baris |
| `waktu_id.py` | Seluruh modul (HARI, BULAN, ANGKA, pola regex) | 126 baris |
| `tugas.py` | `_KATA_TUGAS`, `_NIAT_TAMBAH`, `_NIAT_SELESAI`, `_label_tenggat`, HARI/BULAN | ~40 baris |
| `jadwal_baru.py` | `_NIAT`, `_YA`, `_TIDAK`, `PROMPT`, `PROMPT_TUGAS`, `kalimat_konfirmasi` | ~60 baris |
| `main.py` | `_FRASA_LUPA`, semua `_say_safely(...)`, log | ~25 baris |
| `calendar.py` | HARI, BULAN, `JENIS`, `_label_hari`, teks agenda | ~50 baris |
| `llm.py` | `PROMPT_FAKTA`, instruksi kalender & tugas di system prompt | ~40 baris |

Semuanya **data dan string**, bukan struktur. Tidak ada yang perlu dirancang
ulang — hanya diterjemahkan.

### 1.3 Yang tidak perlu disentuh sama sekali

`audio.py`, `memory.py`, `kalender_lokal.py`, `gcal.py`, `stt.py` (bagian siklus
hidup model), `scripts/*`. Semuanya bebas bahasa.

---

## 2. Milestone 2 — Parakeet

### Berkas
- `agent/stt.py` — ganti isi, pertahankan antarmuka
- `agent/config.py` — tambah `STT_BACKEND`, `STT_MODEL`
- `pyproject.toml` — tambah `onnx-asr[hub]`

### Perubahan
`stt.py` mendapat dua implementasi di balik antarmuka yang sama:

```python
_BACKEND = {"whisper": _WhisperBackend, "parakeet": _ParakeetBackend}
```

`get_model()`, `transcribe()`, `is_loaded()`, `unload_model()` tetap sebagai
fungsi modul — `main.py` dan `llm.py` tidak berubah satu baris pun.

Yang hilang di Parakeet dan perlu ditangani:
- **Tidak ada `initial_prompt`** — `WHISPER_PROMPT` (yang menurunkan WER
  36%→25%) tidak punya padanan. Dampaknya harus diukur ulang.
- **Tidak ada VAD internal** — `vad_filter=True` sekarang membuang hening.
  Perlu diperiksa apakah Parakeet berhalusinasi pada hening seperti Whisper.
- **Tidak ada `language=`** — model ini English saja.

### Dependensi baru
`onnx-asr` — **nol paket tambahan**, semua sudah ada.

### Risiko
| Risiko | Mitigasi |
|---|---|
| Bobot ONNX Parakeet mungkin belum ada di HF | Verifikasi unduhan **sebelum** menyentuh `stt.py` |
| `unload_model()` mungkin tidak melepas VRAM (onnxruntime ≠ ctranslate2) | Ukur dengan `nvidia-smi` sebelum/sesudah, seperti waktu Whisper |
| Akurasi tanpa `initial_prompt` mungkin turun untuk istilah teknis | Ukur lawan 22 rekaman di `logs/rec` |

### Tes
Wajib menggunakan **rekaman Bahasa Inggris baru** — 22 rekaman yang ada
berbahasa Indonesia dan tidak bisa dipakai menilai Parakeet.

Bandingkan lawan Whisper `large-v3-turbo` pada: WER, waktu per kalimat, waktu
muat dingin/hangat, VRAM.

### Kriteria terima
Mic → Parakeet → teks Inggris benar → pipeline lama jalan, **tanpa perubahan di
`main.py`**.

---

## 3. Milestone 3 — Qwen offline

Sudah berjalan. Yang tersisa hanya validasi:

1. Putuskan jaringan
2. Jalankan `scripts/status.ps1`
3. Uji: pertanyaan umum, pertanyaan jadwal, tambah tugas, buat acara
4. Restart, pastikan memori & tugas bertahan

### Berkas
`config.py` — tambah `OFFLINE_MODE`; kalau `true` dan `LLM_BACKEND=claude`,
**gagal saat startup dengan pesan jelas**, jangan diam-diam mencoba jaringan
(MIGRATE_PLAN §18).

---

## 4. Milestone 4 — Kokoro

### Berkas
- `agent/tts.py` — ganti isi, pertahankan `speak(text) -> bytes`
- `agent/config.py` — tambah `TTS_BACKEND`, `TTS_VOICE`
- `scripts/setup.ps1` — unduh bobot Kokoro
- `pyproject.toml` — tambah `kokoro-onnx`

### Pilihan runtime

| Paket | Paket baru | Catatan |
|---|---:|---|
| `kokoro` (resmi) | ~60 | menarik torch 2.13, transformers, spacy |
| **`kokoro-onnx`** | **18** | tanpa torch; onnxruntime sudah ada |

Keduanya butuh `espeak-ng` — sudah ikut terpasang bersama `piper-tts`, jadi
kemungkinan besar bisa dipakai ulang. **Verifikasi ini lebih dulu.**

### Perubahan penting: sample rate
Piper mengeluarkan 22.050 Hz, Kokoro **24.000 Hz**. `audio.play_wav()` sudah
membaca sample rate dari header WAV, jadi **tidak perlu berubah** — tapi
`audio._pad()` memakai `config.SAMPLE_RATE` (16.000) untuk beep. Perlu
diperiksa agar bantalan playback tetap benar.

### Siklus hidup
MIGRATE_PLAN §9 menyarankan Kokoro tetap residen. Masuk akal (82 juta parameter,
jauh lebih kecil dari Piper), tapi **ukur dulu**: Piper sekarang di-warmup dan
memakan ~0 VRAM karena CPU-only. Kalau Kokoro juga CPU-only dan cepat,
pertahankan pola yang ada.

### Risiko
| Risiko | Mitigasi |
|---|---|
| Kokoro lebih lambat dari Piper → jeda bertambah | Ukur waktu sintesis; Piper sekarang ~1 dtk |
| Bentrok `espeak-ng` antara piper-tts dan kokoro | Uji di venv terpisah dulu |
| Kualitas suara Inggris tidak cocok selera | Uji beberapa suara sebelum menetapkan default |

### Kriteria terima
teks → Kokoro → WAV → `audio.play_wav()` → speaker, tanpa kode Kokoro di
`main.py`, dan **tanpa pemotongan di ujung** (uji ulang dengan metode
pengukuran 119 ms yang sudah ada).

---

## 5. Milestone 5 — Migrasi ke Bahasa Inggris

Terbesar dari sisi jumlah baris, tapi paling kecil risikonya — semuanya string.

### Urutan pengerjaan (dari yang paling berisiko)

**5a. `waktu_id.py` → `time_en.py`** (MIGRATE_PLAN §8 opsi 2)

Buat modul baru, jangan ubah yang lama. Alasannya: pengurai ini yang membuat
akurasi tanggal naik dari 2/10 jadi 5/5, dan menyimpannya utuh berarti
Bahasa Indonesia bisa dipulihkan kalau keputusan §0.1 berubah.

`jadwal_baru.py` memilih modul berdasarkan `LANGUAGE` di config.

Pola Inggris yang perlu ditangani: `today`, `tomorrow`, `day after tomorrow`,
`next Friday`, `this Friday`, `in 3 days`, `on the 14th`, `at 3 PM`,
`at half past two`, `quarter to five`, `noon`, `midnight`.

**Perhatian khusus AM/PM.** Pengurai Indonesia memakai pagi/siang/sore/malam;
Inggris memakai AM/PM dengan aturan berbeda — `12 AM` = tengah malam,
`12 PM` = siang. Bug `jam 12 malam` yang sudah ditemukan punya padanan langsung
di sini, jadi ujilah keduanya sejak awal.

**5b. Kata kunci niat** — `tugas.py`, `jadwal_baru.py`, `main.py`

Urutan pengecekan **harus tetap**: tugas sebelum acara. Padanan Inggris punya
tumpang tindih yang sama: *"add a task"* dan *"add an event"* sama-sama cocok
dengan *"add ..."*.

Frasa penghapus memori (`_FRASA_LUPA`) tetap dicocokkan **lokal**, bukan lewat
LLM — perintah yang tidak bisa dibatalkan tidak boleh bergantung pada tebakan
model.

Konfirmasi: `_YA`/`_TIDAK` → `yes/yeah/yep/correct/right/sure/save/go ahead` dan
`no/nope/cancel/wrong/don't/nevermind`. **Pertahankan aturan bahwa jawaban ragu
dibaca sebagai batal.** Padanan jebakan *"hmm apa ya"* di Inggris adalah
*"...right?"* dan *"well, yeah, no"* — perlu kasus uji sendiri.

**5c. Prompt & pesan** — `config.py`, `llm.py`, `calendar.py`, `main.py`

Termasuk aturan format jam. Instruksi Indonesia melarang bentuk "setengah
sembilan"; padanan Inggris melarang jam ambigu tanpa AM/PM.

**5d. Format agenda** — `calendar.py`

HARI/BULAN → nama Inggris. **Pertahankan** pemisah `|`, baris `KELAS
BERIKUTNYA`/`HARI INI`/`BESOK` yang dihitung Python, dan format jam yang
enak diucapkan. Ketiganya yang membuat qwen naik dari 0/3 ke 3/3 — kalau ikut
diterjemahkan sambil disederhanakan, regresi itu akan kembali.

### Data lama
`facts.md`, `tugas.json`, riwayat, dan judul acara masih Bahasa Indonesia.
Sarankan: **tulis ulang `facts.md` manual**, biarkan tugas & kalender apa adanya
(qwen memahami keduanya), dan hapus `history.json` sekali karena riwayat
campur bahasa membingungkan model.

### Kriteria terima
Seluruh perintah suara utama bekerja dalam Bahasa Inggris, dan **tes pemilahan
niat yang ada** (7 kasus tugas-vs-acara, 15 kasus ya/tidak) lulus dalam versi
Inggrisnya.

---

## 6. Milestone 6 — Anggaran sumber daya

### Perkiraan VRAM

| Model | Sekarang | Setelah migrasi |
|---|---:|---:|
| STT | Whisper turbo ~1.984 MB | Parakeet 0,6B **~1.200–1.500 MB** (perkiraan) |
| LLM | qwen2.5:7b 5.420 MB | sama |
| TTS | Piper ~0 (CPU) | Kokoro **~0–400 MB** |
| Desktop | ~1.500 MB | sama |
| **Puncak** | **~7.600 / 8.151 MB** | **~7.300–8.300 MB** |

**Ini tetap mepet.** Kartunya 8.151 MB, dan sekarang pun hanya tersisa 521 MB
saat kedua model dimuat. Kalau Kokoro ditaruh di GPU, kemungkinan besar
melampaui batas.

**Rekomendasi: Kokoro di CPU.** 82 juta parameter cukup ringan, dan ini
mempertahankan pola Piper yang sudah terbukti (0 VRAM, di-warmup, tidak pernah
dilepas).

### Siklus hidup yang disarankan
```
Parakeet → lazy load + lepas saat idle   (seperti Whisper sekarang)
Qwen     → keep-alive Ollama 5 menit     (tidak berubah)
Kokoro   → residen di CPU                (seperti Piper sekarang)
```

Sesuai MIGRATE_PLAN §9, **hanya diadopsi kalau pengukuran mendukung.**

---

## 7. Milestone 7 — Validasi offline

Tambahkan `OFFLINE_MODE=true` yang menolak backend cloud saat startup.

Urutan uji sesuai MIGRATE_PLAN §19: putuskan jaringan → STT → pertanyaan umum →
TTS → memori → kalender → tugas → restart → cek persistensi.

Satu hal yang perlu diperiksa dan mudah terlewat: **unduhan model pertama kali
butuh jaringan.** `onnx-asr` dan `kokoro-onnx` mengambil bobot dari HuggingFace
saat pertama dipakai. `scripts/setup.ps1` harus mengunduh keduanya di muka,
supaya mode offline tidak gagal pada pemakaian pertama.

---

## 8. Keputusan yang perlu diambil sebelum mulai

**Bahasa Indonesia akan hilang.** Ini konsekuensi §0.1, tidak bisa dihindari
selama STT-nya Parakeet. Tiga pilihan:

| Pilihan | Konsekuensi |
|---|---|
| **A. Ikuti rencana apa adanya** | Inggris penuh. Indonesia berhenti bekerja. Paling sesuai MIGRATE_PLAN. |
| **B. Dwibahasa lewat `STT_BACKEND`** | Parakeet untuk Inggris, Whisper untuk Indonesia, dipilih lewat config. Kedua pengurai tanggal disimpan. Biaya: dua jalur untuk dirawat dan diuji. |
| **C. Tetap Whisper, ambil sisanya** | Kokoro + Qwen + Inggris tanpa Parakeet. Whisper `large-v3-turbo` sudah 8% WER dan mendukung 99 bahasa. Yang hilang cuma kemungkinan keunggulan akurasi Parakeet di Inggris — yang **belum terbukti** di mesin ini. |

Aku condong ke **B**: `stt.py` toh harus dijadikan berbasis backend untuk
mendukung Parakeet, jadi menyimpan Whisper hampir tidak menambah kerja, dan
keputusan bahasa jadi bisa dibatalkan.

Tapi ini keputusanmu — MIGRATE_PLAN menulis "optimize only for English", dan
kalau memang itu maksudnya, pilihan A yang paling lurus.

---

## 9. Urutan kerja yang disarankan

Berbeda dari urutan milestone di MIGRATE_PLAN, dengan alasan:

1. **Verifikasi bobot ONNX Parakeet bisa diunduh & jalan** — 30 menit. Kalau
   gagal, seluruh Milestone 2 berubah dan lebih baik tahu sekarang.
2. **Milestone 4 (Kokoro) lebih dulu** — risikonya paling rendah, terisolasi di
   `tts.py`, dan bisa diuji tanpa menyentuh bahasa apa pun.
3. **Milestone 5 (Inggris)** — sebelum Parakeet, supaya saat Parakeet masuk,
   perintah Inggris sudah bisa diuji.
4. **Milestone 2 (Parakeet)**.
5. **Milestone 6 & 7** — pengukuran dan validasi offline.

Menunda Parakeet ke belakang berarti aplikasinya **tetap bisa dipakai di setiap
tahap**, dan bagian yang tidak bisa dibatalkan (kehilangan Bahasa Indonesia)
dikerjakan paling akhir, setelah semua yang lain terbukti.
