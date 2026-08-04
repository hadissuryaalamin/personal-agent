# personal-agent

Voice assistant lokal buat Windows. Jalan diam-diam di background, nggak ada
window. Pencet hotkey → ngomong → dia jawab lewat speaker. **Bahasa Inggris,
sepenuhnya offline** — nol byte keluar dari mesin ini.

```
hotkey → mic → Silero VAD → Parakeet TDT 0.6B → qwen2.5:7b → Kokoro-82M → speaker
```

Empat model, empat tugas: **VAD tau kapan kamu selesai ngomong**, **Parakeet
dengar**, **qwen mikir**, **Kokoro jawab**.

Cuma itu. Nggak ada kalender, tugas, memori antar-sesi, atau integrasi apa pun —
sengaja dikosongin biar bisa dibangun ulang dari dasar yang bersih. Riwayat
percakapan cuma hidup selama proses jalan.

## Yang dibutuhin

- Windows 10/11
- [Ollama](https://ollama.com/download) udah terinstall & jalan
- Python 3.11 (lewat [mise](https://mise.jdx.dev/), ada fallback di bawah)
- Mikrofon + speaker

## Install

```powershell
git clone <repo> personal-agent
cd personal-agent

# 1. Python 3.11 + venv
mise trust
mise install
mise where python@3.11              # catat path-nya
& "<path-di-atas>\python.exe" -m venv .venv-agent

# 2. Dependencies
.\.venv-agent\Scripts\python.exe -m pip install -e .

# 3. Semua bobot model (~5.7 GB)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# 4. Konfigurasi
copy .env.example .env

# 5. Pastikan beneran siap offline
.\.venv-agent\Scripts\python.exe -m agent.cek_offline
```

<details>
<summary>Tanpa mise</summary>

```powershell
py -3.11 -m venv .venv-agent
.\.venv-agent\Scripts\python.exe -m pip install -e .
```

Python 3.13+ belum dipakai karena `ctranslate2` (mesinnya faster-whisper) belum
punya wheel stabil di situ. faster-whisper cuma kepakai kalau
`STT_BACKEND=whisper`, tapi dependensinya tetap ikut terinstall.
</details>

`scripts\setup.ps1` narik semua bobot di muka. Kalau dilewat, tiap model
ke-download sendiri pas pertama dipakai — artinya pemakaian pertama butuh
jaringan, dan itu persis yang bikin mode offline kelihatan "rusak".

## Jalanin

```powershell
.\.venv-agent\Scripts\python.exe -m agent.main
```

Pencet **hotkey sekali** buat mulai, ngomong, **pencet lagi** buat berhenti.
Nggak mau mencet tiap giliran? Lihat [Mode sesi](#mode-sesi).

Karena nggak ada window, feedback-nya bunyi:

| Bunyi | Artinya |
|---|---|
| beep tinggi (880 Hz) | mulai rekam |
| beep sedang (560 Hz) | selesai rekam, lagi mikir |
| dua ketuk pendek (420 Hz) | kedengeran, tapi lagi sibuk |
| beep rendah panjang (240 Hz) | gagal — cek `logs/agent.log` |

## Mode sesi

Sekali pencet hotkey buat **masuk sesi**, terus ngomong bolak-balik tanpa mencet
lagi. Batas kalimat dideteksi suara, bukan tombol.

```
SESSION_MODE=true
```

Sesi tutup kalau: pencet hotkey lagi, diam 30 detik, atau bilang *"goodbye"* /
*"that's all"* / *"stop listening"*.

Batas diam itu **pengaman, bukan kenyamanan**: tanpa itu, lupa nutup berarti
model nyangkut di memori seharian.

**Mic ditutup selama agent ngomong.** Jadi kamu nggak bisa motong di tengah —
tapi agent juga nggak mungkin denger suaranya sendiri, jadi speaker biasa aman
(nggak wajib headphone).

Deteksi suaranya Silero VAD lewat `onnx_asr` — model yang sama yang udah dipakai
Parakeet, jadi **nol paket pip baru**. Terukur:

| | Rata-rata peluang | Frame di atas ambang |
|---|---:|---:|
| Hening | 0,004 | 0/62 |
| Desis keras | 0,011 | 0/62 |
| Ucapan asli | 0,580 | 95/166 |

0,17 ms per frame 32 ms — **190x realtime**, jadi CPU-nya praktis nol.

Kalimat kepotong di tengah? Naikin `VAD_SILENCE_MS`. Agent nyaut ke suara AC?
Naikin `VAD_THRESHOLD`.

## Jalan otomatis pas login

```powershell
# PowerShell as Administrator
powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1
```

```powershell
powershell -File scripts\status.ps1                       # nyala atau nggak?
Start-ScheduledTask -TaskName PersonalAgent               # nyalain
Stop-ScheduledTask -TaskName PersonalAgent                # matiin
Get-Content logs\agent.log -Tail 20 -Wait                 # lihat langsung
powershell -File scripts\install-startup.ps1 -Uninstall   # hapus task
```

### Cuma boleh ada satu agent

Agent nyangkut **hotkey global**. Dua agent artinya tiap pencetan ditangkep
dua-duanya dan keduanya rebutan mic — dan gejalanya nyasar: yang kelihatan bukan
"ada dua agent", tapi "fiturnya nggak jalan", karena yang nyaut duluan agent
lama.

Agent kedua sekarang nolak jalan sendiri:

```
Agent lain udah jalan (PID 41696). Yang ini berhenti — dua agent bakal
rebutan hotkey 'right ctrl'. Cek: powershell -File scripts\status.ps1
```

Kuncinya kunci file dari OS, bukan file PID: kunci OS dilepas otomatis pas
proses mati, jadi agent yang crash nggak ninggalin file yang bikin agent
berikutnya nolak jalan selamanya (diuji, termasuk `kill -9`).

**Ngetes dari terminal padahal autostart nyala?** Matiin task-nya dulu:

```powershell
Stop-ScheduledTask -TaskName PersonalAgent
.\.venv-agent\Scripts\python.exe -m agent.main
```

### Kalau ada yang aneh, cek build-nya duluan

Baris kedua log nyebut commit yang lagi jalan:

```
build: 9a00404 05/08 08:31 (file terbaru 05/08 08:31)
```

Python muat kode pas proses start — **ngedit file nggak nyentuh proses yang udah
jalan.** Ini penyebab paling sering dari "perbaikannya nggak jalan".

## Konfigurasi

Default ada di [agent/config.py](agent/config.py), bisa ditimpa lewat `.env`
(lihat [.env.example](.env.example) — 54 kunci, semuanya beneran dibaca kode).

| Variabel | Default | Keterangan |
|---|---|---|
| `HOTKEY` | `ctrl+space` | **Pakai tombol tunggal** — lihat di bawah |
| `SESSION_MODE` | `false` | `true` = ngobrol kontinu |
| `OFFLINE_MODE` | `true` | Tolak backend jaringan pas startup |
| `LLM_BACKEND` | `ollama` | `ollama` atau `claude` (butuh `OFFLINE_MODE=false`) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model apa pun yang udah di-`ollama pull` |
| `REPLY_MAX_WORDS` | `25` | Batas panjang jawaban — lihat di bawah |
| `STT_BACKEND` | `parakeet` | `parakeet` (Inggris, 0 VRAM) atau `whisper` |
| `TTS_BACKEND` | `kokoro` | `kokoro` (24 kHz, 54 suara) atau `piper` |
| `VAD_SILENCE_MS` | `800` | Sepi segini = kalimat dianggap selesai |

### Pakai tombol tunggal

Tombol yang ditahan **bareng modifier** (`Ctrl+Space`) bikin Windows ngirim
pasangan UP/DOWN palsu terus-terusan — terukur 18 kali dalam 3 detik. Rekamannya
jadi serpihan.

Tombol yang ditahan **sendirian** cuma ngirim DOWN berulang, nggak pernah UP.
Makanya default lokalnya `right ctrl`. Pilihan lain: `f8`, `right shift`.

### Panjang jawaban

qwen nggak nurut sama "one or two sentences". Yang **terbukti manjur** batas kata
eksplisit, bukan batas token:

| | Kata | Audio |
|---|---:|---:|
| Apa adanya | 41,0 | 18,4 dtk |
| `REPLY_MAX_WORDS=25` | 17,8 | 9,1 dtk |
| + minta kalimat pendek | **13,0** | **7,1 dtk** |

Cap token nggak nambah apa-apa (17,0 vs 17,0) dan motong di tengah kata, jadi
`OLLAMA_NUM_PREDICT` cuma jaring pengaman buat yang bener-bener ngelantur.

Paling kerasa di mode sesi: tanpa barge-in, balasan panjang nggak bisa dipotong.

## Catatan resource (8 GB VRAM)

Cuma **satu model yang nempatin GPU**:

| Bagian | Di mana | VRAM | RAM | Kecepatan |
|---|---|---:|---:|---|
| Silero VAD | CPU | 0 | ~50 MB | 190x realtime |
| Parakeet TDT 0.6B | CPU | 0 | ~2,2 GB | ~0,4 dtk/kalimat |
| qwen2.5:7b | GPU | ~5 GB | — | ~2–4 dtk |
| Kokoro-82M | CPU | 0 | ~0,4 GB | ~4x realtime |

Parakeet di CPU **dengan sengaja**: di GPU cuma ~0,2 detik lebih cepat, tapi
ngerebut VRAM dari qwen — dan qwen yang kegeser ke RAM jauh lebih mahal.

Ollama pakai keep-alive 5 menit, jadi pertanyaan pertama setelah nganggur lama
kena muat ulang (~7 detik). `OLLAMA_KEEP_ALIVE=-1` bikin instan, bayarannya
model nempel di VRAM terus.

## Struktur

```
agent/
  main.py          hotkey, mode sesi, orkestrasi pipeline, logging
  config.py        semua konstanta dari .env + penegakan mode offline
  audio.py         rekam mic, playback bersambung, beep
  vad.py           deteksi suara per frame (Silero)
  stt.py           Parakeet / Whisper — muat & lepas otomatis
  llm.py           Ollama / Claude, streaming per kalimat
  tts.py           Kokoro / Piper
  teks.py          penormal teks
  cek_offline.py   verifikasi kesiapan jalan tanpa jaringan
models/            bobot Kokoro & Piper (gitignore)
memory/            agent.lock (gitignore)
scripts/           setup, autostart, status
logs/              agent.log (rotating, 1 MB x 4)
```

~2.300 baris. **[ARSITEKTUR.md](ARSITEKTUR.md)** menjelaskan alasan di balik tiap
keputusan, lengkap dengan angka pengukurannya.

Tiap modul bisa dites sendiri:

```powershell
.\.venv-agent\Scripts\python.exe -m agent.vad          # deteksi suara
.\.venv-agent\Scripts\python.exe -m agent.stt          # transkrip
.\.venv-agent\Scripts\python.exe -m agent.tts "hello"  # suara
.\.venv-agent\Scripts\python.exe -m agent.llm "hi"     # otak
.\.venv-agent\Scripts\python.exe -m agent.audio        # mic & speaker
.\.venv-agent\Scripts\python.exe -m agent.teks         # penormal
```

## Bukti offline

```powershell
.\.venv-agent\Scripts\python.exe -m agent.cek_offline
```

Ngecek setelan, bobot di disk, model beneran kemuat, dan Ollama nyaut. Diuji
dengan **semua soket non-localhost diblokir**: rantai penuh jalan 7/7 langkah,
**nol** percobaan koneksi keluar.

## Troubleshooting

| Gejala | Cek |
|---|---|
| Hotkey nggak ngapa-ngapain | `scripts\status.ps1` — ada agent dobel? build lama? |
| Nggak ada suara sama sekali | Output device pindah. `python -m agent.audio` |
| Rekaman kepotong-potong | Pakai tombol tunggal, bukan kombinasi |
| Kalimat kepotong di mode sesi | Naikin `VAD_SILENCE_MS` |
| Jawaban lama banget | Pertama abis nganggur emang ~7 dtk (Ollama muat ulang) |
| Perbaikan kelihatan nggak jalan | Cek baris `build:` — proses lama megang kode lama |

## Yang belum ada

Semuanya dicopot pas mulai ulang, dan riwayat git-nya masih lengkap kalau mau
diambil lagi:

- **Kalender** — baca jadwal, bikin acara (ICS lokal & Google Calendar)
- **Daftar tugas** — catat, tandai selesai, milih mana yang dikerjain
- **Memori antar-sesi** — riwayat & fakta yang bertahan setelah restart
- **Jawaban pasti tanpa LLM** — jam, tanggal, jadwal dihitung Python
- **Pengurai waktu deterministik** — Inggris (`time_en`) & Indonesia (`waktu_id`)
- **Wake word** — masuk sesi tanpa nyentuh tombol
- **Motong omongan agent** — butuh peredam gema
