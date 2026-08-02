# personal-agent

Voice assistant lokal buat Windows. Jalan diam-diam di background, nggak ada window.
Tahan hotkey → ngomong → lepas → dia jawab lewat speaker. Semua Bahasa Indonesia,
semua diproses di mesin sendiri (nggak ada yang dikirim ke cloud).

```
pencet hotkey → rekam mic → faster-whisper (id) → Claude API atau Ollama → Piper (id_ID) → speaker
```

Otaknya bisa dipilih: **Claude API** (default — lebih pintar dan cepat, butuh internet
dan kunci berbayar) atau **Ollama lokal** (gratis, jalan offline, lebih lambat).
STT dan TTS selalu lokal, jadi suaramu nggak pernah keluar dari mesin ini.

## Yang dibutuhin

- Windows 10/11
- [Ollama](https://ollama.com/download) udah terinstall & jalan
- Python 3.11 (dikelola lewat [mise](https://mise.jdx.dev/); ada fallback venv biasa di bawah)
- Mikrofon + speaker

## Install

```powershell
git clone <repo> personal-agent
cd personal-agent

# 1. Python 3.11 + venv (.venv-agent)
mise trust
mise install
mise where python@3.11   # catat path-nya
& "<path-di-atas>\python.exe" -m venv .venv-agent

# 2. Dependencies
.\.venv-agent\Scripts\python.exe -m pip install -e .

# 3. Model Ollama + voice Piper (~4.7 GB + 60 MB)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# 4. (opsional) konfigurasi
copy .env.example .env
```

<details>
<summary>Tanpa mise (fallback)</summary>

Install Python 3.11 atau 3.12 dari [python.org](https://www.python.org/downloads/), terus:

```powershell
py -3.11 -m venv .venv-agent
.\.venv-agent\Scripts\python.exe -m pip install -e .
```

Python 3.13+ belum dipakai karena `ctranslate2` (mesinnya faster-whisper) belum
punya wheel yang stabil di situ.
</details>

Model Whisper (~460 MB) ke-download otomatis pas pertama kali dipakai.

## Jalanin

```powershell
.\.venv-agent\Scripts\python.exe -m agent.main
```

Terus **pencet hotkey sekali** buat mulai, ngomong, **pencet lagi** buat berhenti.
(Mau gaya tahan-sambil-ngomong? Set `HOTKEY_MODE=hold` di `.env`.)

Karena nggak ada window, feedback-nya bunyi:

| Bunyi | Artinya |
|---|---|
| beep tinggi (880 Hz) | mulai rekam |
| beep sedang (560 Hz) | selesai rekam, lagi mikir |
| dua ketuk pendek (420 Hz) | kedengeran, tapi lagi sibuk — pencetanmu diabaikan |
| beep rendah panjang (240 Hz) | ada yang gagal — cek `logs/agent.log` |

Kalau modelnya lagi terlepas (lihat bagian memori GPU di bawah), dia bakal ngomong
*"sebentar ya, lagi nyiapin"* dulu — pemuatan bisa makan 13–35 detik kalau file-nya
dingin, dan diam selama itu nggak bisa dibedain dari mati.

Pertanyaan pertama agak lama (Ollama naikin model ke VRAM). Selanjutnya cepet.

## Jalan otomatis pas login

```powershell
# PowerShell as Administrator
powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1
```

Bikin task `PersonalAgent` di Task Scheduler: trigger **At log on**, action
`pythonw.exe -m agent.main` (pythonw = tanpa console window). Default-nya tanpa
elevasi karena backend `pynput` nggak butuh admin — tambahin `-Elevated` kalau
kamu pindah ke `HOTKEY_BACKEND=keyboard`.

```powershell
powershell -File scripts\status.ps1                  # nyala atau nggak?
Start-ScheduledTask -TaskName PersonalAgent          # nyalain
Stop-ScheduledTask -TaskName PersonalAgent           # matiin
Get-Content logs\agent.log -Tail 20 -Wait            # lihat aktivitas langsung
powershell -File scripts\install-startup.ps1 -Uninstall   # hapus task
```

`status.ps1` nunjukin: agent nyala/mati, PID, udah jalan berapa lama, autostart
terpasang apa nggak, VRAM, model STT lagi dimuat apa dilepas, isi memori, dan
kapan terakhir kamu ngomong sama dia.

## Konfigurasi

Semua default ada di [agent/config.py](agent/config.py), bisa ditimpa lewat file `.env`
(lihat [.env.example](.env.example)). Yang sering diutak-atik:

| Variabel | Default | Keterangan |
|---|---|---|
| `HOTKEY` | `ctrl+space` | Lihat catatan di bawah — **pakai tombol tunggal** |
| `HOTKEY_MODE` | `toggle` | `toggle` = pencet-pencet, `hold` = tahan sambil ngomong |
| `HOTKEY_BACKEND` | `pynput` | Alternatif: `keyboard` (butuh admin, tapi bisa nelen hotkey) |
| `LLM_BACKEND` | `claude` | `claude` (API) atau `ollama` (lokal) |
| `ANTHROPIC_API_KEY` | — | Wajib kalau `LLM_BACKEND=claude` |
| `CLAUDE_MODEL` | `claude-opus-5` | Mau lebih murah/cepat: `claude-haiku-4-5` |
| `CLAUDE_EFFORT` | `low` | `low`/`medium`/`high`/`xhigh`/`max` |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model apa pun yang udah di-`ollama pull` |
| `WHISPER_MODEL` | `small` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cuda` **jauh** lebih cepat — lihat di bawah |
| `WHISPER_COMPUTE` | `int8` | `float16` kalau pakai `cuda` |
| `PIPER_VOICE` | `models/id_ID-news_tts-medium.onnx` | Path voice `.onnx` |

### Pilih hotkey: pakai tombol tunggal

Kalau tombol ditahan **bareng modifier** (`Ctrl+Alt+Space`, `Ctrl+Space`), Windows
ngirim pasangan UP/DOWN palsu terus-terusan selama ditahan — kadang cuma berjarak
0.01 detik. Efeknya rekaman kepotong jadi serpihan dan yang ketangkep cuma sepenggal
kalimat. Tombol yang ditahan **sendirian** cuma ngirim DOWN berulang, nggak pernah UP,
jadi aman.

Makanya default lokalnya `right ctrl`. Pilihan bagus lain: `f8`, `right shift`,
`right alt`. Bisa spesifik sisi (`right ctrl`) atau bebas (`ctrl` = kiri atau kanan).

`RELEASE_GRACE_SECONDS` (default 0.6) tetap ada sebagai jaring pengaman: rekaman baru
distop kalau tombol kedeteksi lepas selama segitu lama. Kalau kamu tetap mau pakai
kombinasi dan rekamannya kepotong, naikin angka ini — tapi lepasnya jadi terasa lelet.

### Akurasi transkrip

`WHISPER_PROMPT` berisi daftar kosakata yang sering kamu pakai. Whisper mencondongkan
tebakannya ke situ, **tanpa biaya waktu**. Diukur di rekaman asli, ini nurunin WER dari
36% jadi 25% — nama teknis kayak "Whisper" dan "Ollama" yang tadinya jadi "hispar" dan
"olama" langsung benar. Tambahin nama orang, tempat, atau tool yang sering kamu sebut.

### Whisper di GPU (NVIDIA) — perbedaannya besar

Kalau punya GPU NVIDIA, ini peningkatan terbesar yang bisa didapat. Diukur di
rekaman asli, model `medium`:

| | CPU int8 | GPU float16 |
|---|---|---|
| Per kalimat | 6.64 detik | **0.50 detik** |
| WER | 8% | 8% |

Sama persis akurasinya, 13× lebih cepat. Caranya:

```powershell
.\.venv-agent\Scripts\python.exe -m pip install -e ".[gpu]"
```

lalu di `.env`:

```
WHISPER_DEVICE=cuda
WHISPER_COMPUTE=float16
```

Makan ~1.9 GB VRAM selama model dimuat. Kalau kamu juga pakai `LLM_BACKEND=ollama`,
dua-duanya rebutan VRAM — di kartu 8 GB itu masih muat tapi mepet.

**Model dilepas otomatis kalau nganggur.** Bobot model itu data diam, bukan proses —
kepegang di VRAM sepanjang agent hidup walaupun cuma kepakai ~0.5 detik per kalimat.
Di GPU pribadi yang dipakai buat hal lain, itu sayang. Jadi setelah
`WHISPER_IDLE_UNLOAD_SECONDS` (default 900 = 15 menit) tanpa dipakai, modelnya
dilepas dan VRAM balik. Ollama melakukan hal yang sama secara default (keep-alive
5 menit). Set `0` kalau kamu mau model nempel terus.

**Harga muat ulangnya nggak murah, dan bukan rata-rata.** Terukur di mesin uji:
2–7 detik kalau dimuat tak lama setelah dipakai, tapi **13–35 detik** setelah
nganggur berjam-jam — dan nganggur lama persis kondisi yang memicu pelepasan.
Jadi yang kamu bayar hampir selalu kasus terburuknya. Tiga hal meredamnya:
model mulai dimuat begitu kamu menekan hotkey (barengan kamu ngomong), dia
ngomong *"sebentar ya"* supaya diamnya nggak terasa seperti mati, dan pencetan
ulang dijawab dua ketuk. Kalau tetap kerasa lama, `WHISPER_IDLE_UNLOAD_SECONDS=0`.

Catatan: yang balik ~1.9 GB dari 2.0 GB. Sisanya konteks CUDA yang baru lepas pas
prosesnya mati — itu wajar dan nggak numpuk.

**Catatan Windows:** DLL CUDA dari pip nggak ada di `PATH`, dan ctranslate2
nyarinya lewat situ (bukan lewat `add_dll_directory`). [stt.py](agent/stt.py)
ngurus itu otomatis — kalau tetap muncul `cublas64_12.dll is not found`,
berarti paket `[gpu]`-nya belum kepasang.

Kalau masih kurang akurat, naikin `WHISPER_MODEL` ke `medium`. Buat ngukur sendiri, nyalain
`SAVE_RECORDINGS=true` — tiap rekaman disimpan ke `logs/rec/*.wav`, jadi setelan bisa
diuji ulang di suara asli tanpa perlu ngomong berkali-kali. Matiin lagi kalau selesai.

### Pilih otak: Claude API atau Ollama

Default `claude`. Ambil kunci di
[console.anthropic.com](https://console.anthropic.com/settings/keys), taruh di `.env`:

```
LLM_BACKEND=claude
ANTHROPIC_API_KEY=sk-ant-...
```

Ini **berbayar per pakai** — tiap pertanyaan kena tarif token. Buat obrolan pendek
biayanya kecil, tapi tetap ada. `CLAUDE_EFFORT=low` dipakai sebagai default karena
balasan 1–2 kalimat nggak butuh mikir dalam, dan effort rendah bikin jeda lebih pendek.

Mau gratis dan offline? Ganti ke `LLM_BACKEND=ollama` — nggak perlu kunci, tapi
jawabannya lebih lemot dan kurang nyambung.

**Ganti model Ollama:** `ollama pull <model>` terus set `OLLAMA_MODEL=<model>` di `.env`.

**Ganti voice Piper:** ambil dari
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) — download
`.onnx` + `.onnx.json` ke `models/`, terus set `PIPER_VOICE` ke file `.onnx`-nya.
Bisa juga lewat `scripts\setup.ps1 -Voice <nama> -VoicePath <folder-di-repo-hf>`.

**Soal backend hotkey.** Default `pynput`: jalan tanpa admin, tapi nggak bisa nelen
hotkey-nya — `Ctrl+Space` tetep diterusin ke aplikasi yang lagi fokus (biasanya nggak
kerasa, kecuali app-nya emang pakai kombinasi itu). Backend `keyboard` bisa nelen
hotkey (`HOTKEY_SUPPRESS=true`), tapi butuh admin **dan** di sebagian mesin —
termasuk mesin tempat ini dites — `add_hotkey`-nya nggak pernah kepanggil sama
sekali. Kalau mau coba: set `HOTKEY_BACKEND=keyboard` dan pasang startup task pakai
`install-startup.ps1 -Elevated`.

## Catatan resource (8 GB VRAM)

Ini tergantung otak mana yang dipakai:

**Pakai Claude API (default).** GPU-nya bebas buat Whisper sendirian — pakai
`cuda` + `float16` (~2.5 GB). Piper CPU-only, 0 VRAM. Ini kombinasi tercepat:
STT ~0.5 detik, LLM ~1 detik.

**Pakai Ollama lokal.** `qwen2.5:7b` makan ~5 GB, Whisper `medium` di GPU ~2.5 GB —
total 7.5 GB dari 8 GB. Muat, tapi mepet. Kalau mulai bermasalah, turunin Whisper
ke `WHISPER_DEVICE=cpu` atau pakai model Ollama yang lebih kecil. Ollama juga pakai
keep-alive 5 menit, jadi pertanyaan pertama kena warmup — set `OLLAMA_KEEP_ALIVE=-1`
kalau mau instan (bayarannya model nempel di VRAM terus).

## Struktur

```
agent/
  config.py   konstanta + system prompt
  audio.py    rekam mic, playback, beep
  stt.py      faster-whisper
  llm.py      Ollama chat + history (in-memory)
  tts.py      Piper
  main.py     hotkey + wiring pipeline + logging
models/       voice Piper (gitignore)
scripts/      setup.ps1, install-startup.ps1
logs/         agent.log (rotating, 1 MB x 4)
```

Tiap modul bisa dites sendiri:

```powershell
.\.venv-agent\Scripts\python.exe -m agent.tts "halo dunia"       # teks -> speaker
.\.venv-agent\Scripts\python.exe -m agent.stt                    # rekam 5 detik -> teks
.\.venv-agent\Scripts\python.exe -m agent.stt rekaman.wav        # file -> teks
.\.venv-agent\Scripts\python.exe -m agent.llm "apa kabar"        # teks -> teks
.\.venv-agent\Scripts\python.exe -m agent.audio                  # list device + tes beep
```

## Troubleshooting

| Gejala | Kemungkinan |
|---|---|
| Hotkey nggak nyaut | Pastiin `HOTKEY_BACKEND=pynput`. Backend `keyboard` butuh admin dan nggak jalan di semua mesin |
| Nggak ada suara sama sekali | Cek output device: `python -m agent.audio` |
| Suara agent kepotong di ujung | Naikin `PLAYBACK_PAD_SECONDS` (Windows motong ~100 ms tiap playback) |
| Transkrip kosong terus | Mic salah/ke-mute. Cek input device di daftar yang sama |
| Cuma sepenggal kalimat yang ketangkep | Hotkey-nya kombinasi — ganti ke tombol tunggal (lihat atas) |
| Beep rendah tiap dipakai | Buka `logs\agent.log`, error lengkapnya di situ |
| Jawaban lama banget | Cek log: kalau `STT ... detik proses` yang gede, pindahin Whisper ke GPU |
| `cublas64_12.dll is not found` | Paket GPU belum kepasang: `pip install -e ".[gpu]"` |

## Memori antar-sesi

Agent inget kamu walaupun sudah restart. Dua lapis dengan umur beda, disimpan di
`memory/` (gitignore):

| File | Isi | Umur |
|---|---|---|
| `history.json` | Pesan mentah, buat nyambungin obrolan yang kepotong restart | Dibuang setelah `HISTORY_MAX_AGE_HOURS` (default 12 jam) |
| `facts.md` | Hal yang layak diingat lama: nama, jurusan, preferensi | Sampai kamu hapus |

Riwayat sengaja punya batas umur — nyambungin obrolan itu berguna dalam hitungan
jam, tapi 20 pesan dari minggu lalu justru bikin salah konteks. Fakta yang bertahan.

`facts.md` itu teks polos, boleh kamu baca dan sunting sendiri; suntingannya
langsung kepakai tanpa restart. Jumlahnya dibatasi `FACTS_MAX_ITEMS` (default 30)
karena semuanya ikut ke tiap permintaan.

**Cara menghapus:** bilang *"lupakan semua"* atau *"hapus memori"*. Frasa ini
dicocokkan lokal, bukan lewat LLM — perintah yang nggak bisa dibatalkan nggak
boleh gantung pada tebakan model. Bisa juga hapus foldernya, atau matikan total
dengan `MEMORY_ENABLED=false`.

Penyaringan fakta itu satu panggilan API tambahan per obrolan, dijalanin di latar
belakang **setelah** dia menjawab — jadi nggak nambah jeda yang kamu rasakan.

## Kalender (baca aja)

Agent bisa jawab *"besok ada kelas apa?"*, *"kelas berikutnya di mana?"* kalau
kamu kasih link ICS di `.env`:

```
CALENDAR_ICS_URL=https://...
CALENDAR_TZ=Australia/Canberra
```

Ambil link-nya di: Google Calendar → Settings → **Secret address in iCal format**,
Outlook → Calendar → Share → Publish, atau MyTimetable ANU → Export/Subscribe.

> ⚠️ **URL itu setara kata sandi** — siapa pun yang punya bisa baca seluruh
> jadwalmu tanpa login. Simpan di `.env` (gitignore), jangan di kode. Kalau
> bocor, cabut dan buat ulang dari setelan kalendermu.

**Cuma bisa baca.** Link ICS itu terbitan satu arah; nggak ada mekanisme buat
nulis balik, jadi agent nggak bisa bikin acara. Itu butuh API sungguhan dengan
OAuth (Google Calendar API / Microsoft Graph) — belum dibangun.

Jadwal ditarik paling sering tiap `CALENDAR_CACHE_MINUTES` (default 60) di latar
belakang, dan salinannya disimpan di `memory/calendar.ics` supaya restart nggak
perlu nunggu jaringan dan tetap jalan waktu offline.

### Kenapa formatnya berkolom

Agenda yang diselipin ke prompt berbentuk `jam | kode & nama | jenis | lokasi`,
dan baris `KELAS BERIKUTNYA` dihitung di Python — bukan diserahkan ke model.
Dua-duanya bukan hiasan: dengan format padat tanpa pemisah, model terbukti
mengambil nama matkul dari satu baris tapi lokasinya dari baris tetangga, dan
salah menentukan kelas terdekat. Prompt-nya juga melarang bentuk jam "setengah
sembilan" karena sempat kejadian meleset setengah jam dari 09:00.

## Belum ada (tahap berikutnya)

Wake word, notifikasi proaktif, bikin acara kalender (butuh OAuth), integrasi LMS,
baca daftar tugas otomatis.
