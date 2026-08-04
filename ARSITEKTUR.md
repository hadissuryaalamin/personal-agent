# Arsitektur personal-agent

Dokumen ini menjelaskan **bagaimana** sistem ini bekerja dan **kenapa** dibangun
begini. Untuk cara memakai dan menyetel, lihat [README.md](README.md).

Sebagian besar keputusan di sini datang dari pengukuran, bukan preferensi.
Angkanya disertakan supaya bisa ditinjau ulang kalau keadaan berubah.

---

## 1. Gambaran besar

Asisten suara yang jalan diam di latar belakang Windows. Tekan hotkey, ngomong,
tekan lagi — dia menjawab lewat speaker. Bahasa Inggris, sepenuhnya offline.

```
                    ┌──────────────────────────────────────┐
   tekan hotkey ───▶│  main.py — listener & orkestrator    │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   ┌─────────┐   ┌─────┐      ┌──────────┐                ┌──────────┐
   │ audio   │──▶│ vad │─────▶│   stt    │ suara→teks ───▶│   llm    │
   │ mic     │   │batas│      │ Parakeet │                │  qwen2.5 │
   └─────────┘   └─────┘      └──────────┘                └────┬─────┘
        ▲                                                      │
        │                     ┌──────────┐                     │
        └──── speaker ────────│   tts    │◀── teks→suara ──────┘
                              │  Kokoro  │
                              └──────────┘
```

Empat model untuk empat tugas: **Silero VAD tahu kapan kamu selesai bicara**
(CPU, 0 VRAM), **Parakeet mendengar** (CPU, 0 VRAM), **qwen2.5:7b berpikir**
(GPU, ~5 GB), **Kokoro berbicara** (CPU, 0 VRAM). Cuma satu yang menempati GPU —
lihat §4.7.

Tidak ada yang lain. Kalender, daftar tugas, memori antar-sesi, dan penjawab
deterministik pernah ada dan **sengaja dicopot** saat memulai ulang; riwayat
git-nya masih lengkap kalau mau diambil kembali.

### Dua lapis, bukan satu

Yang jalan terus-menerus hanyalah **listener hotkey** — 28 MB RAM, 0,02% CPU,
nol VRAM. Semua yang berat (STT, LLM) baru dimuat saat dipakai dan dilepas
lagi saat nganggur. Ini disengaja: agent hidup 24 jam tapi dipakai beberapa
menit sehari, jadi menahan model di memori sepanjang hari itu pemborosan.

---

## 2. Peta modul

| Modul | Baris | Tanggung jawab |
|---|---:|---|
| `main.py` | 672 | Listener hotkey, mode sesi, orkestrasi pipeline, logging |
| `llm.py` | 288 | Dua backend otak (Ollama/Claude), streaming per kalimat |
| `stt.py` | 275 | Parakeet / Whisper: muat, transkrip, lepas saat nganggur |
| `audio.py` | 271 | Rekam mic, playback bersambung, beep |
| `config.py` | 257 | Semua konstanta dari `.env` + penegakan mode offline |
| `vad.py` | 226 | Deteksi suara per frame (Silero) |
| `tts.py` | 142 | Kokoro / Piper: teks -> WAV |
| `cek_offline.py` | 128 | Verifikasi kesiapan jalan tanpa jaringan |
| `teks.py` | 55 | Penormal teks |

Total ~2.300 baris.

### Arah ketergantungan

```
main ──▶ audio, vad, stt, llm, tts, teks
semuanya ──▶ config
```

Satu arah, tanpa siklus. `config` tidak bergantung pada apa pun; `main` tidak
diimpor siapa pun. Tiap modul model (`vad`, `stt`, `llm`, `tts`) berdiri sendiri
dan bisa dijalankan terpisah lewat `python -m agent.<modul>`.

---

## 3. Alur satu percakapan

```
1. Hotkey ditekan
   └─ jika model STT terlepas, mulai memuatnya SEKARANG (paralel)
   └─ buka stream mic, baru bunyikan beep (non-blocking)

2. Hotkey ditekan lagi
   └─ rekaman berhenti

3. Transkrip (Parakeet)
   └─ jika model belum siap: ucapkan "One moment, just getting ready."

4. Kirim ke LLM
5. Jawaban → Kokoro → speaker, KALIMAT PER KALIMAT (§4.11)
```

Dalam **mode sesi** (`SESSION_MODE=true`), langkah 1–2 diganti: hotkey membuka
sesi sekali, lalu batas tiap ucapan ditentukan VAD. Langkah 3–5 sama persis —
dipakai bareng lewat `_route_and_reply()`.

```
1. Hotkey ditekan → sesi terbuka
2. Ulang sampai tutup:
   └─ tunggu suara, rekam sampai diam 800 ms
   └─ langkah 3-5 di atas
   └─ mic DITUTUP selama agent bicara (§4.9)
3. Tutup kalau: hotkey ditekan lagi, diam 30 detik, atau "goodbye"
```

`_route_and_reply()` sekarang cuma meneruskan ke LLM. Bentuknya dipertahankan
sebagai fungsi terpisah karena **di situlah percabangan niat akan menempel**
kalau nanti ada fitur lagi — dan urutan pengecekannya terbukti gampang salah:
*"add a task to finish the assignment"* juga cocok dengan pola *"add ..."* untuk
membuat acara, jadi tugas harus dicek sebelum acara.

---

---

## 4. Keputusan desain, dengan alasannya

Setiap keputusan di bawah ini pernah diukur atau muncul dari kegagalan nyata.

### 4.1 Hotkey harus satu tombol

Tombol yang ditahan **bersama modifier** (`Ctrl+Space`) membuat Windows
mengirim pasangan UP/DOWN palsu terus-menerus — terukur 18 kali dalam 3 detik,
ada yang berjarak 0,01 detik. Rekaman jadi terpotong-potong; yang tertangkap
cuma serpihan kalimat.

Tombol yang ditahan **sendirian** hanya mengirim DOWN berulang, tanpa UP. Karena
itu defaultnya `right ctrl`.

`RELEASE_GRACE_SECONDS` (0,6 dtk) tetap ada sebagai jaring pengaman: rekaman
baru berhenti kalau tombol terdeteksi lepas selama itu.

### 4.2 Backend hotkey `pynput`, bukan `keyboard`

Library `keyboard` tidak menerima event sama sekali di mesin uji — hook
low-level-nya jalan (terbukti lewat `_winkeyboard.listen` langsung), tapi
dispatch `add_hotkey`/`hook`-nya diam total. `pynput` jalan tanpa admin.

### 4.3 Pipeline jalan di thread terpisah

Callback listener berjalan di thread yang sama dengan pemroses event keyboard.
Kalau di-block, event lepas mengantre di belakang dan `is_held()` tersangkut
`True` selamanya.

### 4.4 Beep mulai bersifat non-blocking

Membuka stream mic makan ~0,5 detik, jadi beep dibunyikan **setelah** stream
siap. Tapi beep itu sendiri harus non-blocking: buffer mic dibuang setelah
`on_ready` selesai, dan user mulai bicara begitu mendengar nada — kalau beep
memblokir, kata pertama ikut terbuang.

### 4.5 Playback diberi bantalan hening

Terukur **119 ms hilang** dari nada 2 detik tanpa bantalan: device output butuh
waktu membuka (awal terpotong) dan buffer internalnya masih berbunyi saat
`sd.wait()` sudah menganggap selesai (akhir terpotong).
`PLAYBACK_PAD_SECONDS=0.2` menutupnya.

### 4.6 Model dilepas saat nganggur

Bobot model itu **data diam**, bukan proses: Whisper memegang ~1,9 GB VRAM
sepanjang agent hidup padahal utilisasi GPU cuma 1% saat idle. Di GPU pribadi
yang dipakai untuk hal lain, itu sayang.

Bayarannya nyata dan bukan rata-rata: muat ulang **2–7 detik** kalau file masih
hangat di cache, tapi **13–35 detik** setelah nganggur berjam-jam — dan nganggur
lama persis kondisi yang memicu pelepasan. Tiga hal meredamnya:

1. Model mulai dimuat begitu hotkey ditekan (paralel dengan user bicara)
2. Agent mengucapkan *"One moment"* supaya diamnya tidak terasa seperti mati
3. Penekanan ulang dijawab dua ketuk pendek

### 4.7 Hanya satu model yang menempati GPU

GPU di mesin ini 8 GB. qwen2.5:7b sendiri sudah ~5 GB. Kalau STT ikut naik ke
GPU, sisanya tinggal ~1 GB dan qwen mulai kegeser ke RAM — yang jauh lebih mahal
daripada waktu yang dihemat.

| Bagian | Di mana | VRAM | RAM | Kecepatan |
|---|---|---:|---:|---|
| Parakeet TDT 0.6B | CPU | 0 | ~2,2 GB | ~0,40 dtk/kalimat |
| qwen2.5:7b | GPU | ~5 GB | — | ~2–4 dtk |
| Kokoro-82M | CPU | 0 | ~0,4 GB | ~4x realtime |

Parakeet di GPU hanya ~0,2 detik lebih cepat. Menukar 0,2 detik dengan tekanan
VRAM pada qwen itu perdagangan yang rugi, jadi Parakeet sengaja dikunci di CPU.

**Ollama warm lebih cepat daripada Claude.** Pengukuran pertama menyimpulkan
Ollama "2x lebih lambat"; yang sebenarnya terukur adalah **pemuatan model**,
bukan inferensi. Ollama memakai keep-alive 5 menit. Saat warm: 1,8–2,2 detik,
melawan ~2,8 detik untuk Claude yang harus menempuh jaringan.

### 4.8 Mode offline ditegakkan saat startup, di `config`

`OFFLINE_MODE=true` menolak backend yang butuh jaringan — `LLM_BACKEND=claude`,
`CALENDAR_ICS_URL`, `GOOGLE_CREDENTIALS_FILE` — dan agent berhenti dengan kode 2.

Dua keputusan kecil di sini penting:

**Ditegakkan saat startup, bukan saat dipakai.** Agent jalan tanpa window.
Kegagalan jaringan di tengah percakapan hanya terdengar seperti agent membisu;
kegagalan di startup tercatat jelas di log.

**Diletakkan di `config.py`, bukan `main.py`.** Pemeriksaan yang cuma ada di satu
entry point akan bocor lewat skrip dan tes. `config.wajib_offline()` bisa
dipanggil siapa saja.

Buktinya bisa dijalankan: `python -m agent.cek_offline` memverifikasi setelan,
bobot di disk, model benar-benar termuat, dan Ollama menyahut. Diuji dengan
seluruh soket non-localhost diblokir, pipeline penuh berjalan 8/8 langkah dengan
**nol** percobaan koneksi keluar.

### 4.9 Mode sesi: mic ditutup selama agent bicara

Dalam mode sesi, batas kalimat ditentukan VAD (Silero, lewat `onnx_asr` —
model yang sama dengan Parakeet, jadi nol paket baru). Konsekuensinya: mic
hidup terus, dan mic yang hidup saat speaker berbunyi akan **mendengar agent
sendiri**, lalu mentranskripnya sebagai ucapan user. Agent mengobrol dengan
dirinya sendiri sampai sesi ditutup paksa.

Tiga jalan keluar; yang dipilih paling sederhana:

| | Bisa memotong? | Butuh headphone? | Risiko |
|---|---|---|---|
| **Mic ditutup saat bicara** | tidak | tidak | nyaris nol |
| Mic hidup terus | ya | **ya** | pakai speaker = kacau |
| Peredam gema | ya | tidak | paling rumit, paling rawan |

Yang dipilih baris pertama. Harganya nyata: balasan panjang tidak bisa
dipotong, kamu terkunci mendengarkan sampai habis. Itulah sebabnya
`REPLY_MAX_WORDS` menjadi wajib, bukan pemanis — lihat §4.10.

VAD-nya juga membedakan **tiga** alasan berhenti, bukan satu. "User menutup
sesi", "sesi mati sendiri karena sepi", dan "suaranya terlalu pendek" harus
terpisah: kalau ketiganya dibaca sama, batuk sekali akan menutup sesi.
Suara terlalu pendek ditelan di dalam VAD dan tidak pernah sampai ke pemanggil.

Tenggat sepi dihitung dari awal dan **tidak** di-reset oleh suara pendek. Kalau
di-reset, ruangan berisik bisa menahan sesi terbuka selamanya tanpa kamu bicara
sekali pun — dan model ikut tertahan di memori selama itu.

### 4.10 Panjang balasan: batas kata, bukan batas token

Terukur pada qwen2.5:7b:

| | Kata | Audio |
|---|---:|---:|
| Apa adanya | 41,0 | 18,4 dtk |
| Batas 25 kata di prompt | 17,8 | 9,1 dtk |
| + minta kalimat pendek | **13,0** | **7,1 dtk** |
| Batas 25 kata + cap 60 token | 17,0 | 9,5 dtk |

Cap token (`num_predict`) **tidak menambah apa pun** di atas batas kata — 17,0
lawan 17,0 — dan memotong paksa di tengah kata, yang terdengar seperti agent
tercekik. Jadi yang dipakai batas kata; `num_predict` tetap ada tapi hanya
sebagai jaring pengaman untuk yang benar-benar mengigau.

Panjang **kalimat** diatur terpisah dari panjang **jawaban**, dan itu bukan
duplikasi: TTS baru bisa mulai berbunyi setelah satu kalimat utuh, jadi satu
kalimat 36 kata menunda bunyi pertama sama saja dengan tidak streaming.

### 4.11 Streaming: yang dipangkas diamnya, bukan totalnya

Jawaban dipotong per kalimat dan langsung disintesis, sementara model masih
mengarang kalimat berikutnya.

| Pertanyaan | Bunyi pertama, tunggu selesai | Bunyi pertama, streaming |
|---|---:|---:|
| "What classes do I have today?" | 17,4 dtk | 8,2 dtk |
| "What should I work on today?" | 11,8 dtk | 3,4 dtk |

Turun 53–71%. Waktu totalnya nyaris tidak berubah — yang hilang adalah
diamnya, dan itu yang membedakan percakapan terasa hidup atau terasa nge-lag.

Pemotongnya harus tahu bahwa `3 p.m.` bukan akhir kalimat. Tanpa itu, TTS
mengucapkan "three pee." lalu berhenti sejenak sebelum "em." — dan `3 p.m.`
justru bentuk yang dikeluarkan Parakeet.

Playback-nya lewat `audio.Speaker`, bukan `play_wav()` berulang. `play_wav()`
menyisipkan bantalan hening di **kedua** ujung tiap potongan, jadi tiap
sambungan kalimat kena 0,4 detik hening — terukur 0,83 detik hening berlebih
pada balasan tiga kalimat. `Speaker` memberi bantalan sekali di awal dan sekali
di akhir seluruh ucapan.

## 5. Penyimpanan

Nyaris tidak ada, dan itu disengaja.

| File | Isi | Umur |
|---|---|---|
| `memory/agent.lock` | Kunci satu-instance | Dilepas OS saat proses mati |
| `logs/agent.log` | Log | Rotating, 1 MB x 4 |

**Riwayat percakapan tidak disimpan ke disk.** Dulu disimpan, dan berkali-kali
membuat model mengarang tentang data yang sudah berubah — sampai mengaku telah
memindahkan tenggat yang acaranya bahkan sudah dihapus. Riwayat berguna untuk
menyambung percakapan, tetapi bukan sumber fakta. Restart = mulai bersih.

Model besar tetap di cache HuggingFace (`~/.cache/huggingface`), bukan di repo:
Parakeet dan Silero VAD diambil `onnx-asr` lewat nama model, bukan lewat path.

---

---

## 6. Bagian yang bisa ditukar

Empat titik dirancang bisa diganti lewat `.env` tanpa menyentuh kode:

**Otak** — `LLM_BACKEND=ollama|claude`. Keduanya di balik `chat()` dan
`chat_stream()` yang sama. Backend yang tidak mendukung streaming tetap jalan:
`chat_stream()` bawaannya mengeluarkan satu potong utuh, jadi pemanggilnya tidak
perlu tahu bedanya.

**Pendengaran** — `STT_BACKEND=parakeet|whisper`. Keduanya di balik
`get_model()` / `transcribe()` / `unload_model()` yang sama. Parakeet cuma
Inggris dan tidak menerima prompt kosakata; Whisper multibahasa tapi menuntut
VRAM kalau dinaikkan ke GPU.

**Suara** — `TTS_BACKEND=kokoro|piper`, keduanya di balik `speak(text) -> WAV`.
Sample rate berbeda (24 kHz vs 22,05 kHz) tapi tidak jadi masalah: `_ke_wav()`
menulis header WAV, dan pemutarnya membaca laju dari header, bukan dari konstanta.

**Backend hotkey** — `pynput` (default) atau `keyboard`, dan mode `toggle`
atau `hold`.

**Gaya interaksi** — `SESSION_MODE=false` (pencet tiap giliran) atau `true`
(sekali pencet, terus ngobrol). Yang berganti cuma handler-nya; transkrip sampai
jawaban dipakai bareng lewat `_route_and_reply()`.

---

## 7. Yang belum ada

Semua di bawah ini **pernah ada** dan dicopot saat memulai ulang. Riwayat
git-nya lengkap — ini bukan daftar keinginan, tapi daftar yang tinggal diambil:

- **Kalender** — baca jadwal, buat acara (ICS lokal & Google Calendar)
- **Daftar tugas** — catat, tandai selesai, pilih mana yang dikerjakan
- **Memori antar-sesi** — riwayat & fakta yang bertahan setelah restart
- **Jawaban pasti tanpa LLM** — jam, tanggal, jadwal dihitung Python. Terukur
  5,7x lebih cepat sampai bunyi pertama, dan jam 6/6 tepat lawan 4/6–6/6 salah
- **Pengurai waktu deterministik** — `time_en` (33/33) dan `waktu_id`

Yang belum pernah ada:

- **Wake word** — masuk sesi tanpa menyentuh tombol
- **Memotong agent** — butuh peredam gema, lihat §4.9
- **Notifikasi proaktif** — agent bicara duluan

### Batasan yang diketahui pada qwen2.5:7b

Dua hal terukur, bukan dugaan.

**Jawaban salah meracuni giliran berikutnya.** Ditanya lokasi kelas dengan
riwayat kosong: 0 dari 8 salah. Tetapi begitu satu jawaban keliru masuk riwayat,
giliran berikutnya menyalin kekeliruan itu — model lebih memercayai ucapannya
sendiri daripada data di system prompt. Ini alasan riwayat tidak lagi disimpan
ke disk (§5), dan alasan `OLLAMA_NUM_CTX` dinaikkan ke 8192.

Bentuk paling mahalnya: model **mengaku telah melakukan sesuatu yang tidak
dilakukannya**. Ditanya *"can you delay that?"*, jawabannya *"the deadline is now
set for September 1st"* — padahal tidak ada satu pun jalur kode yang bisa
mengubah acara. Sekarang system prompt melarangnya secara eksplisit: agent tidak
punya alat apa pun, dan harus mengatakannya. Jawaban salah masih ketahuan saat
dicek; pengakuan palsu membuat orang berhenti mengecek.

**Balasannya lebih panjang dari yang diminta.** Lihat §4.10.

## 8. Cara mengukur ulang

Angka-angka di dokumen ini bisa usang kalau perangkat keras atau modelnya
berubah. Yang perlu diketahui untuk mengukur sendiri:

- **Akurasi STT** — nyalakan `SAVE_RECORDINGS=true`, pakai beberapa hari,
  lalu bandingkan transkrip dengan yang sebenarnya kamu ucapkan
- **VAD** — `python -m agent.vad`. Ucapkan kalimat berjeda di tengah: harus
  keluar sebagai SATU baris, bukan dua. Batuk tidak boleh muncul sama sekali
- **Jeda terasa** — yang penting waktu sampai **bunyi pertama**, bukan waktu
  total. Streaming nyaris tidak mengubah total, tapi memangkas diamnya 53–71%
- **VRAM** — `nvidia-smi`, atau `scripts/status.ps1` untuk ringkasannya
- **Offline** — `python -m agent.cek_offline`. Untuk bukti yang lebih keras,
  blokir semua soket non-localhost lalu jalankan rantai penuh; yang dihitung
  bukan "jalan", tapi **nol percobaan koneksi keluar**

Satu pelajaran yang berulang: **tes yang terlalu longgar menyembunyikan
kesalahan nyata.** Di proyek ini sudah dua kali tes lulus padahal jawabannya
salah, karena kriterianya cuma "ada jawabannya". Ukur nilainya, bukan
keberadaannya.
