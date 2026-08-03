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
   ┌─────────┐                ┌──────────┐                ┌──────────┐
   │ audio   │  rekam ───────▶│   stt    │ suara→teks ───▶│   llm    │
   │ mic     │                │ Parakeet │                │  qwen2.5 │
   └─────────┘                └──────────┘                └────┬─────┘
        ▲                                                      │
        │                     ┌──────────┐                     │
        └──── speaker ────────│   tts    │◀── teks→suara ──────┘
                              │  Kokoro  │
                              └──────────┘
```

Tiga model berbeda untuk tiga tugas berbeda: **Parakeet mendengar** (CPU,
0 VRAM), **qwen2.5:7b berpikir** (GPU, ~5 GB), **Kokoro berbicara** (CPU,
0 VRAM). Cuma satu yang menempati GPU — lihat §4.15.

### Dua lapis, bukan satu

Yang jalan terus-menerus hanyalah **listener hotkey** — 28 MB RAM, 0,02% CPU,
nol VRAM. Semua yang berat (STT, LLM) baru dimuat saat dipakai dan dilepas
lagi saat nganggur. Ini disengaja: agent hidup 24 jam tapi dipakai beberapa
menit sehari, jadi menahan model di memori sepanjang hari itu pemborosan.

---

## 2. Peta modul

| Modul | Baris | Tanggung jawab |
|---|---:|---|
| `main.py` | 570 | Listener hotkey, orkestrasi pipeline, percabangan niat, logging |
| `calendar.py` | 457 | Susun agenda dari sumber yang aktif, format untuk prompt |
| `llm.py` | 396 | Dua backend otak (Ollama/Claude), susun system prompt |
| `jadwal_baru.py` | 277 | Ucapan → acara/tugas terstruktur, kalimat konfirmasi |
| `stt.py` | 275 | Parakeet / Whisper: muat, transkrip, lepas saat nganggur |
| `config.py` | 266 | Semua konstanta dari `.env` + penegakan mode offline |
| `tugas.py` | 222 | Daftar tugas: simpan, tandai selesai, ringkas |
| `gcal.py` | 217 | Google Calendar (baca + tulis, OAuth) |
| `audio.py` | 215 | Rekam mic, playback, beep |
| `time_en.py` | 195 | Urai frasa waktu Inggris **tanpa LLM** |
| `waktu_id.py` | 153 | Versi Indonesia — disimpan, tidak dipanggil |
| `tts.py` | 142 | Kokoro / Piper: teks → WAV |
| `memory.py` | 136 | Riwayat obrolan & fakta ke disk |
| `cek_offline.py` | 128 | Verifikasi kesiapan jalan tanpa jaringan |
| `kalender_lokal.py` | 102 | Kalender file `.ics` (baca + tulis) |

### Arah ketergantungan

```
main ──▶ audio, stt, tts, llm, calendar, tugas, jadwal_baru, kalender_lokal, gcal
llm  ──▶ calendar, memory, tugas
calendar ──▶ gcal, kalender_lokal
jadwal_baru ──▶ waktu_id
semuanya ──▶ config
```

Satu arah, tanpa siklus. `config` tidak bergantung pada apa pun; `main` tidak
diimpor siapa pun. Modul sumber data (`gcal`, `kalender_lokal`) diimpor
**di dalam fungsi** oleh `calendar.py` untuk menghindari siklus impor.

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

4. Percabangan niat — dicek berurutan, urutannya PENTING:
   a. "forget everything"     → hapus memori
   b. jawaban konfirmasi      → simpan/batalkan acara yang menunggu
   c. tugas selesai           → tandai selesai
   d. tambah tugas            → catat tugas
   e. bikin acara             → urai, bacakan ulang, tunggu konfirmasi
   f. selain itu              → kirim ke LLM

5. Jawaban → Kokoro → speaker
6. Latar belakang: simpan riwayat (+ saring fakta kalau dinyalakan)
```

### Kenapa urutan percabangan penting

*"add a task to finish the assignment on Friday"* juga cocok dengan pola
*"add ..."* untuk membuat acara. Kalau acara dicek lebih dulu, tugasmu berakhir sebagai acara
kalender. Karena itu tugas dicek **sebelum** acara.

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

### 4.7 Agenda berformat berkolom, dengan baris hitungan

Format padat tanpa pemisah membuat model mengambil nama matkul dari satu baris
tapi lokasi dari baris tetangga. Sekarang kolom dipisah `|`.

Tiga baris dihitung di Python, bukan diserahkan ke model:

```
NEXT UP: ...
TODAY (Monday August 3): 1 scheduled -> ...
TOMORROW (Tuesday August 4): 2 scheduled -> ...
```

Mencari acara terdekat dan menjumlahkan per hari itu penalaran lintas baris —
persis yang bikin model kecil meleset. Terukur pada qwen2.5:7b:

| | Jawaban lengkap |
|---|---|
| Tanpa baris hitungan | 0/3 (dua kali salah **hari**) |
| Dengan baris hitungan | 2/3 |
| + format jam diucapkan | **3/3** (setara Claude) |

### 4.8 Jam ditulis untuk diucapkan

`jam 9 sampai 11`, bukan `09:00-11:00`. Dua alasan:

- TTS mengeja angka berformat jam apa adanya — kalimat yang sama makan
  4,1 detik lawan 3,0 detik
- Model **menyalin format yang dilihatnya**: begitu baris hitungan memakai
  `09:00`, Claude pun ikut mengucapkan `09:00` padahal sebelumnya "jam sembilan"

Tetap 24 jam supaya "jam 2" tidak ambigu siang/malam.

### 4.9 Tanggal diurai tanpa LLM

`time_en.py` mengurai frasa waktu Inggris secara deterministik, dan hasilnya
**menimpa** jawaban model. Alasannya terukur:

| | qwen2.5:7b | Sonnet 5 |
|---|---|---|
| Tanggal diserahkan ke model | 2/10 | 10/10 |
| Tanggal diurai `time_en` | **5/5** | **5/5** |

Model kecil menjawab "besok" dengan lusa dan "hari Jumat" dengan Senin, dan
tetap begitu setelah diberi tabel tanggal siap pakai di prompt. Frasa waktu itu
himpunan tertutup dengan aturan kaku — lebih tepat dikerjakan kode. Model cukup
mengurus judul dan lokasi, yang memang butuh pemahaman bahasa.

`time_en` lolos 33/33 bentuk, termasuk `3 p.m.` — bentuk ternormalisasi yang
dikeluarkan Parakeet — plus `half past two`, `quarter to five`, `noon`,
`midnight`, dan jebakan 12 AM/PM.

Versi Indonesianya (`waktu_id.py`) sengaja **tidak dihapus**. Kalau suatu saat
kembali ke dua bahasa, yang perlu dibangun ulang hanya perutean bahasanya.

### 4.10 Menulis kalender selalu dikonfirmasi

STT punya WER ~8%. Untuk **pertanyaan**, salah dengar hanya membuat jawaban
ngawur dan langsung ketahuan. Untuk **penulisan**, salah dengar meninggalkan
acara palsu yang baru ketahuan minggu depan.

Tiga hal condong ke arah aman:

- Jawaban ragu dibaca **batal**, bukan simpan
- Kata "ya" hanya dihitung setuju kalau jadi kata pertama — *"hmm apa ya"*
  sempat terbaca sebagai persetujuan sebelum diperbaiki
- Kalau tanggal/jam tidak jelas, acaranya **ditolak**, bukan disimpan dengan
  tebakan

### 4.11 Prompt caching: yang stabil di depan, jam di belakang

Cache itu cocok-awalan — satu byte berbeda membatalkan semua yang di
belakangnya. Jam sekarang berubah tiap menit, jadi menaruhnya di depan berarti
fakta, jadwal, dan tugas ikut terbuang tiap menit. Terukur:

| Susunan | Menit 15:57 | Menit 15:58 |
|---|---|---|
| Jam di depan | 0 dari cache | 0 dari cache |
| **Jam di belakang** | **1.585 dari cache** | **1.585 dari cache** |

`_bagian_prompt()` memisahkan bagian stabil dari yang berubah.

### 4.12 Jendela agenda: 7 hari rinci + yang menyimpang dari pola

Isi agenda ikut dikirim di **setiap** permintaan, jadi jendelanya tidak bisa
asal dilebarkan. Tapi jadwal kuliah berulang mingguan — mengirim 90 hari secara
polos berarti membayar untuk mengulang informasi yang sama tujuh kali.

| Jendela | Token/permintaan |
|---|---|
| 7 hari | 1.477 |
| 90 hari polos | 5.417 |
| **7 hari + luar-pola** | **1.631** |

Di luar jendela rinci, hanya dikirim acara yang ciri hari+jam+judulnya belum
muncul — ujian, kelas pengganti, deadline, acara pribadi. 96% lebih murah dan
justru memunculkan yang berguna.

### 4.13 Memori menyimpan hal tentang user, bukan yang punya sumber lain

Versi awal prompt penyaring menyebut "jadwal rutin" sebagai contoh hal yang
layak diingat, dan `facts.md` langsung terisi jam kuliah lengkap dengan nomor
ruangan. Salinan seperti itu **beku**: kalau jadwal berubah, memorinya jadi
salah dan mulai **membantah** kalender di prompt yang sama.

Aturan pembatas: memori menyimpan hal tentang **user** yang tidak punya sumber
lain — nama, preferensi, proyek. Jadwal punya kalender, tugas punya daftarnya.

### 4.14 Login OAuth tidak boleh dipicu dari alur agent

Agent jalan lewat `pythonw` **tanpa jendela**. Kalau alur normal boleh memicu
login browser, agent akan menggantung menunggu jendela yang mungkin tidak
disadari user — gejalanya persis seperti mati. `gcal.aktif()` ikut memeriksa
keberadaan token, dan login dipisah ke `scripts/login_google.py`.

### 4.15 Hanya satu model yang menempati GPU

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

### 4.16 Mode offline ditegakkan saat startup, di `config`

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

### 4.17 Data lama tidak boleh membuat parser gagal

Judul acara yang ditulis sebelum pindah ke Inggris membawa label Indonesia —
`COMP4620 ... (kuliah)`. Setelah `KINDS` berganti ke Inggris, parser berhenti
mengenalinya: label itu bocor utuh ke prompt sebagai bagian dari nama matkul,
dan kolom jenisnya jadi kosong.

Perbaikannya dua lapis, dan keduanya perlu:

1. `LEGACY_KINDS` di `calendar.py` **menerjemahkan** label lama saat dibaca, jadi
   file lama tetap terbaca benar tanpa disentuh.
2. `scripts/migrasi_jenis_ics.py` menulis ulang datanya sekali (36 `kuliah`,
   20 `lab komputer`), dengan cadangan, dan **tidak menyentuh** judul buatan user
   sendiri — `Bayar Rego` itu isi, bukan label sistem.

`gcal.py` sempat punya daftar jenis kembar. Sekarang meminjam dari `calendar.py`:
daftar kembar berarti satu jalur mengenali label yang jalur lain tolak.

---

## 5. Penyimpanan

Semua di `memory/` (gitignore):

| File | Isi | Umur |
|---|---|---|
| `facts.md` | Hal tentang user | Sampai dihapus |
| `history.json` | 20 pesan terakhir | Dibuang setelah 12 jam |
| `tasks.json` | Daftar tugas | Sampai ditandai selesai |
| `kalender.ics` | Kalender lokal | Permanen |
| `google_token.json` | Token OAuth | Diperbarui otomatis |
| `calendar.ics` | Cache feed ICS | Disegarkan tiap 60 menit |

Penulisan lewat file sementara + `os.replace` — kalau proses mati di tengah
menulis, file lama tetap utuh.

**Riwayat sengaja punya batas umur.** Menyambungkan obrolan yang terpotong
restart berguna dalam hitungan jam; 20 pesan dari minggu lalu justru membuat
salah konteks. Fakta yang bertahan.

---

## 6. Bagian yang bisa ditukar

Lima titik dirancang bisa diganti lewat `.env` tanpa menyentuh kode:

**Otak** — `LLM_BACKEND=ollama|claude`. Keduanya mengimplementasikan antarmuka
sama (`chat()`, `_oneshot()`), jadi seluruh fitur jalan di dua-duanya.

**Pendengaran** — `STT_BACKEND=parakeet|whisper`. Keduanya di balik
`get_model()` / `transcribe()` / `unload_model()` yang sama. Parakeet cuma
Inggris dan tidak menerima prompt kosakata; Whisper multibahasa tapi menuntut
VRAM kalau dinaikkan ke GPU.

**Suara** — `TTS_BACKEND=kokoro|piper`, keduanya di balik `speak(text) -> WAV`.
Sample rate berbeda (24 kHz vs 22,05 kHz) tapi tidak jadi masalah: `_ke_wav()`
menulis header WAV, dan pemutarnya membaca laju dari header, bukan dari konstanta.

**Sumber kalender** — file `.ics` lokal, feed ICS jarak jauh, atau Google
Calendar. `calendar.py` menggabungkan yang aktif; kalau lebih dari satu berisi
acara yang sama, hasilnya dobel — karena itu hanya satu yang dinyalakan.

**Backend hotkey** — `pynput` (default) atau `keyboard`, dan mode `toggle`
atau `hold`.

---

## 7. Yang belum ada

- **Wake word** — panggil tanpa menyentuh tombol
- **Notifikasi proaktif** — agent bicara duluan, misalnya mengingatkan tenggat
- **Ubah/hapus acara lewat suara** — sekarang hanya bisa membuat
- **Integrasi LMS** — tenggat tugas masih dicatat manual
- **Dua bahasa** — `waktu_id.py` masih ada tapi tidak ada perutean bahasanya

### Batasan yang diketahui pada qwen2.5:7b

Dua hal terukur, bukan dugaan:

**Jawaban salah meracuni giliran berikutnya.** Ditanya lokasi kelas dengan
riwayat kosong: 0 dari 8 salah. Tetapi begitu satu jawaban keliru masuk riwayat
(`Engineering Lecture Theatre 2.04` padahal datanya `Fulton Muir, Rm 2.04`),
giliran berikutnya menyalin kekeliruan itu — model lebih memercayai ucapannya
sendiri daripada jadwal di system prompt. `"forget everything"` mengosongkannya.

Ini juga alasan `OLLAMA_NUM_CTX` dinaikkan ke 8192: pada default Ollama 4096,
system prompt (~850 token) plus riwayat yang menumpuk bisa menggeser justru
jadwalnya keluar konteks.

**Balasannya lebih panjang dari yang diminta.** System prompt meminta 1–2
kalimat; qwen kerap memberi 3–4, yang menjadi ~20 detik audio.
`OLLAMA_NUM_PREDICT=160` adalah batas atas untuk yang benar-benar mengigau,
bukan pemaksa ringkas. Claude menuruti batasan ini, qwen tidak.

### ISU TERBUKA: label hari sesekali meleset

**Status: belum diperbaiki, belum bisa direproduksi.** Dicatat di sini supaya
tidak hilang, bukan karena sudah dipahami.

Terjadi 4 Agustus 2026, 06:54. Ditanya *"Do I have assignment for this week?"*,
qwen menjawab:

> *"Your next classes are **tomorrow** from nine to eleven for COMP4620 ..."*

COMP4620 09:00–11:00 itu **hari itu juga**, bukan besok. Model mengambil isi
baris `TODAY` lalu melabelinya "tomorrow". Satu menit kemudian, pertanyaan lain
dijawab benar (*"Today is Tuesday 4 August ..."*).

Yang sudah dipastikan **bukan** penyebabnya:

- **Bukan cache agenda.** `agenda()` menghitung `hari_ini` dari `datetime.now()`
  setiap giliran; tidak ada teks prompt yang disimpan. Diverifikasi dengan
  membuang isi prompt saat itu juga — isinya `TODAY (Tuesday August 4)`, benar.
- **Bukan temperature.** 12 percobaan di 0,7 dan 12 di 0,2: 0 salah.
- **Bukan riwayat lintas tengah malam.** Hipotesis awalnya: kalimat "besok" yang
  diucapkan tanggal 3 menjadi salah setelah tanggal berganti, dan agent memang
  hidup terus sejak 20:23 tanggal 3. Diuji dengan menyuntikkan riwayat kemarin
  yang persis begitu — 5 percobaan, 0 salah.

Total 22 percobaan tanpa satu pun berhasil menirukan. Sesi live punya riwayat
yang tidak bisa direkonstruksi, jadi pemicunya masih terbuka.

**Arah perbaikan yang disarankan** — bukan menambal prompt, tapi mengikuti §4.9:
rutekan pertanyaan jadwal ke jawaban yang dihitung Python, sebagaimana tanggal
sudah diurai tanpa LLM. Baris `NEXT UP` / `TODAY` / `TOMORROW` sudah dihitung
Python; model tidak menambahkan apa pun di situ selain risiko. Untungnya
ganda: benar 100%, dan hilang satu panggilan LLM (~5 detik).

---

## 8. Cara mengukur ulang

Angka-angka di dokumen ini bisa usang kalau perangkat keras atau modelnya
berubah. Yang perlu diketahui untuk mengukur sendiri:

- **Akurasi STT** — nyalakan `SAVE_RECORDINGS=true`, pakai beberapa hari,
  lalu bandingkan transkrip dengan yang sebenarnya kamu ucapkan
- **Akurasi jadwal** — bandingkan jawaban agent dengan data mentah dari
  sumbernya, bukan sekadar "ada jawabannya". Dua kali di proyek ini, tes yang
  terlalu longgar menyembunyikan kesalahan hari yang nyata
- **Token & cache** — log obrolan mencantumkan `cache: N baca, M tulis`.
  Kalau `baca` selalu 0, ada sesuatu yang berubah-ubah tersisip di depan prompt
- **VRAM** — `nvidia-smi`, atau `scripts/status.ps1` untuk ringkasannya
