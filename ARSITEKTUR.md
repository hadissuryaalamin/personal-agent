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
| `vad.py` | 226 | Deteksi suara buat mode sesi (Silero, per frame) |
| `cek_offline.py` | 128 | Verifikasi kesiapan jalan tanpa jaringan |
| `kalender_lokal.py` | 102 | Kalender file `.ics` (baca + tulis) |
| `jawab_pasti.py` | 332 | Jawaban jam/tanggal/jadwal tanpa LLM |
| `teks.py` | 55 | Penormal teks buat semua pencocokan niat |

### Arah ketergantungan

```
main ──▶ audio, vad, stt, tts, llm, calendar, tugas, jadwal_baru, kalender_lokal, gcal, teks
llm  ──▶ calendar, memory, tugas
calendar ──▶ gcal, kalender_lokal
jadwal_baru, tugas ──▶ teks, time_en
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

5. Jawaban → Kokoro → speaker, KALIMAT PER KALIMAT (§4.20)
6. Latar belakang: simpan riwayat (+ saring fakta kalau dinyalakan)
```

Dalam **mode sesi** (`SESSION_MODE=true`), langkah 1–2 diganti: hotkey membuka
sesi sekali, lalu batas tiap ucapan ditentukan VAD. Langkah 3–6 sama persis —
percabangan niatnya dipakai bareng lewat `_route_and_reply()`.

```
1. Hotkey ditekan → sesi terbuka
2. Ulang sampai tutup:
   └─ tunggu suara, rekam sampai diam 800 ms
   └─ langkah 3-6 di atas
   └─ mic DITUTUP selama agent bicara (§4.18)
3. Tutup kalau: hotkey ditekan lagi, diam 30 detik, atau "goodbye"
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

### 4.18 Mode sesi: mic ditutup selama agent bicara

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
`REPLY_MAX_WORDS` menjadi wajib, bukan pemanis — lihat §4.19.

VAD-nya juga membedakan **tiga** alasan berhenti, bukan satu. "User menutup
sesi", "sesi mati sendiri karena sepi", dan "suaranya terlalu pendek" harus
terpisah: kalau ketiganya dibaca sama, batuk sekali akan menutup sesi.
Suara terlalu pendek ditelan di dalam VAD dan tidak pernah sampai ke pemanggil.

Tenggat sepi dihitung dari awal dan **tidak** di-reset oleh suara pendek. Kalau
di-reset, ruangan berisik bisa menahan sesi terbuka selamanya tanpa kamu bicara
sekali pun — dan model ikut tertahan di memori selama itu.

### 4.19 Panjang balasan: batas kata, bukan batas token

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

### 4.20 Streaming: yang dipangkas diamnya, bukan totalnya

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

### 4.21 Satu penormal teks untuk semua pencocokan niat

Tiap modul dulu menormalkan teksnya sendiri, dan semuanya mengganti apostrof
dengan **spasi**. Akibatnya `"don't save it"` pecah menjadi `don | t | save |
it`: tidak ada yang cocok dengan daftar penolakan, lalu kata `save` tertangkap
sebagai persetujuan.

Jadi **`answer_yes("don't save it")` mengembalikan `True`** — agent menyimpan
acara padahal user bilang jangan, persis kebalikan dari yang diminta, dan
persis kegagalan yang seluruh mekanisme konfirmasi ini ada untuk mencegahnya.

`teks.py` menjadi satu-satunya sumber: apostrof **dibuang** (menyambung), tanda
baca lain menjadi spasi. Ini kelas bug yang sama dengan frasa dua kata di
`_YES`/`_NO` yang tidak pernah cocok — dan itu alasan normalisasi tidak boleh
ditulis ulang per modul.

Pencocok frasa penutup sesi punya **dua tingkat** karena alasan serupa. "I'm
done" menutup sesi, tapi "I'm done with assignment one" adalah laporan tugas
selesai — kalau dicocokkan di mana pun, tugasmu tidak pernah tertandai karena
sesinya keburu tutup.

### 4.22 Pertanyaan tertutup dijawab Python, bukan model

Jam, tanggal, dan jadwal punya jawaban yang **sudah pasti**. Menyerahkannya ke
qwen bukan cuma menambah 3-12 detik, tapi menambah cara untuk salah.

| | Jalur pasti | Lewat model |
|---|---:|---:|
| Bunyi pertama, rata-rata | **1,01 dtk** | 5,78 dtk |
| Ketepatan jam | **6/6** | 4/6 sampai 6/6 salah |

Kegagalan modelnya menarik karena bukan soal akses. Jamnya **ada** di prompt dan
terbaca benar — yang salah cara dia menyampaikannya: dia memparafrase jadi
ucapan yang "enak" lalu **membulatkan**. Jam 20:12 menjadi "quarter past eight",
"half past eight", "eight fifteen".

Hipotesis pertama — beri format yang lebih ramah ucapan (`8:12 pm`) — **diuji
dan gagal**: 6/6 salah, lebih buruk daripada format 24 jam. Formatnya justru
mengundang pembulatan. Itu menutup opsi menambal prompt: selama kalimatnya
disusun model, jamnya tidak akan pernah akurat.

Dua hal yang bikin ini aman:

**`answer()` balikin `None` kalau ragu.** Pertanyaan tak dikenal jatuh ke model.
Salah rute di sini membuat pertanyaan wajar dijawab kaku dan melenceng — lebih
buruk daripada lambat. Diuji 12 kalimat yang harus lolos ke model, termasuk
jebakan seperti *"is the library open today"* (ada "today", bukan soal jadwal)
dan *"what do you think about my schedule"* (ada "schedule", butuh penilaian).

**Jadwal dikenali dari dua bagian, bukan hafalan frasa.** Kata jadwal + kata
hari. Daftar frasa utuh terlalu kaku: *"what classes do I have today"* tidak
mengandung `"classes today"` karena ada `"do i have"` menyelip.

Tiap acara jadi **kalimat sendiri**, bukan satu kalimat panjang berkoma — dan
itu bukan kosmetik. TTS baru mulai berbunyi setelah satu kalimat utuh, jadi
jawaban pasti sempat **lebih lambat** dari model (6,26 lawan 5,58 detik) justru
karena dirakit sebagai satu kalimat. Setelah dipecah: 0,49 detik.

### 4.23 Pertanyaan yang menyelip jadi tulisan

Tiga entri sampah pernah masuk daftar tugas — `Assignment Count for August`,
`Assignments Due This Month` (dua kali). Semuanya lahir dari **pertanyaan** yang
terbaca sebagai perintah mencatat.

Pemicunya *"I need to know my assignments this month"*. `"i need to"` ada di
daftar deklarasi tugas, dan penjaga pertanyaan cuma memeriksa **awal** kalimat —
sedangkan "I need to **know**" itu bertanya.

Penjaga awal-kalimat saja tidak cukup; sekarang ada daftar penanda tanya yang
dicek **di mana pun** dalam kalimat: `need to know`, `want to know`,
`wondering`, `tell me about`. Arahnya sengaja tidak simetris — gagal mengenali
tugas berarti kamu mengulang sekali, sedangkan salah menulis meninggalkan entri
palsu yang baru ketahuan berminggu-minggu kemudian.

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

**Gaya interaksi** — `SESSION_MODE=false` (pencet tiap giliran) atau `true`
(sekali pencet, terus ngobrol). Yang berganti cuma handler-nya; percabangan
niat, memori, dan kalender dipakai bareng lewat `_route_and_reply()`.

---

## 7. Yang belum ada

- **Wake word** — masuk sesi tanpa menyentuh tombol sama sekali
- **Memotong agent** — butuh peredam gema, lihat §4.18
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
