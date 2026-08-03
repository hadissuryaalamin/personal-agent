# Arsitektur personal-agent

Dokumen ini menjelaskan **bagaimana** sistem ini bekerja dan **kenapa** dibangun
begini. Untuk cara memakai dan menyetel, lihat [README.md](README.md).

Sebagian besar keputusan di sini datang dari pengukuran, bukan preferensi.
Angkanya disertakan supaya bisa ditinjau ulang kalau keadaan berubah.

---

## 1. Gambaran besar

Asisten suara yang jalan diam di latar belakang Windows. Tekan hotkey, ngomong,
tekan lagi — dia menjawab lewat speaker. Semua Bahasa Indonesia.

```
                    ┌──────────────────────────────────────┐
   tekan hotkey ───▶│  main.py — listener & orkestrator    │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   ┌─────────┐                ┌──────────┐                ┌──────────┐
   │ audio   │  rekam ───────▶│   stt    │ suara→teks ───▶│   llm    │
   │ mic     │                │ Whisper  │                │  otak    │
   └─────────┘                └──────────┘                └────┬─────┘
        ▲                                                      │
        │                     ┌──────────┐                     │
        └──── speaker ────────│   tts    │◀── teks→suara ──────┘
                              │  Piper   │
                              └──────────┘
```

Tiga model berbeda untuk tiga tugas berbeda: **Whisper mendengar**, **LLM
berpikir**, **Piper berbicara**.

### Dua lapis, bukan satu

Yang jalan terus-menerus hanyalah **listener hotkey** — 28 MB RAM, 0,02% CPU,
nol VRAM. Semua yang berat (Whisper, LLM) baru dimuat saat dipakai dan dilepas
lagi saat nganggur. Ini disengaja: agent hidup 24 jam tapi dipakai beberapa
menit sehari, jadi menahan model di memori sepanjang hari itu pemborosan.

---

## 2. Peta modul

| Modul | Baris | Tanggung jawab |
|---|---:|---|
| `main.py` | 453 | Listener hotkey, orkestrasi pipeline, percabangan niat, logging |
| `calendar.py` | 356 | Susun agenda dari sumber ICS/Google, format untuk prompt |
| `llm.py` | 320 | Dua backend otak (Claude/Ollama), susun system prompt |
| `jadwal_baru.py` | 233 | Ucapan → acara/tugas terstruktur, kalimat konfirmasi |
| `gcal.py` | 173 | Google Calendar (baca + tulis, OAuth) |
| `stt.py` | 168 | Whisper: muat, transkrip, lepas saat nganggur |
| `audio.py` | 159 | Rekam mic, playback, beep |
| `tugas.py` | 153 | Daftar tugas: simpan, tandai selesai, ringkas |
| `config.py` | 151 | Semua konstanta, dibaca dari `.env` |
| `waktu_id.py` | 126 | Urai frasa waktu Indonesia **tanpa LLM** |
| `memory.py` | 104 | Riwayat obrolan & fakta ke disk |
| `kalender_lokal.py` | 78 | Kalender file `.ics` (baca + tulis) |
| `tts.py` | 50 | Piper: teks → WAV |

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

3. Transkrip (Whisper)
   └─ jika model belum siap: ucapkan "sebentar ya, lagi nyiapin"

4. Percabangan niat — dicek berurutan, urutannya PENTING:
   a. "lupakan semua"        → hapus memori
   b. jawaban konfirmasi      → simpan/batalkan acara yang menunggu
   c. tugas selesai           → tandai selesai
   d. tambah tugas            → catat tugas
   e. bikin acara             → urai, bacakan ulang, tunggu konfirmasi
   f. selain itu              → kirim ke LLM

5. Jawaban → Piper → speaker
6. Latar belakang: simpan riwayat (+ saring fakta kalau dinyalakan)
```

### Kenapa urutan percabangan penting

*"catat tugas assignment Jumat"* juga cocok dengan pola *"catat ..."* untuk
membuat acara. Kalau acara dicek lebih dulu, tugasmu berakhir sebagai acara
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
2. Agent mengucapkan *"sebentar ya"* supaya diamnya tidak terasa seperti mati
3. Penekanan ulang dijawab dua ketuk pendek

### 4.7 Agenda berformat berkolom, dengan baris hitungan

Format padat tanpa pemisah membuat model mengambil nama matkul dari satu baris
tapi lokasi dari baris tetangga. Sekarang kolom dipisah `|`.

Tiga baris dihitung di Python, bukan diserahkan ke model:

```
KELAS BERIKUTNYA: ...
HARI INI (Senin 3 Agustus): 1 jadwal -> ...
BESOK (Selasa 4 Agustus): 2 jadwal -> ...
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

- Piper mengeja angka berformat jam apa adanya — kalimat yang sama makan
  4,1 detik lawan 3,0 detik
- Model **menyalin format yang dilihatnya**: begitu baris hitungan memakai
  `09:00`, Claude pun ikut mengucapkan `09:00` padahal sebelumnya "jam sembilan"

Tetap 24 jam supaya "jam 2" tidak ambigu siang/malam.

### 4.9 Tanggal diurai tanpa LLM

`waktu_id.py` mengurai frasa waktu Indonesia secara deterministik, dan hasilnya
**menimpa** jawaban model. Alasannya terukur:

| | qwen2.5:7b | Sonnet 5 |
|---|---|---|
| Tanggal diserahkan ke model | 2/10 | 10/10 |
| Tanggal diurai `waktu_id` | **5/5** | **5/5** |

Model kecil menjawab "besok" dengan lusa dan "hari Jumat" dengan Senin, dan
tetap begitu setelah diberi tabel tanggal siap pakai di prompt. Frasa waktu itu
himpunan tertutup dengan aturan kaku — lebih tepat dikerjakan kode. Model cukup
mengurus judul dan lokasi, yang memang butuh pemahaman bahasa.

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

---

## 5. Penyimpanan

Semua di `memory/` (gitignore):

| File | Isi | Umur |
|---|---|---|
| `facts.md` | Hal tentang user | Sampai dihapus |
| `history.json` | 20 pesan terakhir | Dibuang setelah 12 jam |
| `tugas.json` | Daftar tugas | Sampai ditandai selesai |
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

Tiga titik dirancang bisa diganti lewat `.env` tanpa menyentuh kode:

**Otak** — `LLM_BACKEND=claude|ollama`. Keduanya mengimplementasikan antarmuka
sama (`chat()`, `_oneshot()`), jadi seluruh fitur jalan di dua-duanya.

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
