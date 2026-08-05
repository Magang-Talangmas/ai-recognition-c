# AI-Recognition Attendance Engine (C Edition)

Sistem absensi dan pengenalan wajah real-time berbasis C (C11/C99) berperforma tinggi (*ultra low-latency*) yang dirancang khusus untuk skenario CCTV kantor dengan banyak karyawan yang duduk diam di meja kerja.

Repository GitHub: [https://github.com/Magang-Talangmas/ai-recognition-c.git](https://github.com/Magang-Talangmas/ai-recognition-c.git)

---

## 🌟 Fitur & Optimasi Real-Time (Core Innovations)

1. **Recognition Zone (ROI) Masking**:
   - Membatasi area deteksi wajah hanya pada koridor/pintu masuk/tangga (misal area kanan frame: `x1=0.45`, `x2=0.98`).
   - Wajah 4–6 karyawan yang duduk diam di meja kerja **tidak akan pernah diproses oleh ArcFace**, menghemat >80% beban komputasi CPU/GPU.
2. **Motion / Activity Gating**:
   - Melakukan deteksi selisih piksel cepat (*sub-sampled frame differencing*) pada zona ROI.
   - Jika tidak ada orang yang bergerak memasuki area pintu masuk, inferensi SCRFD & ArcFace dilewati (beban AI mendekati 0% saat ruangan statis).
3. **Face Quality Gate**:
   - **Size Gate**: Memfilter wajah terlalu kecil di bawah ambang batas (default: min 35px).
   - **Laplacian Blur Variance**: Memastikan ketajaman wajah sebelum ekstraksi embedding.
   - **5-Point Landmark Pose/Yaw Gate**: Menghitung rasio asimetri jarak mata ke hidung ($|d_L - d_R| / d_{eyes}$). Wajah yang menoleh tajam (*steep profile*) otomatis diabaikan.
4. **Candidate Lifecycle Table & Priority Round-Robin ArcFace Scheduler**:
   - Siklus hidup kandidat: `NEW` $\rightarrow$ `VOTING` $\rightarrow$ `CONFIRMED` $\rightarrow$ `COOLDOWN`.
   - Mengalokasikan kuota inferensi ArcFace yang ketat (default: maks 2 wajah per *tick*) menggunakan giliran *Round-Robin* yang adil agar FPS kamera tetap stabil tanpa *lag spike*.
5. **Temporal Majority Voting (3 dari 5 Vote)**:
   - Identitas karyawan diverifikasi melalui *rolling vote queue* (harus memperoleh 3 vote konsisten dari 5 frame berturut-turut).
6. **In-Memory Cooldown & Duplicate Suppression**:
   - Mencegah spam event ganda jika karyawan masih berdiri di depan kamera setelah absensi tercatat (default: cooldown 300 detik).
7. **Zero-Lag Mailbox Double Buffer & Async Dispatcher**:
   - Thread kamera RTSP membaca frame secara konstan dan membuang frame basi (*stale frames*).
   - Pencatatan event lokal ke SQLite (WAL mode) dan pengiriman HTTP POST ke backend `/api/v1/attendance` dilakukan di background worker thread non-blocking.

---

## 📂 Struktur Direktori

```text
ai-recognition-c/
├── CMakeLists.txt              # Konfigurasi Build CMake
├── config.yaml                 # File konfigurasi utama (ROI, Motion, Scheduler, Face, Backend)
├── .env.example                # Contoh variabel environment
├── .gitignore                  # File git ignore
├── README.md                   # Dokumentasi lengkap arsitektur
├── include/                    # Header file C modular
│   ├── attendance/
│   │   ├── config.h            # Struktur AppConfig, RoiConfig, MotionConfig, SchedulerConfig
│   │   ├── database.h          # SQLite3 CRUD & Event Lifecycle
│   │   ├── face_engine.h       # SCRFD, Laplacian blur, Umeyama Alignment, ArcFace
│   │   ├── matcher.h           # Cosine Similarity, Top-1/Top-2 Margin, In-Memory Cooldown
│   │   ├── vision_utils.h      # CandidateTable, Quality Gate, Round-Robin Scheduler, Crossing
│   │   └── api_dispatcher.h    # Asynchronous HTTP POST worker thread
│   └── third_party/
│       ├── cJSON.h             # JSON serialization
│       └── sqlite3.h           # Embedded database
├── src/                        # Source code implementasi C
│   ├── config.c
│   ├── database.c
│   ├── face_engine.c
│   ├── matcher.c
│   ├── vision_utils.c
│   ├── api_dispatcher.c
│   ├── run_camera.c            # Entry point streaming kamera real-time
│   ├── enroll_employee.c       # CLI pendaftaran template wajah karyawan
│   ├── respond_event.c         # CLI konfirmasi / respons event absensi
│   ├── export_embeddings.c     # CLI inspeksi database embedding binary
│   └── third_party/
│       ├── cJSON.c
│       └── sqlite3.c
└── tests/                      # Comprehensive Unit & Integration Test Suite
    └── test_all.c
```

---

## 🛠️ Panduan Build & Kompilasi

### Prasyarat
- C Compiler (MSVC, GCC, atau Clang)
- *(Opsional)* CMake 3.16+
- *(Opsional)* OpenCV C API, ONNX Runtime C API / OpenVINO C API

### Kompilasi Menggunakan CMake:
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Kompilasi Langsung Menggunakan GCC / Clang:
```bash
gcc -O2 -Iinclude -Isrc -o test_all tests/test_all.c src/config.c src/database.c src/face_engine.c src/matcher.c src/vision_utils.c src/third_party/sqlite3.c src/third_party/cJSON.c -lm -lpthread
./test_all
```

---

## 🚀 Penggunaan (CLI Commands)

### 1. Menjalankan Stream Kamera Real-Time
```bash
./run_camera config.yaml
```

### 2. Mendaftarkan Wajah Karyawan (Enrollment)
```bash
./enroll_employee <employee_id> <photo_folder> [data/embeddings.bin]
# Contoh:
./enroll_employee EMP-001 photos/emp001/ data/embeddings.bin
```

### 3. Mengonfirmasi atau Menolak Event Absensi (CLI)
```bash
# Menampilkan daftar 50 event terakhir:
./respond_event --list

# Mengonfirmasi event:
./respond_event 101 confirm

# Menolak event (Bukan saya / Not Me):
./respond_event 101 reject
```

### 4. Memeriksa Database Template Embedding
```bash
./export_embeddings data/embeddings.bin
```

---

## 📜 Lisensi & Kontributor
- **Organisasi**: PT. Talangmas Teknologi Indonesia / Magang Talangmas
- **Lisensi**: MIT
