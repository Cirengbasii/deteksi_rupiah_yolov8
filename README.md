# deteksi_rupiah_yolov8
Proyek computer vision berbasis YOLOv8 untuk mendeteksi dan mengklasifikasi berbagai pecahan mata uang kertas Rupiah secara real-time.

---

# Deteksi Rupiah menggunakan YOLOv8

Proyek ini merupakan implementasi *computer vision* menggunakan algoritma **YOLOv8** (You Only Look Once) dari Ultralytics untuk mendeteksi dan mengklasifikasi berbagai nominal pecahan uang kertas Rupiah Indonesia. Sistem ini dirancang untuk dapat mengenali mata uang secara cepat dan akurat, baik melalui gambar, video, maupun *webcam* secara *real-time*.

## 🚀 Fitur
- **Deteksi Real-Time**: Mampu mendeteksi pecahan Rupiah dengan *frame rate* (FPS) yang tinggi.
- **Akurasi Tinggi**: Menggunakan arsitektur YOLOv8 yang telah di-tuning untuk objek spesifik (Uang Rupiah).
- **Multi-Klasifikasi**: Mendeteksi berbagai nominal pecahan (misal: Rp1.000, Rp2.000, Rp5.000, Rp10.000, Rp20.000, Rp50.000, Rp100.000).

## 🛠️ Prasyarat & Instalasi

Pastikan Anda sudah menginstal Python (disarankan versi 3.8 atau yang lebih baru).

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/Cirengbasii/deteksi_rupiah_yolov8.git](https://github.com/Cirengbasii/deteksi_rupiah_yolov8.git)
   cd deteksi_rupiah_yolov8



2. **Buat dan aktifkan Virtual Environment (Opsional tetapi disarankan):**
```bash
python -m venv venv
# Untuk Windows:
.\venv\Scripts\activate
# Untuk Linux/Mac:
source venv/bin/activate

```


3. **Instal *dependencies* yang diperlukan:**
```bash
pip install ultralytics opencv-python Progres

```



## 📂 Struktur Direktori (Rekomendasi)

```
├── dataset/               # Folder berisi data latih (images & labels)
├── weights/               # Tempat menyimpan file model terbaik (best.pt)
├── train.py               # Script untuk melatih model
├── detect.py              # Script untuk melakukan inferensi/deteksi
└── README.md              # Dokumentasi proyek

```

## 🏋️ Cara Pelatihan Model (Training)

Jika Anda ingin melatih ulang model dengan dataset Anda sendiri, jalankan perintah berikut:

```
from ultralytics import YOLO

# Memuat model pre-trained
model = YOLO('yolov8n.pt')

# Memulai pelatihan
model.train(data='path/to/data.yaml', epochs=50, imgsz=640, device=0)

```

## 🔍 Cara Menjalankan Deteksi (Inference)

Untuk melakukan uji coba deteksi menggunakan model yang sudah dilatih (`best.pt`):

```
# Deteksi menggunakan kamera / webcam
yolo predict model=weights/best.pt source=0 show=True

```

## 📝 Lisensi

Proyek ini dilisensikan di bawah [MIT License](https://www.google.com/search?q=LICENSE).

```

---

**Tips Tambahan:**
* Bagian `Cirengbasii` pada tautan *clone* di atas disesuaikan dengan username GitHub kamu yang terlihat di gambar.
* Jika nanti model kamu sudah selesai dilatih, kamu bisa menambahkan gambar atau GIF hasil deteksi uangnya di bagian bawah README agar terlihat semakin menarik!

```
