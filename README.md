# Face Emotion Recognition with FaceNet (PyTorch)

Proyek ini adalah sistem deteksi emosi wajah *real-time* yang dibangun menggunakan PyTorch. Aplikasi ini menggunakan arsitektur **FaceNet (InceptionResnetV1)** yang telah di-*pre-train* pada dataset VGGFace2 sebagai *backbone* dan di-*fine-tune* untuk mengenali 7 emosi dasar manusia.

Tujuan utama proyek ini adalah membangun model yang *robust* dengan mengimplementasikan teknik-teknik *deep learning* modern seperti *data augmentation*, *Focal Loss*, dan strategi *learning rate* yang canggih untuk mencapai akurasi tinggi pada dataset yang tidak seimbang.

![Final Evaluation Metrics](Eval-Results/per_class_metrics.jpg)
*Contoh visualisasi metrik evaluasi per kelas yang dihasilkan oleh skrip training.*

---

## Fitur Utama
- **Deteksi 7 Emosi**: Mampu mengenali emosi `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, dan `surprise`.
- **Pre-trained Backbone**: Menggunakan **InceptionResnetV1** dari `facenet-pytorch` yang sudah dilatih pada jutaan wajah, memberikan ekstraksi fitur yang sangat baik.
- **Data Augmentation Ekstensif**: Menerapkan augmentasi kompleks (rotasi, translasi, scaling, shear, jitter warna, grayscale, blur, erasing) untuk meningkatkan generalisasi model.
- **Focal Loss & Class Weights**: Mengatasi masalah ketidakseimbangan kelas (imbalanced dataset) dengan **Focal Loss** dan **Class Weighting**.
- **Advanced Training Strategy**:
    - **Two-Stage Training**: Tahap 1 untuk *feature extraction* (melatih *head*) dan Tahap 2 untuk *fine-tuning* (melatih seluruh jaringan).
    - **Layer-wise Learning Rate Decay (LLRD)**: Menerapkan *learning rate* yang berbeda untuk setiap grup layer pada tahap *fine-tuning*, memungkinkan penyesuaian yang lebih halus.
- **Modern Learning Rate Scheduler**: Menggunakan **SequentialLR** yang mengkombinasikan *Linear Warmup* dan *Cosine Annealing* untuk stabilitas dan konvergensi yang lebih baik.
- **Inferensi Real-time**: Skrip `inference_realtime.py` untuk deteksi emosi langsung dari webcam.
- **Evaluasi Komprehensif**: Secara otomatis menghasilkan dan menyimpan plot *history training*, *confusion matrix*, dan grafik metrik per kelas (Precision, Recall, F1-Score).

---

## Arsitektur & Algoritma

- **Backbone**: `InceptionResnetV1` dari `facenet-pytorch`, diinisialisasi dengan bobot pre-trained `vggface2`.
- **Head Classifier**: Jaringan *fully-connected* kustom dengan beberapa lapisan `Linear`, `LeakyReLU`, `BatchNorm1d`, dan `Dropout` untuk mencegah *overfitting*.
- **Loss Function**: **Focal Loss** dengan *label smoothing* untuk menangani kelas yang sulit dan imbang, dikombinasikan dengan **Center Loss** untuk meningkatkan *intra-class compactness*.
- **Optimizer**: **AdamW**, versi perbaikan dari Adam yang memisahkan *weight decay* dari *gradient updates*.
- **Training Strategy**:
  1. **Stage 1 (Feature Extraction)**: *Backbone* dibekukan (*frozen*), hanya *head classifier* yang dilatih dengan *learning rate* lebih tinggi.
  2. **Stage 2 (Fine-Tuning)**: Seluruh model dilatih (*unfrozen*) dengan **LLRD**, di mana lapisan yang lebih dalam memiliki *learning rate* yang lebih kecil.
- **Deteksi Wajah**: Menggunakan *classifier* Haar Cascade (`haarcascade_frontalface_default.xml`) dari OpenCV untuk mendeteksi lokasi wajah pada gambar atau video *frame*.

---

## Tech Stack
- **Bahasa**: Python 3.8+
- **Framework**: PyTorch, Torchvision
- **Library Utama**:
  - `facenet-pytorch`: Untuk model backbone InceptionResnetV1.
  - `opencv-python`: Untuk pemrosesan gambar, deteksi wajah, dan input webcam.
  - `scikit-learn`: Untuk kalkulasi *class weights* dan *classification report*.
  - `numpy`: Untuk operasi numerik.
  - `matplotlib` & `seaborn`: Untuk visualisasi data dan hasil evaluasi.
  - `tqdm`: Untuk *progress bar* yang informatif saat training dan evaluasi.

---

## Struktur Proyek
Struktur direktori proyek adalah sebagai berikut:

```
Face-Recognition/
│
├── dataset/                # Dataset mentah (folder ini tidak dilacak oleh Git)
│   └── (subdirektori emosi dan gambar)
│
├── Models/                 # Menyimpan file model .pth yang telah dilatih
│   └── best_facenet_emotion_v2.pth   (dihasilkan saat training)
│
├── Eval-Results/           # Menyimpan hasil visualisasi evaluasi (otomatis dibuat)
│   ├── confusion_matrix_final.jpg
│   ├── per_class_metrics.jpg
│   └── training_history_final.jpg
│
├── notebooks/              # Jupyter Notebooks untuk eksperimen
│   └── train_facenet_emotion_v2_fixed.ipynb
│
├── src/                    # Kode sumber utama
│   ├── config.py           # File konfigurasi terpusat (path, parameter training)
│   ├── split_dataset.py    # Skrip untuk membagi dataset menjadi train/val
│   └── train_model_pytorch.py # Skrip utama untuk pipeline training
│
├── .gitignore              # Mengabaikan file yang tidak perlu dilacak
├── README.md               # Dokumentasi proyek (file ini)
├── inference_realtime.py   # Skrip untuk menjalankan deteksi emosi dari webcam
└── requirements.txt        # Daftar dependensi Python
```

---

## Instalasi

1. **Clone Repositori**
   ```bash
   git clone https://github.com/USERNAME/REPO_NAME.git
   cd Face-Recognition
   ```

2. **Buat dan Aktifkan Virtual Environment** (direkomendasikan)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependensi**
   Pastikan Anda memiliki **PyTorch** dengan dukungan CUDA jika Anda memiliki GPU. Kunjungi [situs resmi PyTorch](https://pytorch.org/get-started/locally/) untuk instruksi instalasi yang sesuai dengan sistem Anda. Setelah itu, install dependensi lainnya:
   ```bash
   pip install -r requirements.txt
   ```

---

## Panduan Penggunaan

### 1. Persiapan Dataset
Dataset yang digunakan dalam proyek ini diharapkan memiliki struktur sebagai berikut (format `ImageFolder` standar PyTorch):

```
<dataset_root>/
├── angry/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── disgust/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── fear/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── happy/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── neutral/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── sad/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
└── surprise/
    ├── image_0001.jpg
    ├── image_0002.jpg
    └── ...
```

Disarankan menggunakan dataset emosi wajah yang umum seperti **FER2013**, **AffectNet**, atau dataset lain yang menyediakan label 7 emosi dasar (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`). Pastikan nama folder kelas sesuai dengan `CLASS_NAMES` di `src/config.py`.

### 2. Konfigurasi
Buka file `src/config.py` dan sesuaikan path dan parameter berikut:

- `DATASET_DIR`: Path **absolut** ke direktori *root* dataset Anda (contoh di atas: `<dataset_root>`).
- `SPLIT_DIR`: Direktori output untuk data yang sudah dibagi (akan dibuat otomatis).
- `EPOCHS`, `BATCH_SIZE`, `IMG_SIZE`: Parameter utama untuk proses training.

### 3. Training Model
Setelah konfigurasi selesai, jalankan skrip berikut untuk membagi dataset dan memulai training:

1. **Bagi Dataset** (hanya perlu dijalankan sekali)
   ```bash
   python src/split_dataset.py
   ```
   Skrip ini akan membuat folder di lokasi `SPLIT_DIR` yang berisi subfolder `train` dan `val`.

2. **Mulai Training**
   ```bash
   python src/train_model_pytorch.py
   ```
   Proses ini akan:
   - Memuat data dari `SPLIT_DIR`.
   - Melakukan dua tahap training.
   - Menyimpan model terbaik (`best_facenet_emotion_v2.pth`) dan model final (`emotion_facenet_v2_final.pth`) di dalam folder `Models/`.
   - Menyimpan semua grafik evaluasi di dalam folder `Eval-Results/`.

### 4. Inferensi Real-time
Untuk mencoba model secara langsung menggunakan webcam Anda, jalankan skrip `inference_realtime.py`. Pastikan file model (`best_facenet_emotion_v2.pth` atau `emotion_facenet_v2_final.pth`) sudah ada di folder `Models/`.

```bash
python inference_realtime.py
```
- Tekan tombol **'q'** untuk keluar dari aplikasi.

---

## Hasil dan Evaluasi
Selama dan setelah training, semua hasil evaluasi akan disimpan di folder `Eval-Results/`. Ini termasuk:
- **`training_history_final.jpg`**: Grafik yang menunjukkan progres *loss* dan *accuracy* untuk data training dan validasi, serta perubahan *learning rate* setiap epoch.
- **`confusion_matrix_final.jpg`**: Matriks yang memvisualisasikan performa model dalam membedakan antar kelas.
- **`per_class_metrics.jpg`**: Grafik batang yang merangkum *Precision*, *Recall*, dan *F1-Score* untuk setiap kelas emosi, memberikan gambaran detail tentang di mana model berkinerja baik atau buruk.
- **Laporan Klasifikasi**: Dicetak di terminal pada akhir training, memberikan metrik numerik yang presisi.

---

## Kontribusi
Kontribusi dalam bentuk *pull request* atau *issue* sangat diterima. Jika Anda ingin berkontribusi, silakan *fork* repositori ini dan buat *pull request* dengan penjelasan yang jelas tentang perubahan yang Anda buat.
