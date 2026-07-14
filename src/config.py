import os
import cv2

# ==================== KONFIGURASI PATH ====================

# Base directory proyek
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 1. Path untuk Dataset
#    - DATASET_DIR: Lokasi dataset mentah Anda.
#      Ubah path ini jika Anda meletakkan dataset di tempat lain.
#    - SPLIT_DIR: Folder output untuk dataset yang sudah dibagi (train/val).
DATASET_DIR = "D:/Belajar Python/Data Set/Face/processed_data" # Path absolut ke data mentah
SPLIT_DIR   = os.path.join(BASE_DIR, "face_dataset_split")
TRAIN_DIR   = os.path.join(SPLIT_DIR, "train")
VAL_DIR     = os.path.join(SPLIT_DIR, "val")

# 2. Path untuk Model
#    - MODELS_DIR: Folder untuk menyimpan semua file model (.pth).
#    - CHECKPOINT_PATH: Bobot model dengan validation loss terbaik selama training.
#    - FINAL_MODEL_PATH: Bobot model final setelah seluruh proses training selesai.
MODELS_DIR          = os.path.join(BASE_DIR, "Models")
CHECKPOINT_PATH     = os.path.join(MODELS_DIR, "best_facenet_emotion_v2.pth")
FINAL_MODEL_PATH    = os.path.join(MODELS_DIR, "emotion_facenet_v2_final.pth")

# 3. Path untuk file pendukung
#    - HAAR_CASCADE_PATH: File XML untuk deteksi wajah.
HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# 4. Path untuk output visualisasi
#    - OUTPUTS_DIR: Folder untuk menyimpan grafik (confusion matrix, history).
OUTPUTS_DIR             = os.path.join(BASE_DIR, "Eval-Results")
HISTORY_PLOT_PATH       = os.path.join(OUTPUTS_DIR, "training_history_final.jpg")
CONFUSION_MATRIX_PATH   = os.path.join(OUTPUTS_DIR, "confusion_matrix_final.jpg")
METRICS_PLOT_PATH       = os.path.join(OUTPUTS_DIR, "per_class_metrics.jpg")

# ==================== KONFIGURASI TRAINING ====================
IMG_SIZE    = 160
BATCH_SIZE  = 32
NUM_CLASSES = 7
EPOCHS      = 60
TRAIN_RATIO = 0.8  # 80% data untuk training, 20% untuk validasi

# Urutan kelas harus sesuai dengan nama folder di dalam dataset Anda
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
