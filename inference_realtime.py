import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# Import konfigurasi terpusat
from src.config import (
    MODEL_PATH, IMG_SIZE, CLASS_NAMES, HAAR_CASCADE_PATH, NUM_CLASSES
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== ARSITEKTUR MODEL (SESUAIKAN DENGAN TRAINING) ====================
class FaceEmotionFaceNetModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Backbone FaceNet (pretrained=None karena kita load manual weights)
        self.backbone = InceptionResnetV1(pretrained=None, classify=False) 
        in_features = 512
        
        # Head harus PERSIS sama dengan saat training:
        # 512 -> 512 -> 256 -> 128 -> num_classes
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.4),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.3),

            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def main():
    """Fungsi utama untuk menjalankan pipeline inference real-time."""
    print("🔄 Memuat Model FaceNet...")
    model = FaceEmotionFaceNetModel(num_classes=NUM_CLASSES).to(device)

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Bobot model sukses dimuat dari '{MODEL_PATH}' (Mode Strict)")
    else:
        raise FileNotFoundError(f"❌ File bobot {MODEL_PATH} tidak ditemukan! Pastikan posisinya benar.")

    model.eval()

    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"❌ Gagal memuat Haar Cascade Classifier dari: {HAAR_CASCADE_PATH}")

    print("📹 Membuka Kamera... Tekan 'q' untuk keluar.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal menangkap gambar dari webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
            
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            input_tensor = transform_pipeline(face_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            emotion_label = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item() * 100

            box_color = (0, 255, 0) if confidence_score > 60 else (0, 255, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            display_text = f"{emotion_label} ({confidence_score:.1f}%)"
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        cv2.imshow('Face Emotion Recognition - FaceNet End-to-End', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Program selesai ditutup.")

if __name__ == "__main__":
    main()
