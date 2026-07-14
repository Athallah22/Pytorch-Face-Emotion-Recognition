import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# [Wajib] Hubungkan ke library FaceNet PyTorch
from facenet_pytorch import InceptionResnetV1

# Impor konfigurasi terpusat
from config import (
    BASE_DIR, TRAIN_DIR, VAL_DIR, CHECKPOINT_PATH, FINAL_MODEL_PATH,
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS, CLASS_NAMES, # Keep EPOCHS, it's used
    HISTORY_PLOT_PATH, CONFUSION_MATRIX_PATH, METRICS_PLOT_PATH, OUTPUTS_DIR
)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ==================== DEVICE ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== AUGMENTASI (lebih kaya dari v1) ====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.12, 0.12),
        scale=(0.85, 1.15),
        shear=(-8, 8)
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.10),          
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.20, scale=(0.02, 0.08))  
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)          
        focal_loss  = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==================== CENTER LOSS (opsional, aktifkan di config) ====================
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size    = features.size(0)
        dist_mat      = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.centers.size(0)) + \
                        torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.centers.size(0), batch_size).t()
        dist_mat.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        classes       = torch.arange(self.centers.size(0)).long().to(features.device)
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.centers.size(0))
        mask          = labels_expand.eq(classes.expand(batch_size, self.centers.size(0)))
        dist          = dist_mat * mask.float()
        loss          = dist.clamp(min=1e-12).sum() / batch_size
        return loss

# ==================== MODEL ====================
class FaceEmotionFaceNetModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2')
        in_features   = 512

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

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)   
        return self.head(features), features  

# ==================== LAYER-WISE LR DECAY ====================
def get_llrd_params(model, base_lr=1e-5, decay_factor=0.65):
    """
    Membangun param_groups dengan lr yang menurun makin ke dalam backbone.
    - Head           : base_lr * 10
    - Backbone layer N (terdekat output) : base_lr * decay_factor^0
    - Backbone layer N-1                 : base_lr * decay_factor^1
    - ...
    - Backbone layer 0 (terdekat input)  : base_lr * decay_factor^(N-1)
    """
    param_groups = []

    param_groups.append({
        'params': list(model.head.parameters()),
        'lr': base_lr * 10,
        'name': 'head'
    })

    backbone_children = list(model.backbone.named_children())

    for i, (name, layer) in enumerate(reversed(backbone_children)):
        layer_lr = base_lr * (decay_factor ** i)
        params   = [p for p in layer.parameters() if p.requires_grad]
        if params:
            param_groups.append({
                'params': params,
                'lr': layer_lr,
                'name': f'backbone_{name}'
            })

    return param_groups

# ==================== TRAINING HELPERS ====================
def set_bn_eval(m):
    if m.__class__.__name__.find('BatchNorm') != -1:
        m.eval()

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, min_delta=1e-4, mode='min', checkpoint_path=None):
        self.patience        = patience
        self.verbose         = verbose
        self.min_delta       = min_delta
        self.counter         = 0
        self.best_score      = None
        self.early_stop      = False
        self.best_weights    = None
        self.mode            = mode
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_metric, model):
        score = -val_metric if self.mode == 'max' else val_metric

        if self.best_score is None:
            self.best_score   = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self._save_checkpoint()
        elif score > self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (best={self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score   = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter      = 0
            self._save_checkpoint()

    def _save_checkpoint(self):
        if self.checkpoint_path is not None:
            torch.save(self.best_weights, self.checkpoint_path)

def train_one_epoch(model, loader, optimizer, focal_loss, center_loss, center_optimizer, scaler, device, use_center_loss=False, center_loss_weight=0.003, freeze_bn=False):
    """
    Satu epoch training dengan:
    - Mixed Precision (AMP) jika CUDA tersedia
    - Gradient Clipping
    - Focal Loss + Center Loss (opsional)
    - Freeze BN pada backbone saat fine-tuning
    """
    model.train()
    if freeze_bn:
        model.backbone.apply(set_bn_eval)

    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc="  Train", leave=False)

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_center_loss and center_optimizer is not None:
            center_optimizer.zero_grad()

        if device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                logits, features = model(inputs)
                loss = focal_loss(logits, labels)
                if use_center_loss:
                    loss += center_loss_weight * center_loss(features, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, features = model(inputs)
            loss = focal_loss(logits, labels)
            if use_center_loss:
                loss += center_loss_weight * center_loss(features, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if use_center_loss and center_optimizer is not None:
            if device.type == 'cuda':
                scaler.unscale_(center_optimizer)
            for param in center_loss.parameters():
                if param.grad is not None:
                    param.grad.data *= (1.0 / center_loss_weight)
            center_optimizer.step()

        if not torch.isnan(loss):
            running_loss += loss.item() * inputs.size(0)
            _, predicted  = logits.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader.dataset), 1), correct / max(total, 1)

def evaluate(model, loader, focal_loss, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="  Val  ", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _      = model(inputs)
            loss           = focal_loss(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted  = logits.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader.dataset), 1), correct / max(total, 1)

def train_model(model, train_loader, val_loader,
                optimizer, scheduler,
                focal_loss, center_loss, center_optimizer,
                start_epoch, max_epochs, patience,
                checkpoint_path,
                use_center_loss=False, center_loss_weight=0.003,
                freeze_bn=False):

    scaler        = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    early_stopping = EarlyStopping(patience=patience, verbose=True, mode='min', checkpoint_path=checkpoint_path)
    history       = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    epoch = start_epoch - 1

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, focal_loss, center_loss,
            center_optimizer, scaler, device,
            use_center_loss=use_center_loss,
            center_loss_weight=center_loss_weight,
            freeze_bn=freeze_bn
        )
        val_loss, val_acc = evaluate(model, val_loader, focal_loss, device)

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:>3}/{max_epochs} ({elapsed:.1f}s) | "
            f"loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
            f"lr: {current_lr:.2e}"
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        early_stopping(val_loss, model) 
        if early_stopping.early_stop:
            print("  >> Early stopping. Memuat bobot terbaik...")
            model.load_state_dict(early_stopping.best_weights)
            break

    return epoch + 1, history

# ==================== REPORTING & PLOTTING ====================
def plot_training_history(history, filename='training_history_final.jpg'):
    if not history['lr']:
        return

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Training Metrics — FaceNet v2', fontsize=16)

    # 1. Accuracy
    axes[0].plot(history['train_acc'], label='Train', color='steelblue', linewidth=2)
    axes[0].plot(history['val_acc'],   label='Val',   color='darkorange', linewidth=2)
    axes[0].set_title('Accuracy per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Loss
    axes[1].plot(history['train_loss'], label='Train', color='steelblue', linewidth=2)
    axes[1].plot(history['val_loss'],   label='Val',   color='darkorange', linewidth=2)
    axes[1].set_title('Loss per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Learning Rate
    axes[2].plot(history['lr'], label='LR', color='mediumpurple', linewidth=2)
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, format='jpg', bbox_inches='tight')
    plt.close()
    print(f"  >> Grafik training disimpan: {filename}")



def evaluate_and_plot_metrics(model, val_loader, device, class_names, filename='confusion_matrix_final.jpg', metrics_filename='per_class_metrics.jpg'):
    model.eval()
    all_preds, all_labels = [], []

    print("\nEvaluasi akhir pada validation set...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="  Final Eval"):
            inputs        = inputs.to(device)
            logits, _     = model(inputs)
            _, preds      = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- 1. Confusion Matrix (Normalized) ---
    cm            = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, format='jpg', bbox_inches='tight')
    plt.close()
    print(f"  >> Confusion matrix disimpan: {filename}")

    # --- 2. Classification Report ---
    print("\n" + "="*55)
    print(" " * 15 + "CLASSIFICATION REPORT")
    print("="*55)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("="*55)

    # --- 3. Per-Class Precision, Recall, F1 Bar Chart ---
    precisions = [report[c]['precision'] for c in class_names]
    recalls    = [report[c]['recall']    for c in class_names]
    f1_scores  = [report[c]['f1-score']  for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x,         recalls,    width, label='Recall',    color='darkorange')
    bars3 = ax.bar(x + width, f1_scores,  width, label='F1-Score',  color='seagreen')

    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision, Recall, and F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(metrics_filename, dpi=150, format='jpg', bbox_inches='tight')
    plt.close()
    print(f"  >> Per-class metrics disimpan: {metrics_filename}")


def main():
    """Fungsi utama untuk menjalankan pipeline training."""
    global CLASS_NAMES  

    CLASS_NAMES = [name for name in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, name))]
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "Models"), exist_ok=True)

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---------- Dataset ----------
    print("\nMemuat dataset...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

    num_workers = 0  

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=False)

    labels = train_dataset.targets
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Periksa dan sesuaikan CLASS_NAMES jika ada perbedaan
    if list(train_dataset.classes) != CLASS_NAMES:
        print(f"⚠️ Peringatan: Urutan kelas dataset ({train_dataset.classes}) berbeda dari config ({CLASS_NAMES}). Menggunakan urutan dari dataset.")
        CLASS_NAMES = list(train_dataset.classes)
    
    print(f"Kelas terdeteksi : {CLASS_NAMES}")
    print(f"Class weights    : {dict(zip(CLASS_NAMES, class_weights.round(3)))}")
    print(f"Train samples    : {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # ---------- Model ----------
    model = FaceEmotionFaceNetModel(num_classes=len(CLASS_NAMES)).to(device)

    # ---------- Loss Functions ----------
    focal_loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0, label_smoothing=0.1)
    USE_CENTER_LOSS = True
    CENTER_LOSS_WEIGHT   = 0.003
    center_loss_fn = CenterLoss(num_classes=len(CLASS_NAMES), feat_dim=512, device=device).to(device)
    center_optimizer_fn  = optim.SGD(center_loss_fn.parameters(), lr=0.5)

    # ==================== STAGE 1: FEATURE EXTRACTION ====================
    print("\n" + "="*55)
    print("STAGE 1 — Feature Extraction (backbone frozen)")
    print("="*55)

    for param in model.backbone.parameters():
        param.requires_grad = False

    STAGE1_EPOCHS = 30
    WARMUP_EPOCHS = 5
    COSINE_EPOCHS_1 = STAGE1_EPOCHS - WARMUP_EPOCHS
    optimizer_s1 = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_s1 = LinearLR(optimizer_s1, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_s1 = CosineAnnealingLR(optimizer_s1, T_max=COSINE_EPOCHS_1, eta_min=1e-6)
    scheduler_s1 = SequentialLR(optimizer_s1, schedulers=[warmup_s1, cosine_s1], milestones=[WARMUP_EPOCHS])

    completed_epochs, hist1 = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer_s1, scheduler=scheduler_s1,
        focal_loss=focal_loss_fn, center_loss=center_loss_fn,
        center_optimizer=center_optimizer_fn, start_epoch=0,
        max_epochs=STAGE1_EPOCHS, patience=10, checkpoint_path=CHECKPOINT_PATH,
        use_center_loss=USE_CENTER_LOSS, center_loss_weight=CENTER_LOSS_WEIGHT, freeze_bn=False
    )

    if os.path.exists(CHECKPOINT_PATH):
        print("Memuat bobot terbaik Stage 1...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # ==================== STAGE 2: FINE-TUNING DENGAN LLRD ====================
    print("\n" + "="*55)
    print("STAGE 2 — Fine-tuning dengan Layer-wise LR Decay")
    print("="*55)

    for param in model.parameters():
        param.requires_grad = True

    STAGE2_EPOCHS = EPOCHS - completed_epochs
    BASE_LR_S2 = 1e-5
    LLRD_DECAY = 0.65
    param_groups_s2  = get_llrd_params(model, base_lr=BASE_LR_S2, decay_factor=LLRD_DECAY)
    optimizer_s2 = optim.AdamW(param_groups_s2, weight_decay=1e-4)
    
    print("Layer-wise LR yang digunakan:")
    for pg in param_groups_s2:
        print(f"  {pg['name']:<30} lr = {pg['lr']:.2e}")

    warmup_s2 = LinearLR(optimizer_s2, start_factor=0.1, end_factor=1.0, total_iters=3)
    cosine_s2 = CosineAnnealingLR(optimizer_s2, T_max=max(STAGE2_EPOCHS - 3, 1), eta_min=1e-7)
    scheduler_s2 = SequentialLR(optimizer_s2, schedulers=[warmup_s2, cosine_s2], milestones=[3])

    _, hist2 = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer_s2, scheduler=scheduler_s2,
        focal_loss=focal_loss_fn, center_loss=center_loss_fn,
        center_optimizer=center_optimizer_fn, start_epoch=completed_epochs,
        max_epochs=EPOCHS, patience=10, checkpoint_path=CHECKPOINT_PATH,
        use_center_loss=USE_CENTER_LOSS, center_loss_weight=CENTER_LOSS_WEIGHT, freeze_bn=True
    )

    # ---------- Simpan model final ----------
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\nModel FaceNet v2 selesai dilatih dan disimpan sebagai {FINAL_MODEL_PATH}")

    # ---------- Visualisasi ----------
    full_history = {
        'train_acc':  hist1['train_acc']  + hist2['train_acc'],
        'train_loss': hist1['train_loss'] + hist2['train_loss'],
        'val_acc':    hist1['val_acc']    + hist2['val_acc'],
        'val_loss':   hist1['val_loss']   + hist2['val_loss'],
        'lr':         hist1['lr']         + hist2['lr'],
    }
    plot_training_history(full_history, filename=HISTORY_PLOT_PATH)
    evaluate_and_plot_metrics(model, val_loader, device, CLASS_NAMES, filename=CONFUSION_MATRIX_PATH, metrics_filename=METRICS_PLOT_PATH)

if __name__ == '__main__':
    main()