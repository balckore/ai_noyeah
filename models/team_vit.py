import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import os
import time
from tqdm import tqdm  # 실시간 진행 바 라이브러리

# --- [1. 하이퍼파라미터 및 경로 설정] ---
DATA_DIR = './data/Dataset'  # 데이터셋 경로 (실제 경로에 맞게 수정)
IMG_SIZE = 256  # 팀 공통 규격
BATCH_SIZE = 64  # 4070 Super 최적값
LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_WORKERS = 8  # 7800X3D 최적값 (에러 시 4로 하향)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # GPU 인식 확인
    print(f"🚀 학습 시작 장치: {DEVICE}")
    if torch.cuda.is_available():
        print(f"🔥 사용 중인 GPU: {torch.cuda.get_device_name(0)}")

    # --- [2. 데이터 전처리 및 로더] ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # --- [3. 모델 설정: ViT-Small (256 사이즈 대응)] ---
    # img_size=IMG_SIZE를 넣어줘야 256 이미지 입력 시 에러가 나지 않습니다.
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=1, img_size=IMG_SIZE)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- [4. 학습 및 검증 루프] ---
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation Phase ---
        model.eval()
        val_correct = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE).float().view(-1, 1)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / len(val_set)
        epoch_time = time.time() - start_time

        print(f"\n✅ Epoch {epoch + 1} 결과: Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | 소요시간: {epoch_time:.2f}s ({epoch_time / 60:.1f}분)")

        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print("⭐ Best Model Saved!")

    # --- [5. 최종 테스트 평가] ---
    print("\n" + "=" * 30)
    print("🏁 최종 테스트 세트 평가 시작")
    print("=" * 30)

    model.load_state_dict(torch.load('best_vit_model.pth'))
    model.eval()

    y_true, y_pred = [], []
    test_pbar = tqdm(test_loader, desc="[Final Test]")

    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().view(-1, 1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    final_acc = (np.array(y_true) == np.array(y_pred)).mean()
    final_f1 = f1_score(y_true, y_pred)

    print(f"\n📊 최종 테스트 결과")
    print(f"- Accuracy: {final_acc:.4f}")
    print(f"- F1-Score: {final_f1:.4f}")
    print(f"- Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


if __name__ == '__main__':
    main()