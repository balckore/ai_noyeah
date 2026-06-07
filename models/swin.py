import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm  # 진행률 게이지를 위한 라이브러리


def main():
    # 0. 하이퍼파라미터 및 경로 설정 (조건 엄수)
    DATA_DIR = "./Dataset"  # Train, Val, Test 폴더가 상위에 있는 루트 경로
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4  # 1x10^-4 고정
    EPOCHS = 20  # 20 에포크 구동

    # 4070 Super 활용을 위한 CUDA 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 활성화된 디바이스: {DEVICE} (RTX 4070 Super 가동 준비 완료)")

    # 1. 전처리 (Transforms) 설정 (조건 엄수)
    # 256x256 Resize 및 ImageNet 표준 정규화값 적용
    standard_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. 제공된 Train / Val / Test 폴더 구조 엄수하여 데이터셋 로드
    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'Train'), transform=standard_transform)
    val_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'Val'), transform=standard_transform)
    test_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'Test'), transform=standard_transform)

    # 7800X3D의 우수한 CPU 성능을 활용하기 위해 num_workers를 4 또는 8로 설정하면 데이터 로드 속도가 극대화됩니다.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"📦 데이터 로드 완료 | Train: {len(train_dataset)}장 | Val: {len(val_dataset)}장 | Test: {len(test_dataset)}장")

    # 3. Swin Transformer 모델 빌드 (256x256 입력 보간 자동 적용)
    print("🤖 Swin Transformer 모델 로드 중...")
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        num_classes=1,
        img_size=(256, 256)
    )
    model = model.to(DEVICE)

    # 4. 손실함수 및 최적화 함수 설정 (조건 엄수)
    criterion = nn.BCEWithLogitsLoss()  # 이진 분류 표준 Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # AdamW, 1e-4 고정

    # Best Model 저장을 위한 변수 초기화
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # 5. Training & Validation Loop (게이지 바 추가)
    print("\n🔥 본격적인 학습을 시작합니다. (Total Epochs: 20)")
    for epoch in range(EPOCHS):

        # --- Train Phase ---
        model.train()
        train_loss = 0.0

        # tqdm을 이용한 Train 진행률 게이지 생성
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Train", leave=False, dynamic_ncols=True)

        for images, labels in train_bar:
            images = images.to(DEVICE)
            # BCEWithLogitsLoss 규격에 맞게 [배치사이즈, 1] 형태의 float 차원 변환
            labels = labels.to(DEVICE).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            # 게이지 바 옆에 실시간 Loss 업데이트 표출
            train_bar.set_postfix(Loss=f"{loss.item():.4f}")

        epoch_train_loss = train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []

        # tqdm을 이용한 Validation 진행률 게이지 생성
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Valid", leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1).float()

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # BCEWithLogitsLoss의 로지트(0) 기준 이진 분류 판정
                preds = (outputs > 0).int()

                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(val_trues, val_preds)

        # 에포크가 끝날 때마다 결과 한 줄 요약 출력
        print(
            f"✅ Epoch [{epoch + 1}/{EPOCHS}] 완료 | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # 조건 반사 구문: Validation Accuracy가 가장 높았던 에포크의 가중치 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "best_swin_model.pth")
            print(f"  🌟 [BEST MODEL UPDATE] 최고 가중치 저장 완료 (Val Acc: {best_val_acc:.4f})\n")
        else:
            print("  - 가중치 유지\n")

    print("\n🎉 20 Epoch 학습이 모두 완료되었습니다! 최고 성능 가중치로 Test를 진행합니다.")

    # 6. 평가: 최종 측정 단계 (조건 엄수)
    model.load_state_dict(best_model_wts)
    model.eval()

    test_preds = []
    test_trues = []

    # Test 단계 게이지 바 생성
    test_bar = tqdm(test_loader, desc="Final Test Evaluation", dynamic_ncols=True)

    with torch.no_grad():
        for images, labels in test_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1).float()

            outputs = model(images)
            preds = (outputs > 0).int()

            test_preds.extend(preds.cpu().numpy())
            test_trues.extend(labels.cpu().numpy())

    # Test 데이터셋으로 Accuracy, F1-Score, Confusion Matrix 산출 (조건 엄수)
    test_accuracy = accuracy_score(test_trues, test_preds)
    test_f1 = f1_score(test_trues, test_preds, average='binary')
    test_cm = confusion_matrix(test_trues, test_preds)

    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("      🏆 [최종 TEST DATASET 평가 결과] 🏆      ")
    print("=" * 50)
    print(f"■ 최종 Accuracy (정확도) : {test_accuracy * 100:.2f}%")
    print(f"■ 최종 F1-Score         : {test_f1:.4f}")
    print("■ Confusion Matrix (오차행렬) :")
    print(test_cm)
    print("=" * 50)
    print("최종병기 Swin Transformer 구동 완료. 수고하셨습니다!")


if __name__ == '__main__':
    main()