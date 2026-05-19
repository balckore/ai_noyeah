import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score
import os
from tqdm import tqdm

# [1. 설정값 고정]
BATCH_SIZE = 32
IMAGE_SIZE = 256
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "D:/db/archive_(1)/Dataset"

# [2. 전처리 설정]
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# [3. 데이터 로더 구축]
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_PATH, 'Train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(DATA_PATH, 'Validation'), data_transforms['val_test']),
    'test': datasets.ImageFolder(os.path.join(DATA_PATH, 'Test'), data_transforms['val_test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
}


# [4. 모델 생성]
def get_model():
    model = timm.create_model('tf_efficientnet_b4', pretrained=True)
    n_features = model.classifier.in_features
    model.classifier = nn.Linear(n_features, 1)
    return model.to(DEVICE)


model = get_model()

# --- RECALL 향상을 위한 핵심 설정 ---
# pos_weight=1.5: 가짜(Positive)를 틀렸을 때 벌점을 1.5배 더 줍니다.
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=LR)

# [5. 학습 및 검증 루프]
best_recall = 0.0


def train_model(epochs=20):
    global best_recall
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            pbar = tqdm(dataloaders[phase], unit="batch")
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().view(-1, 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # 기본 예측 (0.5 기준)
                    preds = torch.sigmoid(outputs) > 0.5

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_description(f"{phase.upper()} Phase")

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_recall = recall_score(all_labels, all_preds)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f}')

            # Recall이 가장 높은 모델을 저장 (가짜 탐지 우선 전략)
            if phase == 'val' and epoch_recall > best_recall:
                best_recall = epoch_recall
                torch.save(model.state_dict(), 'best_recall_model.pth')
                print(f"★ 신규 베스트 Recall 모델 저장 완료! (Recall: {epoch_recall:.4f})")


# [6. 최종 테스트 및 임계값 튜닝]
def final_test():
    if os.path.exists('best_recall_model.pth'):
        model.load_state_dict(torch.load('best_recall_model.pth'))
        model.eval()
    else:
        print("가중치 파일이 없습니다. 학습을 먼저 진행하세요.")
        return

    all_probs = []
    all_labels = []

    print("\n--- 최종 테스트 시작 (Test Set) ---")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test']):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().view(-1, 1)
            outputs = model(inputs)
            # 확률값으로 추출
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 임계값 조정 (가짜 탐지율 극대화) ---
    # 0.5 대신 0.3을 적용하여 가짜를 더 공격적으로 잡아냅니다.
    threshold = 0.3
    all_preds = [1 if p > threshold else 0 for p in all_probs]

    print(f"\n--- 최종 결과 (임계값 {threshold} 적용 시) ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Recall (가짜 판별력): {recall_score(all_labels, all_preds):.4f}")
    print(f"F1-Score: {f1_score(all_labels, all_preds):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == '__main__':
    # torch.cuda.empty_cache() # 메모리 정리 필요 시 주석 해제
    train_model(epochs=10)  # 이미 수치가 높으므로 10에폭만 더 돌려봐도 충분합니다.
    final_test()