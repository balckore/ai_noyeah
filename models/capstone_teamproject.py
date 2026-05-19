import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ======================
# 1. 경로 설정
# ======================
train_dir = r"D:\20221089\deepfake_test\Dataset\Train"
val_dir = r"D:\20221089\deepfake_test\Dataset\Validation"
test_dir = r"D:\20221089\deepfake_test\Dataset\Test"

batch_size = 64
epochs = 20
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# 2. 데이터 전처리
# ======================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ======================
# 3. 데이터셋 로드
# ======================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# ======================
# 4. 모델 (ResNet-50)
# ======================
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ======================
# 5. Loss / Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ======================
# 6. 함수 정의
# ======================
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return total_loss, acc


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc

# ======================
# 7. 학습 루프
# ======================
best_val_acc = 0

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch()
    val_acc = evaluate(val_loader)

    print(f"[Epoch {epoch+1}/{epochs}] "
          f"Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc: {val_acc:.2f}%")

    # best 모델 저장 (val 기준)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

print("Training finished.")

# ======================
# 8. Test 평가 (마지막)
# ======================
# best 모델 불러오기
model.load_state_dict(torch.load("best_model.pth"))

test_acc = evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")